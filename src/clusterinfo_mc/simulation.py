"""Nested Monte Carlo runner for paired ClusterInfo/no-ClusterInfo trajectories.

Population uncertainty is drawn once per outer run. Each inner visitor passes
through conditional help-seeking, action-intent and action gates. Only a
help-seeker is modeled as a CH patient. This one-patient-per-help-seeker
assumption includes caregivers as proxies, rather than as additional patients.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import math
import platform
import random
from statistics import fmean, stdev
from typing import Any, Mapping

from .funnel import draw_funnel, quantile
from .model import SCENARIOS, ModelConfig, draw_population, simulate_patient


@dataclass(frozen=True)
class RunConfig:
    outer_runs: int = 40
    patients_per_run: int = 250
    seed: int = 20261028
    a2: float = 0.5
    monthly_unique_ids: int = 2400
    id_correction_low: float = 0.65
    id_correction_mode: float = 0.8
    id_correction_high: float = 0.95
    action_probability_low: float = 0.6
    action_probability_mode: float = 0.9
    action_probability_high: float = 1.0

    def __post_init__(self) -> None:
        for name in ("outer_runs", "patients_per_run", "seed", "monthly_unique_ids"):
            value = getattr(self, name)
            minimum = 1 if name in ("outer_runs", "patients_per_run") else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "a2", "id_correction_low", "id_correction_mode", "id_correction_high",
            "action_probability_low", "action_probability_mode", "action_probability_high",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite probability between 0 and 1")
        for stem in ("id_correction", "action_probability"):
            if not getattr(self, stem + "_low") <= getattr(self, stem + "_mode") <= getattr(self, stem + "_high"):
                raise ValueError(f"{stem} must satisfy low <= mode <= high")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RunConfig":
        if not isinstance(values, Mapping):
            raise ValueError("run configuration must be an object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown run configuration keys: {', '.join(sorted(map(str, unknown)))}")
        return cls(**values)


def derive_seed(master_seed: int, *parts: str | int) -> int:
    """Stable substream seed, independent of Python's process-randomized hash."""
    # Length prefixes avoid ambiguous concatenations such as ('a1', 2)/('a',12).
    items = [str(master_seed), *(str(part) for part in parts)]
    payload = "".join(f"{len(item)}:{item}" for item in items).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _triangular(rng: random.Random, config: RunConfig, stem: str) -> float:
    return rng.triangular(
        getattr(config, stem + "_low"),
        getattr(config, stem + "_high"),
        getattr(config, stem + "_mode"),
    )


def _mean_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _mc_se(values: list[float]) -> float | None:
    return stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "median": quantile(values, 0.5),
        "p025": quantile(values, 0.025),
        "p975": quantile(values, 0.975),
    }


def _assert_finite(value: Any, path: str = "result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite simulation output at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def run_simulation(
    run_config: RunConfig,
    model_config: ModelConfig,
    scenario: str = "base",
    include_patients: bool = False,
) -> dict:
    """Simulate nested paired trajectories and return a JSON-ready report.

``patients_per_run`` is the number of *true visitors*, including visitors who
are not seeking help. AP is the unconditional mean avoided DLES per true
visitor; AM is AP times monthly IDs times a single multiple-ID correction.
Neither a second helped probability nor a counterfactual-share multiplier is
applied: nonresponse and counterfactual catch-up belong in paired trajectories.

The same master seed and run sizes align patient random streams across model
scenarios and a2 values. Scenario differences do not change a later patient's
random stream. Percentiles across outer runs combine the assumed population
uncertainty, funnel sampling uncertainty and finite inner Monte Carlo noise;
they are not a calibrated confidence or credible interval for actual impact.
"""
    if not isinstance(run_config, RunConfig):
        raise ValueError("run_config must be a RunConfig")
    if not isinstance(model_config, ModelConfig):
        raise ValueError("model_config must be a ModelConfig")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose one of {', '.join(SCENARIOS)}")
    if not isinstance(include_patients, bool):
        raise ValueError("include_patients must be a bool")

    runs = []
    patient_rows = []
    for run_index in range(run_config.outer_runs):
        streams = {
            name: derive_seed(run_config.seed, "outer", run_index, name)
            for name in ("funnel_cohort", "funnel_overlap", "action", "id_correction", "population")
        }
        funnel = draw_funnel(
            random.Random(streams["funnel_cohort"]),
            random.Random(streams["funnel_overlap"]),
            run_config.a2,
        )
        action_probability = _triangular(random.Random(streams["action"]), run_config, "action_probability")
        id_correction = _triangular(random.Random(streams["id_correction"]), run_config, "id_correction")
        population = draw_population(random.Random(streams["population"]), model_config)
        deltas, no_ci_values, ci_values = [], [], []
        conditional = {key: [] for key in ("help_seeking", "action_intent", "action")}
        counts = {key: 0 for key in ("visitors", "help_seeking", "action_intent", "action", "benefited", "harmed")}
        strata: dict[str, dict[str, Any]] = {}
        headstarts = []
        for patient_index in range(run_config.patients_per_run):
            gate_seed = derive_seed(run_config.seed, "outer", run_index, "patient", patient_index, "gates")
            clinical_seed = derive_seed(run_config.seed, "outer", run_index, "patient", patient_index, "clinical")
            gate_rng = random.Random(gate_seed)
            # Always draw all gates so scenario comparison cannot perturb them.
            u_help, u_intent, u_action = (gate_rng.random() for _ in range(3))
            help_seeking = u_help < funnel.help_seeking
            action_intent = help_seeking and u_intent < funnel.action_intent_given_help
            acts = action_intent and u_action < action_probability
            record = None
            if help_seeking:
                result = simulate_patient(
                    random.Random(clinical_seed), population, model_config, acts=acts, scenario=scenario,
                )
                record = asdict(result)
                no_ci, ci, delta = result.no_ci_dles, result.ci_dles, result.delta_dles
                stratum = result.initial_stratum
                if not math.isclose(delta, no_ci - ci, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("patient result delta_dles must equal no_ci_dles - ci_dles")
                if no_ci < 0 or ci < 0:
                    raise ValueError("patient burdens must be nonnegative")
                if acts:
                    headstarts.append(result.headstart_days)
            else:
                no_ci = ci = delta = 0.0
                stratum = "not_help_seeking"
            deltas.append(delta)
            no_ci_values.append(no_ci)
            ci_values.append(ci)
            counts["visitors"] += 1
            for key, flag in (("help_seeking", help_seeking), ("action_intent", action_intent), ("action", acts)):
                counts[key] += int(flag)
                if flag:
                    conditional[key].append(delta)
            counts["benefited"] += int(delta > 0)
            counts["harmed"] += int(delta < 0)
            bucket = strata.setdefault(stratum, {"count": 0, "acting_count": 0, "benefited_count": 0, "deltas": []})
            bucket["count"] += 1
            bucket["acting_count"] += int(acts)
            bucket["benefited_count"] += int(delta > 0)
            bucket["deltas"].append(delta)
            if include_patients:
                patient_rows.append({
                    "run": run_index,
                    "patient": patient_index,
                    "gate_seed": gate_seed,
                    "clinical_seed": clinical_seed,
                    "help_seeking": help_seeking,
                    "action_intent": action_intent,
                    "acts": acts,
                    "no_ci_dles": no_ci,
                    "ci_dles": ci,
                    "delta_dles": delta,
                    "initial_stratum": stratum,
                    "trajectory": record,
                })
        ap = fmean(deltas)
        true_monthly_visitors = run_config.monthly_unique_ids * id_correction
        am = ap * true_monthly_visitors
        n = run_config.patients_per_run
        strata_report = {
            stratum: {
                "count": bucket["count"],
                "acting_count": bucket["acting_count"],
                "benefited_count": bucket["benefited_count"],
                "mean_dles_avoided": fmean(bucket["deltas"]),
                "ap_contribution_dles_per_visitor": math.fsum(bucket["deltas"]) / n,
                "paired_mc_se_dles": _mc_se(bucket["deltas"]),
            }
            for stratum, bucket in sorted(strata.items())
        }
        runs.append({
            "run": run_index,
            "seeds": streams,
            "funnel": asdict(funnel),
            "action_probability_given_intent": action_probability,
            "multiple_id_correction": id_correction,
            "true_monthly_visitors": true_monthly_visitors,
            "population": asdict(population),
            "ap_dles_per_visitor": ap,
            "ap_hours_per_visitor": ap * 24,
            "am_dles_per_monthly_cohort": am,
            "am_hours_per_monthly_cohort": am * 24,
            "no_ci_dles_per_visitor": fmean(no_ci_values),
            "ci_dles_per_visitor": fmean(ci_values),
            "benefit_probability": counts["benefited"] / n,
            "harm_probability": counts["harmed"] / n,
            "benefit_probability_given_action": sum(delta > 0 for delta in conditional["action"]) / counts["action"] if counts["action"] else None,
            "paired_mc_se_dles_per_visitor": _mc_se(deltas),
            "mean_headstart_days_given_action": _mean_or_none(headstarts),
            "counts": counts,
            "conditional_mean_dles_avoided": {key: _mean_or_none(values) for key, values in conditional.items()},
            "strata": strata_report,
        })

    metric_names = (
        "ap_dles_per_visitor", "ap_hours_per_visitor", "am_dles_per_monthly_cohort", "am_hours_per_monthly_cohort",
        "no_ci_dles_per_visitor", "ci_dles_per_visitor", "benefit_probability", "harm_probability",
    )
    report = {
        "metadata": {
            "schema_version": 1,
            "python_version": platform.python_version(),
            "scenario": scenario,
            "seed_derivation": "first 128 bits of SHA-256 over length-prefixed master seed and stream labels",
            "funnel_source": "7–15 October 2026 cohort, with 1–20 October overlapping actionable-learning respondents",
            "interpretation": "Outer 2.5th–97.5th percentiles combine assumed population uncertainty, funnel sampling uncertainty and inner Monte Carlo noise; not a calibrated confidence interval.",
            "assumptions": [
                "One true help-seeking visitor corresponds to one CH patient; other visitors contribute zero burden and benefit.",
                "Actionable learning is used as an action-intent proxy; a2 is fixed within a sensitivity scenario.",
                "Action probability and multiple-ID correction have independent triangular priors; clinical dependence is supplied by the model.",
                "Monthly cohort outcomes accrue over the full model horizon, not within one calendar month.",
                "No additional helped, counterfactual-share, headstart or episodic multiplier is applied.",
            ],
        },
        "run_config": asdict(run_config),
        "model_config": asdict(model_config),
        "summary": {name: _summarize([row[name] for row in runs]) for name in metric_names},
        "runs": runs,
    }
    if include_patients:
        report["patients"] = patient_rows
    _assert_finite(report)
    return report
