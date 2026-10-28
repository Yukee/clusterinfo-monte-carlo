"""Paired patient trajectories for the ClusterInfo acceleration model.

Clinical distributions follow the *equations* in Alfredo Parra's pinned model,
independently implemented here. Timing, visitor calibration and intervention
priors are illustrative assumptions, not fitted treatment-effect estimates.
All time units are days except explicitly named attack durations (minutes).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from random import Random
from typing import Any, Iterator, Mapping


SCENARIOS = (
    "base", "no_ci", "zero_headstart", "no_ci_peak", "no_peak",
    "reverse_headstarts", "equal_headstarts",
)
STRATA = (
    "undiagnosed", "diagnosed_without_full_access", "diagnosed_with_access",
)


@dataclass(frozen=True)
class ModelConfig:
    """Explicit assumptions. Array fields serialize as JSON arrays.

    ``initial_state_weights`` order: undiagnosed none/acute/preventive/both,
    then diagnosed none/acute/preventive/both. The illustrative means preserve
    .18 acute and .23 preventive marginal access; these are NOT rates conditional
    on diagnosis, nor evidence of calibration to actual ClusterInfo visitors.
    Headstart entries are lognormal medians for diagnosis/access/optimization
    pathways, classified by whether a proposed event advances natural care.
    """

    horizon_days: float = 730.0
    episodic_fraction: float = 0.85
    visit_during_bout_probability: float = 0.8
    e_bouts_mean: float = 1.2
    e_bouts_sd: float = 1.1
    e_bout_weeks_mean: float = 8.5
    e_bout_weeks_sd: float = 5.7
    e_attack_rate_mean: float = 2.0
    e_attack_rate_sd: float = 1.5
    c_attack_rate_mean: float = 2.5
    c_attack_rate_sd: float = 1.8
    min_attack_rate: float = 0.5
    max_attack_rate: float = 8.0
    chronic_active_fraction: float = 0.9
    severity_mean: float = 7.0
    severity_between_sd: float = 1.2
    severity_within_sd: float = 0.9
    duration_median_minutes: float = 45.0
    duration_log_sd: float = 0.70
    duration_floor_minutes: float = 5.0
    duration_cap_minutes: float = 360.0
    duration_intensity_slope: float = 0.08
    acute_success_mean: float = 0.64
    acute_success_sd: float = 0.22
    acute_treated_fraction: float = 0.85
    acute_duration_mean_minutes: float = 15.0
    acute_duration_sd_minutes: float = 6.0
    acute_duration_floor_minutes: float = 3.0
    acute_peak_reduction: float = 0.10
    preventive_response_probability: float = 0.42
    preventive_reduction_mean: float = 0.55
    preventive_reduction_sd: float = 0.20
    preventive_reduction_floor: float = 0.50
    preventive_reduction_cap: float = 0.95
    preventive_peak_reduction: float = 0.10
    profile_rise_minutes: float = 9.0
    profile_decline_minutes: float = 20.0

    initial_state_weights: tuple[float, ...] = (
        0.43, 0.03, 0.02, 0.02, 0.22, 0.09, 0.15, 0.04,
    )
    initial_state_concentration: float = 100.0
    uncertainty_logit_sd: float = 0.45
    clinical_uncertainty_logit_sd: float = 0.0
    uncertainty_log_sd: float = 0.40
    population_dependence: float = 0.60
    patient_dependence_sd: float = 0.50
    natural_diagnosis_probability: float = 0.60
    natural_acute_probability: float = 0.65
    natural_preventive_probability: float = 0.60
    natural_diagnosis_delay_days: float = 365.0
    natural_access_delay_days: float = 90.0
    ci_diagnosis_probability: float = 0.75
    ci_acute_access_probability: float = 0.55
    ci_preventive_access_probability: float = 0.50
    ci_optimization_probability: float = 0.65
    uptake_delay_days: float = 14.0
    acquisition_delay_days: float = 30.0
    delay_log_sd: float = 0.75
    headstart_days_by_stratum: tuple[float, ...] = (180.0, 90.0, 45.0)
    headstart_log_sd: float = 0.70
    acute_treated_gain: float = 0.08
    acute_success_gain: float = 0.15
    acute_duration_gain: float = 0.30
    acute_peak_gain: float = 0.08
    preventive_response_gain: float = 0.20
    preventive_frequency_gain: float = 0.15
    preventive_peak_gain: float = 0.08

    def __post_init__(self) -> None:
        arrays = {"initial_state_weights": 8, "headstart_days_by_stratum": 3}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in arrays:
                if not isinstance(value, (list, tuple)) or len(value) != arrays[f.name]:
                    raise ValueError(f"{f.name} must contain {arrays[f.name]} numbers")
                for item in value:
                    _number(item, f.name)
                    if item < 0:
                        raise ValueError(f"{f.name} must be nonnegative")
                object.__setattr__(self, f.name, tuple(float(x) for x in value))
            else:
                _number(value, f.name)
                if value < 0:
                    raise ValueError(f"{f.name} must be nonnegative")
        probabilities = (
            "episodic_fraction", "visit_during_bout_probability", "chronic_active_fraction",
            "acute_success_mean", "acute_treated_fraction", "acute_peak_reduction",
            "preventive_response_probability", "preventive_reduction_mean",
            "preventive_reduction_floor", "preventive_reduction_cap",
            "preventive_peak_reduction", "population_dependence",
        )
        for f in fields(self):
            if f.name in probabilities or f.name.endswith(("_probability", "_gain")):
                if getattr(self, f.name) > 1:
                    raise ValueError(f"{f.name} must be in [0, 1]")
        for name in ("e_bouts_mean", "e_bout_weeks_mean", "e_attack_rate_mean",
                     "c_attack_rate_mean", "duration_median_minutes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.horizon_days <= 36500:
            raise ValueError("horizon_days must be in (0, 36500]")
        if not 1 <= self.severity_mean <= 10:
            raise ValueError("severity_mean must be in [1, 10]")
        if not 0 <= self.min_attack_rate <= self.max_attack_rate <= 24:
            raise ValueError("attack rate bounds must satisfy 0 <= min <= max <= 24")
        if not self.duration_floor_minutes <= self.duration_cap_minutes <= 1440:
            raise ValueError("duration bounds must satisfy 0 <= floor <= cap <= 1440")
        if not self.preventive_reduction_floor <= self.preventive_reduction_cap:
            raise ValueError("preventive_reduction_floor must not exceed cap")
        if not math.isclose(sum(self.initial_state_weights), 1.0, abs_tol=1e-10):
            raise ValueError("initial_state_weights must sum to 1")
        if self.initial_state_concentration != 0 and not 0.01 <= self.initial_state_concentration <= 1e9:
            raise ValueError("initial_state_concentration must be 0 or in [0.01, 1e9]")
        for name in ("uncertainty_logit_sd", "clinical_uncertainty_logit_sd", "uncertainty_log_sd", "patient_dependence_sd",
                     "delay_log_sd", "headstart_log_sd", "duration_log_sd"):
            if getattr(self, name) > 3:
                raise ValueError(f"{name} must be at most 3")
        # Bound otherwise unbounded scales to catch unit mistakes and overflow.
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name != "initial_state_concentration" and isinstance(value, (int, float)) and value > 1e6:
                raise ValueError(f"{f.name} must be at most 1000000")
        if max(self.headstart_days_by_stratum) > 1e6:
            raise ValueError("headstart medians must be at most 1000000 days")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ModelConfig:
        if not isinstance(values, Mapping):
            raise ValueError("model config must be an object")
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown model settings: {', '.join(sorted(map(str, unknown)))}")
        return cls(**values)


def _number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True)
class Population:
    initial_state_weights: tuple[float, ...]
    episodic_fraction: float
    acute_success_mean: float
    preventive_response_probability: float
    natural_diagnosis_probability: float
    natural_acute_probability: float
    natural_preventive_probability: float
    natural_diagnosis_delay_days: float
    natural_access_delay_days: float
    ci_diagnosis_probability: float
    ci_acute_access_probability: float
    ci_preventive_access_probability: float
    ci_optimization_probability: float
    uptake_delay_days: float
    acquisition_delay_days: float
    headstart_days_by_stratum: tuple[float, ...]
    acute_treated_gain: float
    acute_success_gain: float
    acute_duration_gain: float
    acute_peak_gain: float
    preventive_response_gain: float
    preventive_frequency_gain: float
    preventive_peak_gain: float
    shared_barrier_z: float


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _shift_probability(p: float, shift: float) -> float:
    if p == 0 or p == 1 or shift == 0:
        return p
    logit = math.log(p) - math.log1p(-p) + shift
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1 + exp_logit)


def draw_population(rng: Random, config: ModelConfig) -> Population:
    """Draw one joint set of population assumptions for an outer run.

    The shared normal factor represents barriers: higher barriers lower chances
    of diagnosis/access/optimization and increase delays/headstarts. Conditional
    normals give a configurable copula dependence, not independent marginal
    interval endpoints. The eight-cell Dirichlet retains dependence among its
    cell weights. Zero prior SDs and zero concentration fix the respective inputs.
    """
    shared = rng.gauss(0, 1)
    shared_scale = math.sqrt(config.population_dependence)
    independent_scale = math.sqrt(1 - config.population_dependence)

    def z() -> float:
        return shared_scale * shared + independent_scale * rng.gauss(0, 1)

    def probability(value: float) -> float:
        return _shift_probability(value, -config.uncertainty_logit_sd * z())

    def days(value: float) -> float:
        return value * math.exp(config.uncertainty_log_sd * z())

    if config.initial_state_concentration == 0:
        weights = config.initial_state_weights
    else:
        raw = [rng.gammavariate(w * config.initial_state_concentration, 1) if w else 0.0
               for w in config.initial_state_weights]
        total = sum(raw)
        if not total > 0:
            raise ValueError("Dirichlet draw underflowed; increase initial_state_concentration")
        weights = tuple(x / total for x in raw)
    values: dict[str, Any] = {
        "initial_state_weights": weights,
        "shared_barrier_z": shared,
        "headstart_days_by_stratum": tuple(days(x) for x in config.headstart_days_by_stratum),
        # No evidence supports coupling subtype or intrinsic clinical response
        # to the access-barrier factor; their population uncertainty is separate.
        "episodic_fraction": _shift_probability(config.episodic_fraction, config.clinical_uncertainty_logit_sd * rng.gauss(0, 1)),
        "acute_success_mean": _shift_probability(config.acute_success_mean, config.clinical_uncertainty_logit_sd * rng.gauss(0, 1)),
        "preventive_response_probability": _shift_probability(config.preventive_response_probability, config.clinical_uncertainty_logit_sd * rng.gauss(0, 1)),
    }
    for f in fields(Population):
        if f.name in values:
            continue
        value = getattr(config, f.name)
        values[f.name] = days(value) if f.name.endswith("_days") else probability(value)
    return Population(**values)


@dataclass(frozen=True)
class Timeline:
    """An event time of None means never in the modeled potential history."""

    diagnosis: float | None
    acute_access: float | None
    preventive_access: float | None
    acute_optimization: float | None = None
    preventive_optimization: float | None = None


@dataclass(frozen=True)
class PatientResult:
    no_ci_dles: float
    ci_dles: float
    delta_dles: float
    initial_stratum: str
    episodic: bool
    headstart_days: float
    proposed_headstart_days: float
    headstart_stratum: str
    initial_diagnosed: bool
    initial_acute: bool
    initial_preventive: bool
    no_ci_timeline: Timeline
    ci_timeline: Timeline
    natural_timeline: Timeline
    attack_count: int
    no_ci_attacks_prevented: int
    ci_attacks_prevented: int
    active_days: float
    visit_in_bout: bool
    period_days: float | None
    bout_days: float | None
    phase_days: float | None


def _gamma(rng: Random, mean: float, sd: float) -> float:
    if sd == 0:
        return mean
    return rng.gammavariate((mean / sd) ** 2, sd * sd / mean)


def _lognormal(rng: Random, median: float, log_sd: float) -> float:
    # Still draw when median/sd are zero so scenario switches keep RNG alignment.
    return median * math.exp(log_sd * rng.gauss(0, 1))


def active_intervals(horizon_days: float, period_days: float, bout_days: float,
                     phase_days: float) -> Iterator[tuple[float, float]]:
    """Intersect periodic half-open bouts with [0, horizon).

    ``phase_days`` is the elapsed time within the cycle at the visit, not the
    absolute date of a bout start. A phase inside the bout allows ongoing bouts
    at exposure and correctly retains only their remaining active time.
    """
    for value, name in ((horizon_days, "horizon_days"), (period_days, "period_days"),
                        (bout_days, "bout_days"), (phase_days, "phase_days")):
        _number(value, name)
    if horizon_days < 0 or period_days <= 0 or not 0 <= bout_days <= period_days:
        raise ValueError("invalid horizon/period/bout duration")
    if not 0 <= phase_days < period_days:
        raise ValueError("phase_days must lie in [0, period_days)")
    if bout_days == 0:
        return
    start = -phase_days
    while start < horizon_days:
        end = min(horizon_days, start + bout_days)
        if end > max(0.0, start):
            yield max(0.0, start), end
        start += period_days


def extreme_minutes(duration_minutes: float, peak: int,
                    rise_minutes: float = 9.0, decline_minutes: float = 20.0) -> float:
    """Discrete NRS-bucket >=9 proxy used in the reference model.

    The combined linear ramps allocate equal time to integer levels 1..peak.
    For peak 9, one ninth of the ramps is extreme; for 10, two tenths. This is
    not a continuous threshold integral starting at the numerical value 9.
    """
    for value, name in ((duration_minutes, "duration_minutes"), (rise_minutes, "rise_minutes"),
                        (decline_minutes, "decline_minutes")):
        _number(value, name)
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if isinstance(peak, bool) or not isinstance(peak, int) or not 1 <= peak <= 10:
        raise ValueError("peak must be an integer in [1, 10]")
    return _extreme_minutes(duration_minutes, peak, rise_minutes + decline_minutes)


def _extreme_minutes(duration: float, peak: int, nominal_ramp: float) -> float:
    if peak < 9:
        return 0.0
    ramp = min(duration, nominal_ramp)
    return duration - ramp + ramp * (peak - 8) / peak


def _earlier(*times: float | None) -> float | None:
    present = [x for x in times if x is not None]
    return min(present) if present else None


def _later(first: float, second: float | None) -> float | None:
    return max(first, second) if second is not None else None


def _occurs(time: float | None, at: float) -> bool:
    return time is not None and time <= at


def _cell(rng: Random, weights: tuple[float, ...]) -> int:
    u = rng.random()
    cumulative = 0.0
    for i, weight in enumerate(weights):
        cumulative += weight
        if u < cumulative:
            return i
    return len(weights) - 1  # floating-point normalization only


def simulate_patient(rng: Random, population: Population, config: ModelConfig,
                     *, acts: bool, scenario: str = "base") -> PatientResult:
    """Simulate one potential attack process under paired care histories.

    No actor/nonresponder discount is applied here. A shared attack and shared
    prevention/treatment/success uniforms make earlier, nonnegative improvements
    weakly beneficial attack by attack. Optimization catches up in the no-CI arm
    after the same sampled headstart. Natural diagnosis/access can happen first.
    State is evaluated at attack onset. Attacks crossing the horizon have their
    observed duration clipped *before* bucket-profile scoring (a documented
    boundary approximation that preserves monotone coupling).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    if not isinstance(acts, bool):
        raise ValueError("acts must be a boolean")
    c, p = config, population
    state = _cell(rng, p.initial_state_weights)
    diagnosed = state >= 4
    acute = bool(state % 4 in (1, 3))
    preventive = bool(state % 4 in (2, 3))
    stratum_index = 0 if not diagnosed else (2 if acute and preventive else 1)
    episodic = rng.random() < p.episodic_fraction
    severity = _clip(rng.gauss(c.severity_mean, c.severity_between_sd), 1, 10)
    success_base = _clip(rng.gauss(p.acute_success_mean, c.acute_success_sd), 0, 0.95)
    responder_u = rng.random()
    frequency_base = _clip(rng.gauss(c.preventive_reduction_mean, c.preventive_reduction_sd),
                           c.preventive_reduction_floor, c.preventive_reduction_cap)
    # A barrier shared by this person's events and gains creates dependence
    # beyond the outer population dependence. Large barriers increase delays
    # and potential headstart while reducing the chance of CI providing access.
    barrier = math.exp(c.patient_dependence_sd * rng.gauss(0, 1))
    probability = lambda x: _shift_probability(x, -math.log(barrier))
    delay = lambda x: barrier * _lognormal(rng, x, c.delay_log_sd)
    natural_diagnosis_draw = rng.random() < probability(p.natural_diagnosis_probability)
    natural_acute_draw = rng.random() < probability(p.natural_acute_probability)
    natural_preventive_draw = rng.random() < probability(p.natural_preventive_probability)
    natural_dx_delay = delay(p.natural_diagnosis_delay_days)
    natural_ac_delay = delay(p.natural_access_delay_days)
    natural_pr_delay = delay(p.natural_access_delay_days)
    ci_diagnosis_draw = rng.random() < probability(p.ci_diagnosis_probability)
    ci_acute_draw = rng.random() < probability(p.ci_acute_access_probability)
    ci_preventive_draw = rng.random() < probability(p.ci_preventive_access_probability)
    optimize_ac = rng.random() < probability(p.ci_optimization_probability)
    optimize_pr = rng.random() < probability(p.ci_optimization_probability)
    uptake = delay(p.uptake_delay_days)
    acquire_ac = delay(p.acquisition_delay_days)
    acquire_pr = delay(p.acquisition_delay_days)
    # Build candidate care first so headstart depends on the actual pathway,
    # not merely initial diagnosis status. No random draws occur in this block.
    natural_dx = 0.0 if diagnosed else (natural_dx_delay if natural_diagnosis_draw else None)
    natural_ac = 0.0 if acute else (
        natural_dx + natural_ac_delay if natural_dx is not None and natural_acute_draw else None)
    natural_pr = 0.0 if preventive else (
        natural_dx + natural_pr_delay if natural_dx is not None and natural_preventive_draw else None)
    natural = Timeline(natural_dx, natural_ac, natural_pr)
    proposed_dx = uptake if not diagnosed and ci_diagnosis_draw else None
    ci_dx = _earlier(natural_dx, proposed_dx)
    proposed_ac = _later(uptake, ci_dx) if not acute and ci_acute_draw else None
    proposed_pr = _later(uptake, ci_dx) if not preventive and ci_preventive_draw else None
    proposed_ac = proposed_ac + acquire_ac if proposed_ac is not None else None
    proposed_pr = proposed_pr + acquire_pr if proposed_pr is not None else None
    natural_ac_ci = 0.0 if acute else (ci_dx + natural_ac_delay if ci_dx is not None and natural_acute_draw else None)
    natural_pr_ci = 0.0 if preventive else (ci_dx + natural_pr_delay if ci_dx is not None and natural_preventive_draw else None)
    ci_ac, ci_pr = _earlier(natural_ac_ci, proposed_ac), _earlier(natural_pr_ci, proposed_pr)

    def advances(candidate: float | None, natural_time: float | None) -> bool:
        return candidate is not None and (natural_time is None or candidate < natural_time)

    headstart_index = (0 if advances(ci_dx, natural_dx) else
                       1 if advances(ci_ac, natural_ac) or advances(ci_pr, natural_pr) else 2)
    headstart_stratum = ("diagnosis", "access", "optimization")[headstart_index]
    headstart_medians = p.headstart_days_by_stratum
    if scenario == "reverse_headstarts":
        headstart_medians = tuple(reversed(headstart_medians))
    elif scenario == "equal_headstarts":
        headstart_medians = (sum(headstart_medians) / 3,) * 3
    proposed_headstart = barrier * _lognormal(rng, headstart_medians[headstart_index], c.headstart_log_sd)
    if scenario == "zero_headstart":
        proposed_headstart = 0.0
    # Draw gain magnitudes regardless of eligibility, to keep scenario RNGs
    # aligned. There is a point mass at no improvement through optimization draws.
    gain_names = ("acute_treated_gain", "acute_success_gain", "acute_duration_gain", "acute_peak_gain",
                  "preventive_response_gain", "preventive_frequency_gain", "preventive_peak_gain")
    gains = {name: _clip(getattr(p, name) * rng.uniform(0, 2) / barrier, 0, 1) for name in gain_names}
    if scenario in ("no_ci_peak", "no_peak"):
        gains["acute_peak_gain"] = gains["preventive_peak_gain"] = 0.0
    acute_peak = c.acute_peak_reduction if scenario != "no_peak" else 0.0
    preventive_peak = c.preventive_peak_reduction if scenario != "no_peak" else 0.0

    # Every patient has the same latent no-CI future whether or not they act on
    # CI. Action gates exposure to the accelerated arm, not the no-CI history.
    no_dx = _earlier(natural_dx, proposed_dx + proposed_headstart if proposed_dx is not None else None)
    # Earlier diagnosis can unlock ordinary care as well as a direct acquisition
    # opportunity. Reuse each access chance and lag in both diagnosis histories.
    natural_ac_no = 0.0 if acute else (no_dx + natural_ac_delay if no_dx is not None and natural_acute_draw else None)
    natural_pr_no = 0.0 if preventive else (no_dx + natural_pr_delay if no_dx is not None and natural_preventive_draw else None)
    no_ac = _earlier(natural_ac_no, proposed_ac + proposed_headstart if proposed_ac is not None else None)
    no_pr = _earlier(natural_pr_no, proposed_pr + proposed_headstart if proposed_pr is not None else None)
    gain_ac = _later(uptake, ci_ac) if optimize_ac else None
    gain_pr = _later(uptake, ci_pr) if optimize_pr else None
    ci = Timeline(ci_dx, ci_ac, ci_pr, gain_ac, gain_pr)
    no_ci = Timeline(no_dx, no_ac, no_pr,
                     max(no_ac, gain_ac + proposed_headstart) if gain_ac is not None and no_ac is not None else None,
                     max(no_pr, gain_pr + proposed_headstart) if gain_pr is not None and no_pr is not None else None)
    if not acts or scenario == "no_ci":
        ci = no_ci
    # Realized acceleration in a treatment transition inside the horizon. This
    # can be less than the proposed headstart when natural care arrives first.
    gaps = [b - a for name in ("acute_access", "preventive_access", "acute_optimization", "preventive_optimization")
            if (a := getattr(ci, name)) is not None and (b := getattr(no_ci, name)) is not None
            and a < c.horizon_days]
    headstart = max([0.0, *gaps])

    bouts = max(_gamma(rng, c.e_bouts_mean, c.e_bouts_sd), 1e-9)
    weeks = _gamma(rng, c.e_bout_weeks_mean, c.e_bout_weeks_sd)
    e_rate = _clip(_gamma(rng, c.e_attack_rate_mean, c.e_attack_rate_sd), c.min_attack_rate, c.max_attack_rate)
    c_rate = _clip(_gamma(rng, c.c_attack_rate_mean, c.c_attack_rate_sd), c.min_attack_rate, c.max_attack_rate)
    visit_in_bout = rng.random() < c.visit_during_bout_probability
    phase_u = rng.random()
    if episodic:
        period = 365.0 / bouts
        bout = min(7 * weeks, 273.0 / bouts)  # <=39 active weeks per 365-day cycle total
        phase = phase_u * bout if visit_in_bout else bout + phase_u * (period - bout)
        intervals = active_intervals(c.horizon_days, period, bout, phase)
        rate = e_rate
    else:
        period = bout = phase = None
        intervals = iter([(0.0, c.horizon_days)])
        rate = c_rate * c.chronic_active_fraction
        visit_in_bout = True
    active_days = 0.0
    totals = [0.0, 0.0]
    prevented = [0, 0]
    attack_count = 0
    nominal_ramp = c.profile_rise_minutes + c.profile_decline_minutes
    response_improved = p.preventive_response_probability + gains["preventive_response_gain"] * (1 - p.preventive_response_probability)
    frequency_improved = frequency_base + gains["preventive_frequency_gain"] * (c.preventive_reduction_cap - frequency_base)
    success_improved = success_base + gains["acute_success_gain"] * (0.95 - success_base)
    treated_improved = c.acute_treated_fraction + gains["acute_treated_gain"] * (1 - c.acute_treated_fraction)

    for start, end in intervals:
        active_days += end - start
        if rate == 0:
            continue
        at = start + rng.expovariate(rate)
        while at < end:
            attack_count += 1
            peak = int(_clip(round(rng.gauss(severity, c.severity_within_sd)), 1, 10))
            duration_factor = _clip(1 + c.duration_intensity_slope * (peak - c.severity_mean), 0.3, 2.5)
            duration = _clip(_lognormal(rng, c.duration_median_minutes, c.duration_log_sd) * duration_factor,
                             c.duration_floor_minutes, c.duration_cap_minutes)
            prevent_u, treat_u, success_u = rng.random(), rng.random(), rng.random()
            relief = max(c.acute_duration_floor_minutes,
                         rng.gauss(c.acute_duration_mean_minutes, c.acute_duration_sd_minutes))
            remaining_minutes = (c.horizon_days - at) * 1440
            for arm, history in enumerate((no_ci, ci)):
                pr_gain = _occurs(history.preventive_optimization, at)
                responder = _occurs(history.preventive_access, at) and responder_u < (
                    response_improved if pr_gain else p.preventive_response_probability)
                reduction = (frequency_improved if pr_gain else frequency_base) if responder else 0.0
                if prevent_u < reduction:
                    prevented[arm] += 1
                    continue
                ac_gain = _occurs(history.acute_optimization, at)
                aborted = (_occurs(history.acute_access, at)
                           and treat_u < (treated_improved if ac_gain else c.acute_treated_fraction)
                           and success_u < (success_improved if ac_gain else success_base))
                observed_duration = duration
                final_peak = peak
                if responder:
                    cut = _clip(preventive_peak + (gains["preventive_peak_gain"] if pr_gain else 0), 0, 1)
                    final_peak = max(1, round(final_peak * (1 - cut)))
                if aborted:
                    short = max(c.acute_duration_floor_minutes,
                                relief * (1 - gains["acute_duration_gain"]) if ac_gain else relief)
                    observed_duration = min(observed_duration, short)
                    cut = _clip(acute_peak + (gains["acute_peak_gain"] if ac_gain else 0), 0, 1)
                    final_peak = max(1, round(final_peak * (1 - cut)))
                totals[arm] += _extreme_minutes(min(observed_duration, remaining_minutes), final_peak, nominal_ramp)
            at += rng.expovariate(rate)
    no_burden, ci_burden = (x / 1440 for x in totals)
    return PatientResult(
        no_ci_dles=no_burden, ci_dles=ci_burden, delta_dles=no_burden - ci_burden,
        initial_stratum=STRATA[stratum_index], episodic=episodic,
        headstart_days=headstart, proposed_headstart_days=proposed_headstart,
        headstart_stratum=headstart_stratum,
        initial_diagnosed=diagnosed, initial_acute=acute, initial_preventive=preventive,
        no_ci_timeline=no_ci, ci_timeline=ci, natural_timeline=natural,
        attack_count=attack_count, no_ci_attacks_prevented=prevented[0], ci_attacks_prevented=prevented[1],
        active_days=active_days if episodic else active_days * c.chronic_active_fraction,
        visit_in_bout=visit_in_bout, period_days=period, bout_days=bout, phase_days=phase,
    )
