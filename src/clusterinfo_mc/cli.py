"""Command-line runs, auditable outputs and paired sensitivity comparisons."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from . import __version__
from .funnel import bootstrap
from .model import ModelConfig, SCENARIOS
from .simulation import RunConfig, run_simulation


SOURCE_DOCUMENT = "https://docs.google.com/document/d/1WB6VoIIBVKr90k6ssXVMHTFe-uj1TRCrWt20gIvWWec/edit?tab=t.0#heading=h.8ecp1bsjz1a0"
REFERENCE_COMMIT = "4cc805cb629a8d472807b03cd2b020656510a960"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_config(path: Path | None) -> tuple[RunConfig, ModelConfig]:
    data = {} if path is None else json.loads(path.read_text(), object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or set(data) - {"run", "model"}:
        raise ValueError("config must be an object containing only 'run' and 'model'")
    return RunConfig.from_dict(data.get("run", {})), ModelConfig.from_dict(data.get("model", {}))


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _output_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output path is not an empty directory: {path}; choose a new path")
    path.mkdir(parents=True, exist_ok=True)


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _code_identity() -> dict:
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    identity = {"source_sha256": digest.hexdigest(), "git_head": None, "git_dirty": None}
    if not (root / ".git").exists():
        return identity
    try:
        identity["git_head"] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        identity["git_dirty"] = bool(subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return identity


def _add_provenance(result: dict, scenario: str) -> None:
    inputs = {"run": result["run_config"], "model": result["model_config"], "scenario": scenario}
    result["provenance"] = {
        "package_version": __version__,
        **_code_identity(),
        "source_document": SOURCE_DOCUMENT,
        "reference_model_commit": REFERENCE_COMMIT,
        "specification_read_date": "2026-10-28",
        "config_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        "calibration_status": "Illustrative assumptions; not a calibrated impact estimate.",
    }


def _report(result: dict, scenario: str) -> str:
    lines = ["# ClusterInfo Monte Carlo run", "", f"Scenario: `{scenario}`", "",
             "**Illustrative assumptions; not a calibrated impact estimate.**", "",
             "The percentile ranges combine outer parameter uncertainty with finite inner simulation noise. "
             "They are not conventional confidence intervals or monthly realized relief.", "",
             "| Metric | Mean | Median | 2.5th percentile | 97.5th percentile |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for key, value in result["summary"].items():
        if isinstance(value, dict) and all(k in value for k in ("mean", "median", "p025", "p975")):
            lines.append(f"| {key} | {value['mean']:.6g} | {value['median']:.6g} | {value['p025']:.6g} | {value['p975']:.6g} |")
    lines += ["", "AP is accumulated DLES avoided per corrected visitor over the configured horizon. "
              "AM scales AP to one monthly arrival cohort using the ID correction once. "
              "One DLES is 24 hours in the reference model's extreme-pain proxy.", "",
              "See `result.json` for all inputs, population draws, strata, sample sizes, seeds, "
              "within-run Monte Carlo standard errors and provenance. See `docs/methodology.md` "
              "in the repository for assumptions and limitations.", ""]
    return "\n".join(lines)


def save_result(path: Path, result: dict, scenario: str) -> None:
    _add_provenance(result, scenario)
    _json(path / "result.json", result)
    _json(path / "config.json", {"run": result["run_config"], "model": result["model_config"]})
    _csv(path / "runs.csv", result["runs"])
    if "patients" in result:
        _csv(path / "patients.csv", result["patients"])
    (path / "report.md").write_text(_report(result, scenario))


def _simulation_arguments(parser) -> None:
    parser.add_argument("--config", type=Path, help="JSON with 'run' and 'model' settings; omitted fields use defaults")
    parser.add_argument("--outer-runs", type=int)
    parser.add_argument("--patients", type=int, help="simulated visitors per outer run")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--a2", type=float, help="fixed sensitivity assumption, between 0 and 1")
    parser.add_argument("--horizon-days", type=float)
    parser.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    parser.add_argument("--patient-output", action="store_true", help="include synthetic patient audit rows (larger files)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired Monte Carlo model of accelerated cluster-headache relief.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("init-config", help="write all default assumptions as editable JSON")
    config.add_argument("--out", type=Path, required=True)
    boot = commands.add_parser("bootstrap", help="reproduce the joint visitor bootstrap from the proposal")
    boot.add_argument("--draws", type=int, default=30000)
    boot.add_argument("--seed", type=int, default=20261020)
    boot.add_argument("--overlap-seed", type=int, default=20261021)
    boot.add_argument("--a2", type=float, default=0.5)
    boot.add_argument("--out", type=Path, required=True)
    simulate = commands.add_parser("simulate", help="simulate one fixed scenario")
    _simulation_arguments(simulate)
    simulate.add_argument("--scenario", choices=SCENARIOS, default="base")
    sensitivity = commands.add_parser("sensitivity", help="compare a2 bounds, peak effects, headstarts and null controls")
    _simulation_arguments(sensitivity)
    return parser


def _configs(args):
    run, model = load_config(args.config)
    updates = {key: getattr(args, key) for key in ("outer_runs", "seed", "a2") if getattr(args, key) is not None}
    if args.patients is not None:
        updates["patients_per_run"] = args.patients
    run = RunConfig.from_dict(asdict(run) | updates)
    if args.horizon_days is not None:
        model = ModelConfig.from_dict(asdict(model) | {"horizon_days": args.horizon_days})
    return run, model


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init-config":
            if args.out.exists():
                raise ValueError(f"config already exists: {args.out}")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            _json(args.out, {"run": asdict(RunConfig()), "model": asdict(ModelConfig())})
            print(f"Wrote {args.out}")
        elif args.command == "bootstrap":
            _output_directory(args.out)
            result = bootstrap(args.draws, args.seed, args.overlap_seed, args.a2)
            _json(args.out / "bootstrap.json", result)
            for key, endpoints in result["intervals"].items():
                if key != "weights":
                    print(f"{key}: {result['point_estimate'][key]:.1%} [{endpoints[0]:.1%}, {endpoints[1]:.1%}]")
            print(f"Saved {args.out / 'bootstrap.json'}")
        else:
            run, model = _configs(args)
            _output_directory(args.out)
            if args.command == "simulate":
                result = run_simulation(run, model, scenario=args.scenario, include_patients=args.patient_output)
                save_result(args.out, result, args.scenario)
                print(f"Saved {args.out / 'report.md'}")
            else:
                comparisons = []
                variants = [("a2_low", "base", 0.0), ("base", "base", run.a2), ("a2_high", "base", 1.0)]
                variants += [(scenario, scenario, run.a2) for scenario in SCENARIOS if scenario != "base"]
                for label, scenario, a2 in variants:
                    print(f"Running {label} (a2={a2:g})...", flush=True)
                    destination = args.out / label
                    destination.mkdir()
                    result = run_simulation(replace(run, a2=a2), model, scenario=scenario, include_patients=args.patient_output)
                    save_result(destination, result, scenario)
                    comparisons.append({"label": label, "scenario": scenario, "a2": a2,
                                        "summary": result["summary"]})
                _json(args.out / "comparison.json", comparisons)
                rows = []
                for item in comparisons:
                    row = {key: item[key] for key in ("label", "scenario", "a2")}
                    for metric, stats in item["summary"].items():
                        if isinstance(stats, dict) and "mean" in stats:
                            row[metric + "_mean"] = stats["mean"]
                    rows.append(row)
                _csv(args.out / "comparison.csv", rows)
                print(f"Saved {args.out / 'comparison.csv'}")
        return 0
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
