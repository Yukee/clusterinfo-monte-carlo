# ClusterInfo Monte Carlo

A reproducible Python implementation of the paired CI/no-CI Monte Carlo procedure
in [Patient helped estimates](https://docs.google.com/document/d/1WB6VoIIBVKr90k6ssXVMHTFe-uj1TRCrWt20gIvWWec/edit?tab=t.0#heading=h.8ecp1bsjz1a0),
read on 28 October 2026.

**This is an exploratory model with illustrative assumptions, not a calibrated
estimate of ClusterInfo's impact.** The visitor bootstrap is based on the
document's aggregate observations. Most diagnosis, access, uptake, headstart and
efficacy-improvement parameters still need evidence or elicitation.

## Run it

Python **3.12 or newer**; no runtime dependencies or credentials. Python 3.15
reproduces the document's exact bootstrap random-number implementation.

```sh
python3 run.py bootstrap --a2 0 --out results/bootstrap-low
python3 run.py bootstrap --a2 1 --out results/bootstrap-high
python3 run.py simulate --outer-runs 10 --patients 200 --out results/demo
```

The bootstrap uses 30,000 resamples by default. The published 95% intervals are
35.8–46.7% for help-seeking and 10.5–18.0% / 30.0–42.1% for actionable learning
at `a2=0` / `a2=1`. Actionable learning serves as a proxy for action intent;
it is not observed treatment uptake or benefit.

For a larger exploratory run, increase both levels of sampling:

```sh
python3 run.py simulate --config examples/default.json \
  --outer-runs 200 --patients 2000 --out results/larger-run
```

This evaluates many individual attacks and can take substantial time. Start
with a small run to inspect assumptions. Use the reported inner Monte Carlo
standard errors and repeated seeds to assess simulation precision before
interpreting outer percentile ranges.

Optional package installation provides the `clusterinfo-mc` command:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/clusterinfo-mc --help
```

## Change the assumptions

```sh
python3 run.py init-config --out my-assumptions.json
python3 run.py simulate --config my-assumptions.json --a2 0.5 \
  --horizon-days 1095 --out results/three-years
```

JSON has two sections, `run` and `model`. Missing fields use defaults; unknown
keys, duplicate JSON keys, invalid probabilities and nonfinite values fail.
`examples/default.json` lists every setting. `a2` is a fixed sensitivity
assumption, not a sampled probability distribution. Defaults use its midpoint
0.5 without claiming that this is empirically preferred.

```sh
python3 run.py sensitivity --outer-runs 10 --patients 200 \
  --out results/sensitivity
```

This compares `a2` endpoints, the chosen base value, no additional CI peak
reduction, zero treatment peak reductions in both arms, reversed/equal
headstart ordering, and no-CI/zero-headstart null controls. Scenarios reuse
the same random-number streams. A single variant can be run with
`simulate --scenario no_peak` (see `simulate --help`).

## What is implemented

- Uncertain population parameters sampled once per outer run, including a
  joint visitor bootstrap that preserves the eight overlapping respondents.
- Nested help-seeking → action intent → action gates, with a separate uncertain
  action-given-intent probability.
- Joint initial diagnosis and acute/preventive access; subsequent diagnosis
  and access in both potential histories.
- Shared patient traits and attacks in paired CI/no-CI trajectories, uptake
  delays, episodic bout timing, and headstarts conditional on the achieved
  diagnosis, access or optimization pathway.
- Acute and preventive access and efficacy changes, including no improvement
  and nonresponse; both arms can eventually obtain the same improvement.
- Extreme-pain DLES accumulated over a common horizon, per corrected visitor
  and per monthly arrival cohort; no extra counterfactual-share, helped,
  headstart or episodic multipliers.

See [the mathematical model and limitations](docs/methodology.md) and
[source provenance](docs/provenance.md). The initial-access defaults reproduce
the reference model's **marginal** 18%/23% rates. Their allocation across
diagnosis strata is illustrative and must be recalibrated for CI visitors.

## Outputs

Every simulation saves:

- `result.json`: full configuration, seed/version/provenance, each outer draw,
  summary percentiles, strata and within-run paired Monte Carlo error.
- `runs.csv`: one row per outer draw; nested diagnostics stored as JSON cells.
- `report.md`: a readable summary.
- `config.json`: resolved assumptions for rerunning the analysis.
- `patients.csv` with `--patient-output`: **synthetic** patient audit rows.

`sensitivity` also saves `comparison.json` and `comparison.csv`.
Output directories must be new or empty to protect prior runs. `results/` is
ignored by Git. No live analytics access or personal records are required.

AP is DLES avoided per corrected visitor **over the chosen horizon**. AM is
the accumulated relief from one month's arrival cohort; it is not relief
realized within that calendar month. The outer percentile ranges combine
assumed parameter distributions, bootstrap uncertainty and finite inner
simulation error. They are not conventional confidence intervals.

## Development

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests cover published bootstrap intervals, shared-person dependence,
paired-trajectory null controls, DLES arithmetic, treatment/bout timing,
configuration validation and output scaling. GitHub Actions runs the same
tests on Python 3.12 and 3.15 when the repository is pushed.

The implementation uses the reference model's documented equations and
defaults independently; no upstream source code is vendored. A distribution
license has not yet been selected for this repository.
