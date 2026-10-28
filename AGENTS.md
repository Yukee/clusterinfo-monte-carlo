# Working on this model

This is an independent Python repository, separate from the ClusterInfo website.
Read README.md and docs/methodology.md before changing the model.

- Use Python 3.12+ and the standard library unless a dependency has a clear need.
- Keep source defaults separate from illustrative, uncalibrated assumptions.
- Preserve paired potential outcomes and random-number streams across scenarios.
- Sample uncertain population parameters once per outer run; never independently
  resample the help-seeking and actionable-learning bootstrap quantities.
- Preserve the documented shared-person bootstrap overlap.
- Do not reintroduce separate counterfactual-share, helped-probability, headstart,
  or episodic multipliers on benefits already computed from the trajectories.
- Avoid committing patient-level data, credentials, or generated result folders.
- Add tests for changed mathematical behavior and run:
  `PYTHONPATH=src python -m unittest discover -s tests -v`.
- Do not push or publish the repository without the user's instruction.
