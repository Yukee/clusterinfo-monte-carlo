# Provenance

## Procedure

- [Patient helped estimates, Monte Carlo section](https://docs.google.com/document/d/1WB6VoIIBVKr90k6ssXVMHTFe-uj1TRCrWt20gIvWWec/edit?tab=t.0#heading=h.8ecp1bsjz1a0),
  read 2026-10-28. Its paired-trajectory proposal governs this implementation.
- The same document's “20 October 2026 — Bootstrap uncertainty for actionable
  learning” section supplies the aggregate counts, overlap and random seeds.
  Only those aggregate counts are embedded here; no individual analytics
  records or complete private-document copy is included.

## Clinical reference

Alfredo Parra's [cluster-headache-burden](https://github.com/alfredoparra/cluster-headache-burden)
at commit `4cc805cb629a8d472807b03cd2b020656510a960`:

- [Executable defaults and burden equations](https://github.com/alfredoparra/cluster-headache-burden/blob/4cc805cb629a8d472807b03cd2b020656510a960/ch_simulation.py)
- [Source discussion](https://github.com/alfredoparra/cluster-headache-burden/blob/4cc805cb629a8d472807b03cd2b020656510a960/CH_simulation_sources.md)
- [README](https://github.com/alfredoparra/cluster-headache-burden/blob/4cc805cb629a8d472807b03cd2b020656510a960/README.md)

The pinned executable uses patient/attack severity SDs 1.2/0.9 and 10% acute
and preventive peak reductions. Some accompanying prose describes older
values (1.6/1.0 and zero peak reductions); this implementation follows the
executable and the user-provided proposal.

The reference repository has no stated open-source license at that revision.
This project independently implements the mathematical model and records
attribution rather than copying or vendoring its code.

## Evidence versus assumptions

The reference clinical defaults are reference-model assumptions, not a new
validation of medical efficacy. The visitor counts are aggregate observations
transcribed from the proposal. Conditional response independence and transfer
across observation windows remain identification assumptions.

The proposal gives no fitted distributions for initial diagnosis strata,
future care, delays, CI efficacy shifts, headstarts or their dependencies.
The defaults in those areas are explicit illustrative choices. In particular,
preserving 18% acute / 23% preventive marginal access does not calibrate the
joint distribution to ClusterInfo visitors.

No new medical evidence was inferred or generated for these parameters.
