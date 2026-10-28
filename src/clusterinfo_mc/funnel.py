"""Joint visitor bootstrap for the 7–15 October 2026 cohort.

These are aggregate observations from the proposal, not individual records. The
bootstrap carries shared-person dependence between help-seeking and actionable
learning. The latter is a proxy for action intent, not observed treatment benefit.
Python 3.12+ is required for ``Random.binomialvariate``. The source's numerical
reproduction uses Python 3.15; random algorithms can differ between versions.
"""

from dataclasses import asdict, dataclass
import math
import random
import sys
from typing import Sequence


# Intent Yes, No, unanswered, ordered within engagement scores E = 0, 1, 2, 3.
COHORT_COUNTS = ((29, 47, 607), (59, 96, 256), (71, 81, 151), (58, 58, 78))
COHORT_SIZE = 1591
OVERLAP_YES = 8
OUTSIDE_TOTAL = 12
OUTSIDE_YES = 11


class InvalidFunnelDraw(ValueError):
    """The resample cannot identify one of the required conditional estimates."""


@dataclass(frozen=True)
class FunnelDraw:
    help_seeking: float
    action_intent_given_help: float
    a3: float
    weights: tuple[float, float, float, float]


def _validate_a2(a2: float) -> None:
    if isinstance(a2, bool) or not isinstance(a2, (int, float)) or not math.isfinite(a2) or not 0 <= a2 <= 1:
        raise ValueError("a2 must be a finite probability between 0 and 1")


def estimate_funnel(
    counts: Sequence[Sequence[int]], a3: float, a2: float
) -> FunnelDraw:
    """Apply the engagement-adjusted estimator to one set of cohort counts.

Zero response counts in a nonempty engagement stratum make its help-seeking
share undefined. We fail explicitly; substituting zero or redrawing silently
would change the estimator. An empty stratum contributes zero mass.
"""
    _validate_a2(a2)
    if not isinstance(a3, (int, float)) or not math.isfinite(a3) or not 0 <= a3 <= 1:
        raise ValueError("a3 must be a finite probability between 0 and 1")
    if len(counts) != 4 or any(len(row) != 3 for row in counts):
        raise ValueError("counts must contain four (Yes, No, unanswered) rows")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for row in counts for n in row):
        raise ValueError("cohort counts must be nonnegative integers")
    total = sum(sum(row) for row in counts)
    if total == 0:
        raise InvalidFunnelDraw("empty cohort: help-seeking denominator is zero")
    help_mass = []
    for score, (yes, no, unanswered) in enumerate(counts):
        n = yes + no + unanswered
        if n == 0:
            help_mass.append(0.0)
        elif yes + no == 0:
            raise InvalidFunnelDraw(f"engagement score {score} has no intent respondents")
        else:
            help_mass.append(n * yes / (yes + no))
    total_help = sum(help_mass)
    if total_help == 0:
        raise InvalidFunnelDraw("estimated help-seeking mass is zero")
    weights = tuple(h / total_help for h in help_mass)
    return FunnelDraw(total_help / total, a2 * weights[2] + a3 * weights[3], a3, weights)


def point_estimate(a2: float = 0.5) -> FunnelDraw:
    """Point estimate with a0=a1=0, observed a3=19/20, and fixed a2."""
    return estimate_funnel(COHORT_COUNTS, 19 / 20, a2)


def _sample_cohort(rng: random.Random) -> tuple[tuple[int, int, int], ...]:
    """Multinomial visitor resampling via sequential conditional binomials."""
    observed = [n for row in COHORT_COUNTS for n in row]
    remaining_draws = COHORT_SIZE
    remaining_weight = COHORT_SIZE
    sampled = []
    for weight in observed[:-1]:
        count = rng.binomialvariate(remaining_draws, weight / remaining_weight)
        sampled.append(count)
        remaining_draws -= count
        remaining_weight -= weight
    sampled.append(remaining_draws)
    return tuple(tuple(sampled[i : i + 3]) for i in range(0, 12, 3))


def draw_funnel(rng: random.Random, overlap_rng: random.Random, a2: float = 0.5) -> FunnelDraw:
    """Draw one joint outer-run funnel estimate using two independent streams.

The overlapping actionable-learning respondents are a subset of the sampled
E=3/intent-Yes group, so their multiplicity depends on that SAME cohort draw.
The outside group is separately resampled at its fixed size of 12.
"""
    _validate_a2(a2)
    counts = _sample_cohort(rng)
    overlap_count = overlap_rng.binomialvariate(counts[3][0], OVERLAP_YES / COHORT_COUNTS[3][0])
    outside_yes = overlap_rng.binomialvariate(OUTSIDE_TOTAL, OUTSIDE_YES / OUTSIDE_TOTAL)
    a3 = (overlap_count + outside_yes) / (overlap_count + OUTSIDE_TOTAL)
    return estimate_funnel(counts, a3, a2)


def quantile(values: Sequence[float], p: float) -> float:
    """Linear interpolation at (B - 1) * p, as specified in the proposal."""
    if not values:
        raise ValueError("quantiles require at least one value")
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("quantile probability must be between 0 and 1")
    if any(not math.isfinite(x) for x in values):
        raise ValueError("quantiles require finite values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def bootstrap(
    draws: int = 30_000,
    seed: int = 20261020,
    overlap_seed: int = 20261021,
    a2: float = 0.5,
) -> dict:
    """Summarize percentile intervals for one fixed a2 sensitivity scenario.

Raises InvalidFunnelDraw with the failing draw number if any required
denominator is zero. No invalid samples are silently skipped or repaired.
Use identical seeds across a2 values to compare paired sensitivity scenarios.
These intervals reflect sampling uncertainty only, conditional on the stated
nonresponse and transferability assumptions. They are not benefit intervals.
"""
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    _validate_a2(a2)
    rng = random.Random(seed)
    overlap_rng = random.Random(overlap_seed)
    samples = []
    for index in range(draws):
        try:
            samples.append(draw_funnel(rng, overlap_rng, a2))
        except InvalidFunnelDraw as exc:
            raise InvalidFunnelDraw(f"bootstrap draw {index + 1}: {exc}") from exc

    def interval(values: Sequence[float]) -> list[float]:
        return [quantile(values, 0.025), quantile(values, 0.975)]

    return {
        "draws": draws,
        "seed": seed,
        "overlap_seed": overlap_seed,
        "a2": a2,
        "python_version": sys.version.split()[0],
        "cohort_size": COHORT_SIZE,
        "invalid_draws": 0,
        "point_estimate": asdict(point_estimate(a2)),
        "intervals": {
            name: interval([getattr(sample, name) for sample in samples])
            for name in ("help_seeking", "action_intent_given_help", "a3")
        }
        | {"weights": [interval([sample.weights[e] for sample in samples]) for e in range(4)]},
        "interval_level": 0.95,
        "method": "joint stratified visitor percentile bootstrap; linear quantiles",
        "interpretation": "Sampling uncertainty conditional on a2 and response/transfer assumptions; actionable learning is not treatment benefit.",
    }
