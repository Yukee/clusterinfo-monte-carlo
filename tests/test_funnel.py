import math
import random
import sys
import unittest
from unittest.mock import patch

from clusterinfo_mc.funnel import (
    COHORT_COUNTS,
    COHORT_SIZE,
    InvalidFunnelDraw,
    _sample_cohort,
    bootstrap,
    draw_funnel,
    estimate_funnel,
    point_estimate,
    quantile,
)


class FunnelTests(unittest.TestCase):
    def test_observed_estimates_match_document(self):
        low = point_estimate(0)
        high = point_estimate(1)
        self.assertEqual(sum(map(sum, COHORT_COUNTS)), 1591)
        self.assertAlmostEqual(low.help_seeking, 0.4121, delta=0.00005)
        self.assertAlmostEqual(low.weights[2], 0.21588, delta=0.000005)
        self.assertAlmostEqual(low.weights[3], 0.1480, delta=0.00005)
        self.assertAlmostEqual(low.action_intent_given_help, 0.14056, delta=0.000005)
        self.assertAlmostEqual(high.action_intent_given_help, 0.35644, delta=0.00001)
        self.assertEqual(low.a3, 0.95)

    def test_multinomial_draws_conserve_cohort_and_are_deterministic(self):
        rng1, rng2 = random.Random(123), random.Random(123)
        for _ in range(100):
            sample = _sample_cohort(rng1)
            self.assertEqual(sample, _sample_cohort(rng2))
            self.assertEqual(sum(map(sum, sample)), COHORT_SIZE)
            self.assertTrue(all(n >= 0 for row in sample for n in row))

    def test_overlap_uses_sampled_e3_yes_multiplicity(self):
        counts = ((29, 47, 607), (59, 96, 256), (71, 81, 151), (87, 58, 49))
        calls = []

        class ScriptedRandom:
            def binomialvariate(self, n, p):
                calls.append((n, p))
                return (9, 10)[len(calls) - 1]

        with patch("clusterinfo_mc.funnel._sample_cohort", return_value=counts):
            result = draw_funnel(random.Random(1), ScriptedRandom(), 0.2)
        self.assertEqual(calls, [(87, 8 / 58), (12, 11 / 12)])
        self.assertEqual(result.a3, 19 / 21)
        self.assertEqual(result, estimate_funnel(counts, 19 / 21, 0.2))

    def test_a2_scenarios_use_same_joint_draw(self):
        for seed in range(20):
            low = draw_funnel(random.Random(seed), random.Random(seed + 30), 0)
            high = draw_funnel(random.Random(seed), random.Random(seed + 30), 1)
            self.assertEqual(low.help_seeking, high.help_seeking)
            self.assertEqual(low.weights, high.weights)
            self.assertEqual(low.a3, high.a3)
            self.assertAlmostEqual(high.action_intent_given_help - low.action_intent_given_help, low.weights[2])

    def test_nonempty_unanswered_stratum_is_undefined(self):
        with self.assertRaisesRegex(InvalidFunnelDraw, "score 0"):
            estimate_funnel(((0, 0, 1), (1, 0, 0), (1, 0, 0), (1, 0, 0)), 0.95, 0)
        with self.assertRaisesRegex(InvalidFunnelDraw, "empty cohort"):
            estimate_funnel(((0, 0, 0),) * 4, 0.95, 0)
        with self.assertRaisesRegex(InvalidFunnelDraw, "mass is zero"):
            estimate_funnel(((0, 1, 0),) * 4, 0.95, 0)
        result = estimate_funnel(((0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0)), 1, 1)
        self.assertEqual(result.help_seeking, 1)
        self.assertEqual(result.weights, (0, 0.5, 0, 0.5))

    def test_invalid_draw_not_silently_discarded(self):
        with patch("clusterinfo_mc.funnel.draw_funnel", side_effect=InvalidFunnelDraw("test")):
            with self.assertRaisesRegex(InvalidFunnelDraw, "draw 1: test"):
                bootstrap(draws=2)

    def test_probability_and_count_validation(self):
        for a2 in (-0.1, 1.1, math.nan, math.inf, True):
            with self.subTest(a2=a2), self.assertRaises(ValueError):
                point_estimate(a2)
        for draws in (0, -1, 1.5, True):
            with self.subTest(draws=draws), self.assertRaises(ValueError):
                bootstrap(draws=draws)
        with self.assertRaises(ValueError):
            estimate_funnel(((1, 1, -1),) * 4, 0.95, 0)

    def test_linear_quantiles(self):
        self.assertEqual(quantile([30, 10, 20], 0.25), 15)
        self.assertEqual(quantile([9], 0.975), 9)
        self.assertEqual(quantile([1, 3], 0), 1)
        self.assertEqual(quantile([1, 3], 1), 3)
        with self.assertRaises(ValueError):
            quantile([], 0.5)

    def test_bootstrap_document_intervals(self):
        # The published source used 3.15; other supported runtimes may use a
        # different binomial RNG algorithm, but should agree within Monte Carlo
        # sampling error. Check the published one-decimal percentages on 3.15.
        low = bootstrap(a2=0)
        high = bootstrap(a2=1)
        self.assertEqual(low["intervals"]["help_seeking"], high["intervals"]["help_seeking"])
        expected = (
            (low["intervals"]["help_seeking"], (0.358, 0.467)),
            (low["intervals"]["action_intent_given_help"], (0.105, 0.180)),
            (high["intervals"]["action_intent_given_help"], (0.300, 0.421)),
        )
        tolerance = 0.0005 if sys.version_info[:2] == (3, 15) else 0.003
        for actual_pair, expected_pair in expected:
            for actual, target in zip(actual_pair, expected_pair):
                self.assertAlmostEqual(actual, target, delta=tolerance)
        self.assertEqual(low["invalid_draws"], 0)


if __name__ == "__main__":
    unittest.main()
