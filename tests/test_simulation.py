from dataclasses import dataclass, replace
import json
import math
import unittest
from unittest.mock import patch

from clusterinfo_mc.funnel import FunnelDraw
from clusterinfo_mc.model import ModelConfig
from clusterinfo_mc.simulation import RunConfig, derive_seed, run_simulation


@dataclass(frozen=True)
class FakePopulation:
    rate: float = 0.5


@dataclass(frozen=True)
class FakePatient:
    no_ci_dles: float
    ci_dles: float
    delta_dles: float
    initial_stratum: str = "diagnosed_no_access"
    episodic: bool = True
    headstart_days: float = 90


def fake_patient(rng, population, config, *, acts, scenario="base"):
    baseline = 1 + rng.random()
    delta = 0.5 if acts and scenario not in ("no_ci", "zero_headstart") else 0
    return FakePatient(baseline, baseline - delta, delta)


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.run_config = RunConfig(outer_runs=3, patients_per_run=30)
        self.model_config = ModelConfig()
        self.population_patch = patch("clusterinfo_mc.simulation.draw_population", return_value=FakePopulation())
        self.patient_patch = patch("clusterinfo_mc.simulation.simulate_patient", side_effect=fake_patient)
        self.population_mock = self.population_patch.start()
        self.patient_mock = self.patient_patch.start()
        self.addCleanup(self.population_patch.stop)
        self.addCleanup(self.patient_patch.stop)

    def test_repeated_runs_identical_and_json_finite(self):
        first = run_simulation(self.run_config, self.model_config, include_patients=True)
        second = run_simulation(self.run_config, self.model_config, include_patients=True)
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)
        self.assertEqual(len(first["patients"]), 90)
        self.assertNotIn("patients", run_simulation(self.run_config, self.model_config))

    def test_population_draw_once_per_outer_run(self):
        run_simulation(self.run_config, self.model_config)
        self.assertEqual(self.population_mock.call_count, self.run_config.outer_runs)

    def test_funnel_is_conditional_and_nonseekers_do_not_enter_patient_model(self):
        report = run_simulation(self.run_config, self.model_config, include_patients=True)
        people = report["patients"]
        self.assertEqual(self.patient_mock.call_count, sum(p["help_seeking"] for p in people))
        for p in people:
            self.assertFalse(p["acts"] and not p["action_intent"])
            self.assertFalse(p["action_intent"] and not p["help_seeking"])
            if not p["help_seeking"]:
                self.assertEqual(p["no_ci_dles"], 0)
                self.assertEqual(p["ci_dles"], 0)
                self.assertIsNone(p["trajectory"])

    def test_scaling_and_no_extra_funnel_or_counterfactual_multiplier(self):
        funnel = FunnelDraw(1, 1, 1, (0, 0, 0, 1))
        config = replace(self.run_config, monthly_unique_ids=100, id_correction_low=0.8, id_correction_mode=0.8,
                         id_correction_high=0.8, action_probability_low=1, action_probability_mode=1, action_probability_high=1)
        with patch("clusterinfo_mc.simulation.draw_funnel", return_value=funnel):
            report = run_simulation(config, self.model_config)
        for row in report["runs"]:
            self.assertEqual(row["ap_dles_per_visitor"], 0.5)
            self.assertEqual(row["am_dles_per_monthly_cohort"], 40)
            self.assertEqual(row["ap_hours_per_visitor"], 12)
            self.assertEqual(row["am_hours_per_monthly_cohort"], 960)
            self.assertEqual(row["benefit_probability"], 1)
            self.assertEqual(row["paired_mc_se_dles_per_visitor"], 0)
            self.assertEqual(row["counts"]["action"], config.patients_per_run)

    def test_all_nonseekers_zero_and_empty_conditionals_are_null(self):
        with patch("clusterinfo_mc.simulation.draw_funnel", return_value=FunnelDraw(0, 1, 1, (0, 0, 0, 1))):
            report = run_simulation(self.run_config, self.model_config)
        self.patient_mock.assert_not_called()
        for row in report["runs"]:
            self.assertEqual(row["ap_dles_per_visitor"], 0)
            self.assertEqual(row["no_ci_dles_per_visitor"], 0)
            self.assertEqual(row["counts"]["help_seeking"], 0)
            self.assertIsNone(row["benefit_probability_given_action"])
            self.assertIsNone(row["conditional_mean_dles_avoided"]["action"])

    def test_zero_action_has_zero_benefit_without_erasing_baseline(self):
        config = replace(self.run_config, action_probability_low=0, action_probability_mode=0, action_probability_high=0)
        report = run_simulation(config, self.model_config)
        for row in report["runs"]:
            self.assertEqual(row["ap_dles_per_visitor"], 0)
            self.assertEqual(row["no_ci_dles_per_visitor"], row["ci_dles_per_visitor"])
            self.assertGreater(row["no_ci_dles_per_visitor"], 0)

    def test_common_random_numbers_across_scenarios_and_a2(self):
        base = run_simulation(self.run_config, self.model_config, include_patients=True)
        null = run_simulation(self.run_config, self.model_config, scenario="no_ci", include_patients=True)
        low = run_simulation(replace(self.run_config, a2=0), self.model_config, include_patients=True)
        high = run_simulation(replace(self.run_config, a2=1), self.model_config, include_patients=True)
        for b, n, lo, hi in zip(base["patients"], null["patients"], low["patients"], high["patients"]):
            self.assertEqual(b["clinical_seed"], n["clinical_seed"])
            self.assertEqual(b["acts"], n["acts"])
            self.assertEqual(b["no_ci_dles"], n["no_ci_dles"])
            self.assertEqual(n["delta_dles"], 0)
            self.assertEqual(lo["help_seeking"], hi["help_seeking"])
            self.assertEqual(lo["no_ci_dles"], hi["no_ci_dles"])
            self.assertFalse(lo["acts"] and not hi["acts"])

    def test_strata_contributions_and_paired_standard_error(self):
        report = run_simulation(self.run_config, self.model_config, include_patients=True)
        for row in report["runs"]:
            self.assertEqual(sum(s["count"] for s in row["strata"].values()), self.run_config.patients_per_run)
            self.assertAlmostEqual(sum(s["ap_contribution_dles_per_visitor"] for s in row["strata"].values()), row["ap_dles_per_visitor"])
            values = [p["delta_dles"] for p in report["patients"] if p["run"] == row["run"]]
            mean = sum(values) / len(values)
            expected = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1) / len(values))
            self.assertAlmostEqual(row["paired_mc_se_dles_per_visitor"], expected)

    def test_single_patient_mc_se_is_null(self):
        report = run_simulation(replace(self.run_config, patients_per_run=1), self.model_config)
        self.assertTrue(all(row["paired_mc_se_dles_per_visitor"] is None for row in report["runs"]))

    def test_validation_and_stable_seeds(self):
        self.assertEqual(RunConfig.from_dict({}), RunConfig())
        for values in ({"outer_runs": 0}, {"patients_per_run": True}, {"seed": -1}, {"a2": math.nan},
                       {"a2": 2}, {"typo": 1}, {"id_correction_low": 0.9}, {"monthly_unique_ids": 1.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RunConfig.from_dict(values)
        with self.assertRaises(ValueError):
            run_simulation(self.run_config, self.model_config, scenario="typo")
        self.assertEqual(derive_seed(1, "x", 2), derive_seed(1, "x", 2))
        self.assertNotEqual(derive_seed(1, "x1", 2), derive_seed(1, "x", 12))

    def test_reject_nonfinite_or_inconsistent_patient_results(self):
        with patch("clusterinfo_mc.simulation.draw_funnel", return_value=FunnelDraw(1, 1, 1, (0, 0, 0, 1))):
            self.patient_mock.side_effect = lambda *args, **kwargs: FakePatient(1, 0, math.nan)
            with self.assertRaises(ValueError):
                run_simulation(self.run_config, self.model_config)
            self.patient_mock.side_effect = lambda *args, **kwargs: FakePatient(1, 0, 0.5)
            with self.assertRaisesRegex(ValueError, "must equal"):
                run_simulation(self.run_config, self.model_config)


if __name__ == "__main__":
    unittest.main()
