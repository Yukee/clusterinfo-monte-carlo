"""Clinical mathematics and causal coupling, independent of the runner/CLI."""

from dataclasses import asdict, replace
import json
from random import Random
import unittest

from clusterinfo_mc.model import (
    ModelConfig, SCENARIOS, active_intervals, draw_population, extreme_minutes,
    simulate_patient,
)


def fixed_config(**changes):
    defaults = dict(
        uncertainty_logit_sd=0, uncertainty_log_sd=0,
        initial_state_concentration=0, patient_dependence_sd=0,
        delay_log_sd=0, headstart_log_sd=0,
    )
    defaults.update(changes)
    return ModelConfig(**defaults)


def single_state(index):
    return tuple(1.0 if i == index else 0.0 for i in range(8))


def patient(config, *, seed=7, acts=True, scenario="base"):
    return simulate_patient(Random(seed), draw_population(Random(83), config), config,
                            acts=acts, scenario=scenario)


class ConfigTests(unittest.TestCase):
    def test_json_round_trip_and_defaults(self):
        default = ModelConfig()
        self.assertEqual(ModelConfig.from_dict({}), default)
        self.assertEqual(ModelConfig.from_dict(json.loads(json.dumps(asdict(default)))), default)

    def test_invalid_settings_rejected(self):
        for settings in (
            {"bad_name": 1}, {"horizon_days": float("nan")}, {"horizon_days": True},
            {"horizon_days": 0}, {"e_bouts_mean": 0}, {"ci_diagnosis_probability": 1.1},
            {"uncertainty_log_sd": -1}, {"initial_state_weights": [1, 2]},
            {"initial_state_weights": [1] * 8}, {"headstart_days_by_stratum": [10, -1, 20]},
            {"headstart_days_by_stratum": [10, True, 20]}, {"severity_mean": {}},
            {"initial_state_weights": [0, 0, 0, 0, 0, 0, 0, float("inf")]},
            {"min_attack_rate": 8, "max_attack_rate": 1},
            {"preventive_reduction_floor": .9, "preventive_reduction_cap": .5},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ModelConfig.from_dict(settings)
        with self.assertRaises(ValueError):
            ModelConfig.from_dict([])

    def test_fixed_population_preserves_marginals_and_joint_cells(self):
        config = fixed_config()
        population = draw_population(Random(42), config)
        w = population.initial_state_weights
        self.assertAlmostEqual(sum(w[i] for i in (1, 3, 5, 7)), .18)
        self.assertAlmostEqual(sum(w[i] for i in (2, 3, 6, 7)), .23)
        self.assertAlmostEqual(sum(w[4:]), .50)
        self.assertAlmostEqual(population.ci_diagnosis_probability, config.ci_diagnosis_probability)
        self.assertEqual(population.headstart_days_by_stratum, config.headstart_days_by_stratum)

    def test_population_is_reproducible_and_serializable(self):
        config = ModelConfig()
        a = draw_population(Random(1), config)
        self.assertEqual(a, draw_population(Random(1), config))
        self.assertNotEqual(a, draw_population(Random(2), config))
        self.assertAlmostEqual(sum(a.initial_state_weights), 1)
        self.assertTrue(all(0 <= x <= 1 for x in a.initial_state_weights))
        json.dumps(asdict(a), allow_nan=False)

    def test_clinical_population_defaults_are_fixed_and_separate(self):
        config = ModelConfig()
        for seed in range(4):
            p = draw_population(Random(seed), config)
            self.assertAlmostEqual(p.episodic_fraction, .85)
            self.assertAlmostEqual(p.acute_success_mean, .64)
            self.assertAlmostEqual(p.preventive_response_probability, .42)
        uncertain = replace(config, clinical_uncertainty_logit_sd=.5)
        self.assertNotEqual(draw_population(Random(1), uncertain).episodic_fraction,
                            draw_population(Random(2), uncertain).episodic_fraction)
        with self.assertRaises(ValueError):
            ModelConfig(clinical_uncertainty_logit_sd=4)


class ProfileTests(unittest.TestCase):
    def test_reference_bucket_math(self):
        self.assertEqual(extreme_minutes(60, 8), 0)
        self.assertAlmostEqual(extreme_minutes(60, 9), 31 + 29 / 9)
        self.assertAlmostEqual(extreme_minutes(60, 10), 36.8)
        self.assertAlmostEqual(extreme_minutes(15, 10), 3)
        self.assertAlmostEqual(extreme_minutes(15, 9), 15 / 9)
        self.assertEqual(extreme_minutes(0, 10), 0)
        self.assertEqual(extreme_minutes(60, 10, 0, 0), 60)

    def test_score_monotonic_in_peak_and_duration(self):
        for peak in range(1, 11):
            values = [extreme_minutes(d, peak) for d in range(0, 361)]
            self.assertEqual(values, sorted(values))
        for duration in (3, 10, 29, 40, 180):
            values = [extreme_minutes(duration, peak) for peak in range(1, 11)]
            self.assertEqual(values, sorted(values))
            self.assertLessEqual(max(values), duration)

    def test_invalid_profiles(self):
        for duration, peak in ((-1, 9), (float("nan"), 9), (30, 11), (30, 9.0), (30, True)):
            with self.assertRaises(ValueError):
                extreme_minutes(duration, peak)


class BoutTests(unittest.TestCase):
    def test_periodic_intersections(self):
        self.assertEqual(list(active_intervals(40, 20, 5, 2)), [(0, 3), (18, 23), (38, 40)])
        self.assertEqual(list(active_intervals(10, 20, 5, 8)), [])
        self.assertEqual(list(active_intervals(40, 20, 5, 8)), [(12, 17), (32, 37)])
        self.assertEqual(list(active_intervals(40, 20, 0, 8)), [])
        with self.assertRaises(ValueError):
            list(active_intervals(40, 20, 5, 20))

    def test_gains_during_remission_produce_no_acceleration_benefit(self):
        # Ongoing 28-day bout at exposure, next bout after day337. Both arms
        # obtain acute access during remission (day60 versus day150).
        config = fixed_config(
            horizon_days=365, episodic_fraction=1, visit_during_bout_probability=1,
            e_bouts_mean=1, e_bouts_sd=0, e_bout_weeks_mean=4, e_bout_weeks_sd=0,
            e_attack_rate_mean=4, e_attack_rate_sd=0,
            initial_state_weights=single_state(4),
            natural_acute_probability=0, natural_preventive_probability=0,
            ci_acute_access_probability=1, ci_preventive_access_probability=0,
            ci_optimization_probability=0, uptake_delay_days=60, acquisition_delay_days=0,
            headstart_days_by_stratum=(90, 90, 90),
            severity_mean=10, severity_between_sd=0, severity_within_sd=0,
        )
        result = patient(config)
        self.assertGreater(result.attack_count, 0)
        self.assertEqual(result.ci_timeline.acute_access, 60)
        self.assertEqual(result.no_ci_timeline.acute_access, 150)
        self.assertEqual(result.delta_dles, 0)
        self.assertAlmostEqual(result.active_days, 28)


class TimelineTests(unittest.TestCase):
    def timing_config(self, **changes):
        values = dict(
            horizon_days=365, initial_state_weights=single_state(0),
            natural_diagnosis_probability=1, natural_diagnosis_delay_days=100,
            natural_acute_probability=1, natural_preventive_probability=1,
            natural_access_delay_days=10, ci_diagnosis_probability=1,
            ci_acute_access_probability=0, ci_preventive_access_probability=0,
            ci_optimization_probability=0, uptake_delay_days=20,
            headstart_days_by_stratum=(90, 90, 90), min_attack_rate=0, max_attack_rate=0,
        )
        values.update(changes)
        return fixed_config(**values)

    def test_future_diagnosis_and_access_without_ci(self):
        config = self.timing_config(ci_diagnosis_probability=0)
        result = patient(config)
        self.assertEqual(result.no_ci_timeline.diagnosis, 100)
        self.assertEqual(result.no_ci_timeline.acute_access, 110)
        self.assertEqual(result.no_ci_timeline.preventive_access, 110)

    def test_early_diagnosis_unlocks_ordinary_care(self):
        result = patient(self.timing_config())
        self.assertEqual(result.ci_timeline.diagnosis, 20)
        self.assertEqual(result.no_ci_timeline.diagnosis, 100)
        self.assertEqual(result.ci_timeline.acute_access, 30)
        self.assertEqual(result.no_ci_timeline.acute_access, 110)
        self.assertEqual(result.headstart_days, 80)
        self.assertEqual(result.proposed_headstart_days, 90)

    def test_natural_care_can_precede_ci(self):
        result = patient(self.timing_config(natural_diagnosis_delay_days=5))
        self.assertEqual(result.ci_timeline.diagnosis, 5)
        self.assertEqual(result.ci_timeline.acute_access, 15)
        self.assertEqual(result.ci_timeline, result.no_ci_timeline)
        self.assertEqual(result.headstart_days, 0)

    def test_never_diagnosed_or_treated_is_explicit_none(self):
        result = patient(self.timing_config(natural_diagnosis_probability=0, ci_diagnosis_probability=0))
        self.assertIsNone(result.no_ci_timeline.diagnosis)
        self.assertIsNone(result.no_ci_timeline.acute_access)
        self.assertIsNone(result.no_ci_timeline.preventive_access)
        json.dumps(asdict(result), allow_nan=False)

    def test_existing_access_without_diagnosis_is_retained(self):
        result = patient(self.timing_config(
            initial_state_weights=single_state(3),
            natural_diagnosis_probability=0, ci_diagnosis_probability=0,
        ))
        self.assertIsNone(result.ci_timeline.diagnosis)
        self.assertEqual(result.ci_timeline.acute_access, 0)
        self.assertEqual(result.ci_timeline.preventive_access, 0)
        self.assertEqual(result.initial_stratum, "undiagnosed")

    def test_headstart_sensitivities(self):
        config = self.timing_config(headstart_days_by_stratum=(180, 90, 45))
        self.assertEqual(patient(config).proposed_headstart_days, 180)
        self.assertEqual(patient(config, scenario="reverse_headstarts").proposed_headstart_days, 45)
        self.assertEqual(patient(config, scenario="equal_headstarts").proposed_headstart_days, 105)

    def test_undiagnosed_optimization_uses_optimization_headstart(self):
        config = self.timing_config(
            initial_state_weights=single_state(3), natural_diagnosis_probability=0,
            ci_diagnosis_probability=0, ci_optimization_probability=1,
            headstart_days_by_stratum=(180, 90, 45),
        )
        result = patient(config)
        self.assertEqual(result.initial_stratum, "undiagnosed")
        self.assertEqual(result.headstart_stratum, "optimization")
        self.assertEqual(result.proposed_headstart_days, 45)

    def test_new_access_uses_access_headstart(self):
        config = self.timing_config(
            initial_state_weights=single_state(4), natural_acute_probability=0,
            natural_preventive_probability=0, ci_acute_access_probability=1,
            headstart_days_by_stratum=(180, 90, 45),
        )
        result = patient(config)
        self.assertEqual(result.headstart_stratum, "access")
        self.assertEqual(result.proposed_headstart_days, 90)


class CouplingTests(unittest.TestCase):
    def test_nonactors_and_null_controls_are_exactly_zero(self):
        config = ModelConfig(horizon_days=120)
        population = draw_population(Random(811), config)
        for seed in range(20):
            base = simulate_patient(Random(seed), population, config, acts=True)
            nonactor = simulate_patient(Random(seed), population, config, acts=False)
            null = simulate_patient(Random(seed), population, config, acts=True, scenario="no_ci")
            zero = simulate_patient(Random(seed), population, config, acts=True, scenario="zero_headstart")
            for result in (nonactor, null, zero):
                self.assertEqual(result.delta_dles, 0)
                self.assertEqual(result.no_ci_timeline, result.ci_timeline)
            self.assertEqual(nonactor.no_ci_dles, base.no_ci_dles)
            self.assertEqual(null.no_ci_dles, base.no_ci_dles)
            self.assertEqual(nonactor.no_ci_timeline, base.no_ci_timeline)
            self.assertEqual(null.no_ci_timeline, base.no_ci_timeline)
            self.assertEqual(base.attack_count, zero.attack_count)

    def test_shared_coupling_cannot_create_harm(self):
        config = ModelConfig(horizon_days=365)
        for outer in range(3):
            population = draw_population(Random(outer), config)
            for seed in range(20):
                for scenario in ("base", "no_ci_peak", "no_peak"):
                    result = simulate_patient(Random(seed), population, config, acts=True, scenario=scenario)
                    self.assertGreaterEqual(result.delta_dles, 0)
                    self.assertGreaterEqual(result.ci_attacks_prevented, result.no_ci_attacks_prevented)
                    self.assertGreaterEqual(result.ci_dles, 0)
                    json.dumps(asdict(result), allow_nan=False)

    def test_peak_sensitivity_removes_baseline_peak_effects(self):
        config = fixed_config(
            horizon_days=10, episodic_fraction=0, initial_state_weights=single_state(6),
            severity_mean=10, severity_between_sd=0, severity_within_sd=0,
            preventive_response_probability=1, preventive_reduction_floor=0,
            preventive_reduction_cap=0, preventive_peak_reduction=.2,
            ci_optimization_probability=0, ci_acute_access_probability=0,
            natural_acute_probability=0,
        )
        base = patient(config)
        no_peak = patient(config, scenario="no_peak")
        self.assertEqual(base.no_ci_dles, 0)
        self.assertGreater(no_peak.no_ci_dles, 0)
        self.assertEqual(no_peak.delta_dles, 0)

    def test_no_ci_peak_disables_only_extra_peak_gain(self):
        config = fixed_config(
            horizon_days=5, episodic_fraction=0, initial_state_weights=single_state(5),
            severity_mean=10, severity_between_sd=0, severity_within_sd=0,
            acute_success_mean=1, acute_success_sd=0, acute_treated_fraction=1,
            acute_peak_reduction=0, acute_peak_gain=1, acute_duration_gain=0,
            acute_duration_mean_minutes=360, acute_duration_sd_minutes=0,
            ci_optimization_probability=1, uptake_delay_days=0,
            headstart_days_by_stratum=(1000, 1000, 1000),
            ci_preventive_access_probability=0, natural_preventive_probability=0,
        )
        base = patient(config)
        no_extra = patient(config, scenario="no_ci_peak")
        self.assertEqual(base.no_ci_dles, no_extra.no_ci_dles)
        self.assertGreater(base.delta_dles, 0)
        self.assertEqual(no_extra.delta_dles, 0)

    def test_invalid_scenario_and_action(self):
        with self.assertRaises(ValueError):
            patient(ModelConfig(), scenario="typo")
        with self.assertRaises(ValueError):
            patient(ModelConfig(), acts=1)

    def test_saturated_optimization_can_remove_all_extreme_burden(self):
        class MaximumGainRandom(Random):
            def uniform(self, a, b):
                return b

        config = fixed_config(
            horizon_days=10, episodic_fraction=0, initial_state_weights=single_state(7),
            severity_mean=10, severity_between_sd=0, severity_within_sd=0,
            acute_success_mean=0, acute_success_sd=0, preventive_response_probability=0,
            acute_treated_gain=1, acute_success_gain=1, acute_duration_gain=1, acute_peak_gain=1,
            preventive_response_gain=1, preventive_frequency_gain=1, preventive_peak_gain=1,
            ci_optimization_probability=1, uptake_delay_days=0,
            headstart_days_by_stratum=(1000, 1000, 1000),
        )
        result = simulate_patient(MaximumGainRandom(7), draw_population(Random(83), config), config, acts=True)
        self.assertGreater(result.no_ci_dles, 0)
        self.assertEqual(result.ci_dles, 0)
        self.assertEqual(result.delta_dles, result.no_ci_dles)


if __name__ == "__main__":
    unittest.main()
