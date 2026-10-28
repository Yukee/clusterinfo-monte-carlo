"""Exercise checkout execution and the persisted audit artifacts end to end."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run.py"


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(path.read_text(), parse_constant=reject_constant)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def run_cli(self, *arguments, success=True):
        # Running outside the checkout verifies run.py's import setup as well
        # as the CLI. No editable install or PYTHONPATH is required.
        completed = subprocess.run(
            [sys.executable, str(RUNNER), *map(str, arguments)],
            cwd=self.directory,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if success:
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        else:
            self.assertEqual(completed.returncode, 2, completed.stderr + completed.stdout)
            self.assertIn("error:", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
        return completed

    def test_bootstrap_output_and_overwrite_protection(self):
        output = self.directory / "bootstrap"
        self.run_cli("bootstrap", "--draws", 300, "--a2", 0, "--out", output)
        artifact = output / "bootstrap.json"
        result = read_json(artifact)
        self.assertEqual(result["draws"], 300)
        self.assertEqual(result["seed"], 20261020)
        self.assertEqual(result["overlap_seed"], 20261021)
        self.assertEqual(result["invalid_draws"], 0)
        self.assertAlmostEqual(result["point_estimate"]["action_intent_given_help"], 0.14055902254151972)
        self.assertEqual(len(result["intervals"]["weights"]), 4)
        before = artifact.read_bytes()
        refused = self.run_cli("bootstrap", "--draws", 300, "--out", output, success=False)
        self.assertIn("not an empty directory", refused.stderr)
        self.assertEqual(artifact.read_bytes(), before)

    def test_init_config_roundtrip_and_cli_overrides(self):
        config_path = self.directory / "config.json"
        self.run_cli("init-config", "--out", config_path)
        generated = read_json(config_path)
        self.assertEqual(set(generated), {"run", "model"})
        self.assertEqual(generated["run"]["seed"], 20261028)
        before = config_path.read_bytes()
        self.run_cli("init-config", "--out", config_path, success=False)
        self.assertEqual(config_path.read_bytes(), before)
        output = self.directory / "roundtrip"
        self.run_cli("simulate", "--config", config_path, "--outer-runs", 1, "--patients", 2,
                     "--seed", 7654, "--horizon-days", 0.25, "--out", output)
        saved = read_json(output / "config.json")
        expected_run = generated["run"] | {"outer_runs": 1, "patients_per_run": 2, "seed": 7654}
        expected_model = generated["model"] | {"horizon_days": 0.25}
        self.assertEqual(saved, {"run": expected_run, "model": expected_model})

    def test_simulate_writes_consistent_json_csv_and_patient_audit(self):
        output = self.directory / "simulation"
        self.run_cli("simulate", "--outer-runs", 2, "--patients", 12, "--horizon-days", 14,
                     "--patient-output", "--out", output)
        for name in ("report.md", "runs.csv", "result.json", "config.json", "patients.csv"):
            self.assertTrue((output / name).is_file(), name)
        result = read_json(output / "result.json")
        saved_config = read_json(output / "config.json")
        self.assertEqual(saved_config, {"run": result["run_config"], "model": result["model_config"]})
        self.assertEqual(result["metadata"]["scenario"], "base")
        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual(len(result["patients"]), 24)
        self.assertEqual(len(result["provenance"]["config_sha256"]), 64)
        self.assertIn("not a calibrated impact estimate", result["provenance"]["calibration_status"])
        with (output / "runs.csv").open(newline="") as stream:
            csv_runs = list(csv.DictReader(stream))
        self.assertEqual(len(csv_runs), 2)
        for csv_row, json_row in zip(csv_runs, result["runs"]):
            self.assertEqual(float(csv_row["ap_dles_per_visitor"]), json_row["ap_dles_per_visitor"])
            self.assertEqual(json.loads(csv_row["counts"]), json_row["counts"])
        with (output / "patients.csv").open(newline="") as stream:
            csv_patients = list(csv.DictReader(stream))
        self.assertEqual(len(csv_patients), 24)
        for row in result["patients"]:
            self.assertAlmostEqual(row["delta_dles"], row["no_ci_dles"] - row["ci_dles"])
        report = (output / "report.md").read_text()
        self.assertIn("Illustrative assumptions", report)
        self.assertIn("am_dles_per_monthly_cohort", report)

    def test_sensitivity_variants_share_seeds_and_null_controls_are_zero(self):
        output = self.directory / "sensitivity"
        self.run_cli("sensitivity", "--outer-runs", 1, "--patients", 16, "--horizon-days", 14,
                     "--seed", 445566, "--patient-output", "--out", output)
        comparisons = read_json(output / "comparison.json")
        expected = {"a2_low", "base", "a2_high", "no_ci", "zero_headstart", "no_ci_peak", "no_peak",
                    "reverse_headstarts", "equal_headstarts"}
        self.assertEqual({item["label"] for item in comparisons}, expected)
        results = {name: read_json(output / name / "result.json") for name in expected}
        for result in results.values():
            self.assertEqual(result["run_config"]["seed"], 445566)
            self.assertEqual(result["run_config"]["outer_runs"], 1)
            self.assertEqual(result["runs"][0]["seeds"], results["base"]["runs"][0]["seeds"])
        for name in ("no_ci", "zero_headstart"):
            self.assertEqual(results[name]["summary"]["ap_dles_per_visitor"]["mean"], 0)
            self.assertEqual(results[name]["summary"]["am_dles_per_monthly_cohort"]["mean"], 0)
            self.assertTrue(all(row["delta_dles"] == 0 for row in results[name]["patients"]))
        for low, high in zip(results["a2_low"]["patients"], results["a2_high"]["patients"]):
            self.assertEqual(low["clinical_seed"], high["clinical_seed"])
            self.assertEqual(low["help_seeking"], high["help_seeking"])
            self.assertEqual(low["no_ci_dles"], high["no_ci_dles"])
            self.assertFalse(low["acts"] and not high["acts"])
        with (output / "comparison.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["label"] for row in rows}, expected)

    def test_invalid_configs_fail_cleanly_before_writing_results(self):
        invalid = (
            ('{"run":{"seed":1,"seed":2}}', "duplicate JSON key"),
            ('{"run":{"unknown":1}}', "unknown"),
            ('{"model":{"unknown":1}}', "unknown"),
            ('{"other":{}}', "config must be"),
            ('null', "config must be"),
            ('{"run":null}', "configuration"),
            ('{"model":null}', "config must be an object"),
            ('{"run":{"a2":NaN}}', "finite"),
            ('{"model":{"horizon_days":Infinity}}', "finite"),
        )
        for index, (text, message) in enumerate(invalid):
            with self.subTest(config=text):
                config = self.directory / f"invalid-{index}.json"
                config.write_text(text)
                output = self.directory / f"invalid-output-{index}"
                result = self.run_cli("simulate", "--config", config, "--out", output, success=False)
                self.assertIn(message, result.stderr)
                self.assertFalse((output / "result.json").exists())

    def test_invalid_arguments_are_errors_not_tracebacks(self):
        cases = (
            ("simulate", "--patients", "0"),
            ("simulate", "--outer-runs", "0"),
            ("simulate", "--a2", "nan"),
            ("simulate", "--horizon-days", "inf"),
            ("simulate", "--scenario", "missing"),
            ("bootstrap", "--draws", "0"),
        )
        for index, arguments in enumerate(cases):
            with self.subTest(arguments=arguments):
                output = self.directory / f"bad-arguments-{index}"
                self.run_cli(*arguments, "--out", output, success=False)
                self.assertFalse((output / "result.json").exists())
                self.assertFalse((output / "bootstrap.json").exists())


if __name__ == "__main__":
    unittest.main()
