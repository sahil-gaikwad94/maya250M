"""Phase 5 tests: measurement identities, executable harness, NOT_RUN semantics, A-P matrix."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.config import get_config
from maya250m.evaluation import (
    ABLATIONS,
    EvalReport,
    EvalTask,
    Measurement,
    MeasurementError,
    MeasurementIdentity,
    MeasurementStatus,
    ablation_table,
    apply_ablation,
    constraint_verifier,
    contains_all_verifier,
    exact_match_verifier,
    pass_at_k,
    run_ablation,
    run_ablation_matrix,
    run_tasks,
    unit_test_verifier,
)
from maya250m.understanding import parse_task_spec
from maya250m.vexx import Vexx35Controller


class TestMeasurementIdentities(unittest.TestCase):
    def test_unrun_measurement_cannot_carry_a_score(self):
        with self.assertRaises(MeasurementError):
            Measurement(
                identity=MeasurementIdentity.BENCHMARK,
                name="humaneval",
                value=0.0,
                status=MeasurementStatus.NOT_RUN,
            )

    def test_measured_requires_value(self):
        with self.assertRaises(MeasurementError):
            Measurement(identity=MeasurementIdentity.BENCHMARK, name="humaneval")

    def test_reports_refuse_to_mix_identities(self):
        report = EvalReport(title="val")
        report.add(
            Measurement(
                identity=MeasurementIdentity.HELD_OUT_VALIDATION,
                name="val_ppl",
                value=12.5,
                n=100,
            )
        )
        with self.assertRaises(MeasurementError):
            report.add(
                Measurement(identity=MeasurementIdentity.BENCHMARK, name="pass@1", value=0.4, n=50)
            )

    def test_compare_requires_same_identity(self):
        a = EvalReport(title="a")
        a.add(Measurement(MeasurementIdentity.BENCHMARK, "pass@1", 0.4, n=10))
        b = EvalReport(title="b")
        b.add(Measurement(MeasurementIdentity.HELD_OUT_VALIDATION, "val_ppl", 9.0, n=10))
        with self.assertRaises(MeasurementError):
            a.compare_with(b)

    def test_not_run_report_refuses_numeric_claims(self):
        report = EvalReport(title="bench")
        report.add(
            Measurement(MeasurementIdentity.BENCHMARK, "humaneval", status=MeasurementStatus.NOT_RUN)
        )
        self.assertEqual(report.status, MeasurementStatus.NOT_RUN)
        with self.assertRaises(MeasurementError):
            report.require_measured("humaneval")


class TestHarness(unittest.TestCase):
    def _tasks(self):
        return [
            EvalTask("exact", "2+2=?", exact_match_verifier("4")),
            EvalTask("contains", "name three colors", contains_all_verifier(["red", "green"])),
        ]

    def test_no_generator_means_not_run(self):
        report = run_tasks(self._tasks(), generate=None)
        self.assertEqual(report.status, MeasurementStatus.NOT_RUN)
        for measurement in report.measurements:
            self.assertIsNone(measurement.value)

    def test_generator_is_verified_not_trusted(self):
        answers = {"2+2=?": "4", "name three colors": "red, green, blue"}
        report = run_tasks(self._tasks(), generate=lambda prompt: answers[prompt])
        self.assertEqual(report.require_measured("exact"), 1.0)
        self.assertEqual(report.require_measured("contains"), 1.0)
        self.assertEqual(report.require_measured("pass@1"), 1.0)

        wrong = run_tasks(self._tasks(), generate=lambda prompt: "nothing useful")
        self.assertEqual(wrong.require_measured("pass@1"), 0.0)

    def test_generator_crash_is_a_failure_not_a_pass(self):
        def boom(_prompt):
            raise RuntimeError("model died")

        report = run_tasks(self._tasks(), generate=boom)
        self.assertEqual(report.require_measured("pass@1"), 0.0)
        self.assertEqual(report.get("exact").metadata["passed_samples"], 0)

    def test_pass_at_k_math(self):
        self.assertAlmostEqual(pass_at_k(10, 2, 1), 0.2)
        self.assertAlmostEqual(pass_at_k(5, 5, 3), 1.0)
        self.assertAlmostEqual(pass_at_k(4, 1, 2), 0.5)
        self.assertEqual(pass_at_k(10, 0, 1), 0.0)
        with self.assertRaises(ValueError):
            pass_at_k(2, 3, 1)

    def test_constraint_verifier_uses_parsed_spec(self):
        spec = parse_task_spec("Write a function that must not add new dependencies.")
        verifier = constraint_verifier(spec)
        # constraint_coverage is a documented lexical *addressing* audit.
        self.assertTrue(verifier("# no new dependencies added\ndef f(): return 1").passed)
        self.assertFalse(verifier("def f(): return 1").passed)

    def test_unit_test_verifier_executes_code(self):
        vexx = Vexx35Controller()
        verifier = unit_test_verifier(
            "add", [{"input": [2, 3], "expected": 5}, {"input": [-1, 1], "expected": 0}], vexx=vexx
        )
        self.assertTrue(verifier("def add(a, b):\n    return a + b\n").passed)
        self.assertFalse(verifier("def add(a, b):\n    return a - b\n").passed)


class TestAblationMatrix(unittest.TestCase):
    def test_sixteen_named_ablations(self):
        self.assertEqual(sorted(ABLATIONS), [chr(ord("A") + i) for i in range(26)])
        for key, spec in ABLATIONS.items():
            self.assertTrue(spec.name)
            self.assertTrue(spec.description)

    def test_apply_ablation_is_pure_and_validated(self):
        base = get_config("MAYA_TINY_TEST")
        untouched = dict(base)
        ablated = apply_ablation(base, "E")
        self.assertEqual(base, untouched)
        self.assertEqual(ablated["mtp_depth"], 0)

        full = apply_ablation(base, "F")
        self.assertEqual(set(full["hybrid_schedule"]), {"full"})
        self.assertEqual(len(full["hybrid_schedule"]), int(base["layers"]))

        reduced = apply_ablation(base, "D")
        self.assertEqual(reduced["recurrent_loops"], 1)

        with self.assertRaises(KeyError):
            apply_ablation(base, "ZZ")

    def test_every_ablation_produces_a_valid_config(self):
        for preset in ("MAYA_TINY_TEST", "MAYA_250M"):
            rows = ablation_table(preset)
            self.assertEqual(len(rows), 26)
            self.assertTrue(all(row["valid_config"] for row in rows), preset)

    def test_unrun_ablation_has_no_metric(self):
        base = get_config("MAYA_TINY_TEST")
        result = run_ablation("H", base)
        self.assertEqual(result.status, MeasurementStatus.NOT_RUN)
        measurement = result.to_measurement()
        self.assertIsNone(measurement.value)
        self.assertEqual(measurement.status, MeasurementStatus.NOT_RUN)
        self.assertIn("no training", result.reason)

    def test_ablation_with_trainer_measures(self):
        base = get_config("MAYA_TINY_TEST")

        def fake_train(config, spec):
            return {"score": 0.5 if spec.disables else 1.0, "params": 1.0}

        results = run_ablation_matrix(base, keys=["A", "B"], train_fn=fake_train)
        self.assertEqual([r.key for r in results], ["A", "B"])
        self.assertEqual(results[0].status, MeasurementStatus.MEASURED)
        self.assertAlmostEqual(results[1].metrics["score"], 0.5)

    def test_failed_ablation_is_recorded_as_failed(self):
        base = get_config("MAYA_TINY_TEST")

        def boom(config, spec):
            raise RuntimeError("OOM")

        result = run_ablation("J", base, train_fn=boom)
        self.assertEqual(result.status, MeasurementStatus.FAILED)
        self.assertIn("OOM", result.reason)
        self.assertIsNone(result.to_measurement().value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
