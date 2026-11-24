"""12.4.3 tests: executable statistical and dimensional consistency audit."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.vexx import Vexx35Controller


class TestGRIMGranularity(unittest.TestCase):
    def test_impossible_percentage_is_flagged(self):
        result = Vexx35Controller.verify_statistical_report(
            "34.5% of N=10 patients improved on the new drug."
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "CONFLICT")
        self.assertEqual(result["type"], "grim_violation")
        self.assertIn("FINAL: CONFLICT", result["explanation"])

    def test_feasible_percentage_passes(self):
        self.assertIsNone(
            Vexx35Controller.verify_statistical_report("30% of N=10 patients improved.")
        )

    def test_impossible_mean_total_is_flagged(self):
        result = Vexx35Controller.verify_statistical_report(
            "mean = 2.5 errors per subject, N = 3 subjects."
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "grim_violation")


class TestTestStatisticConsistency(unittest.TestCase):
    def test_impossible_z_p_pair_is_flagged(self):
        result = Vexx35Controller.verify_statistical_report(
            "The effect was not significant (z = 0.4, p = 0.0001)."
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "statistical_inconsistency")
        self.assertEqual(result["verdict"], "CONFLICT")

    def test_consistent_z_p_pair_passes(self):
        self.assertIsNone(
            Vexx35Controller.verify_statistical_report("z = 1.96, p = 0.05")
        )

    def test_impossible_t_p_pair_is_flagged(self):
        result = Vexx35Controller.verify_statistical_report(
            "t = 0.4, df = 10, p = 0.0001"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "statistical_inconsistency")

    def test_chi_square_consistency(self):
        self.assertIsNone(
            Vexx35Controller.verify_statistical_report("chi2 = 3.84, df = 1, p = 0.05")
        )
        flagged = Vexx35Controller.verify_statistical_report(
            "chi2 = 0.5, df = 4, p = 0.0001"
        )
        self.assertIsNotNone(flagged)
        self.assertEqual(flagged["type"], "statistical_inconsistency")

    def test_chi_square_survival_is_exact(self):
        self.assertAlmostEqual(
            Vexx35Controller._chi2_survival(3.84, 1), 0.0500, places=3
        )
        self.assertAlmostEqual(
            Vexx35Controller._chi2_survival(0.5, 4), 0.9735, places=3
        )



    def test_alpha_boundary_flip_and_moderate_p_discrepancy_are_caught(self):
        # z = 1.60 implies two-sided p = 0.1096 (> 0.05); reporting p = 0.04 (< 0.05) is a 2.74x ratio
        # that v4 missed because ratio < 3.0, but flips significance across alpha = 0.05!
        flagged = Vexx35Controller.verify_statistical_report("z = 1.60, p = 0.04")
        self.assertIsNotNone(flagged)
        self.assertEqual(flagged["type"], "statistical_inconsistency")

        # Exact Student-t survival for df = 5: t = 2.0 implies two-sided p = 0.1019; reporting p = 0.04 is caught!
        flagged_t = Vexx35Controller.verify_statistical_report("t = 2.0, df = 5, p = 0.04")
        self.assertIsNotNone(flagged_t)
        self.assertEqual(flagged_t["type"], "statistical_inconsistency")

        # Exact Student-t for t = 2.571, df = 5 implies p = 0.0500 -> passes!
        self.assertIsNone(
            Vexx35Controller.verify_statistical_report("t = 2.571, df = 5, p = 0.05")
        )


class TestDimensionalHomogeneity(unittest.TestCase):
    def test_unit_vectors(self):
        self.assertEqual(
            Vexx35Controller.parse_unit_dimensions("kg*m/s^2"),
            {"M": 1, "L": 1, "T": -2},
        )
        self.assertEqual(
            Vexx35Controller.parse_unit_dimensions("N"),
            {"M": 1, "L": 1, "T": -2},
        )
        self.assertIsNone(Vexx35Controller.parse_unit_dimensions("bananas"))

    def test_inhomogeneous_equation_is_flagged(self):
        result = Vexx35Controller.verify_statistical_report(
            "Check the units: 5 kg*m/s^2 = 5 kg*m/s"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "dimensional_mismatch")
        self.assertEqual(result["verdict"], "CONFLICT")

    def test_homogeneous_equation_passes(self):
        self.assertIsNone(
            Vexx35Controller.verify_statistical_report(
                "Check the units: 5 kg*m/s^2 = 5 N"
            )
        )


class TestCalibrationIntegration(unittest.TestCase):
    def test_new_checks_run_before_the_legacy_heuristics(self):
        result = Vexx35Controller.verify_epistemic_calibration(
            "34.5% of N=10 patients improved (z = 0.4, p = 0.0001)."
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["verdict"], "CONFLICT")

    def test_legacy_ci_p_behaviour_is_preserved(self):
        result = Vexx35Controller.verify_epistemic_calibration(
            "95% CI [-1.2, 0.4], p = 0.01"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "statistical_conflict")


if __name__ == "__main__":
    unittest.main(verbosity=2)
