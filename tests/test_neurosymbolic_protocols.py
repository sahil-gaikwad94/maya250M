"""12.2.1 / 12.4.2 tests: differential fuzzing and the counterexample-first protocol."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.reasoning import (
    RESEARCH_STRATEGY,
    StrategySelector,
    certify_universal_claim,
    counterexample_first_plan,
    decompose_task,
)
from maya250m.reasoning.decompose import DecompositionError
from maya250m.understanding import TaskSpec
from maya250m.vexx import Vexx35Controller


SORT_OK = (
    "def solve(xs):\n"
    "    out = []\n"
    "    for value in xs:\n"
    "        out.append(value)\n"
    "    return sorted(out)\n"
)
SORT_MUTATING = (
    "def solve(xs):\n"
    "    xs.sort()\n"
    "    return xs\n"
)
SORT_OFF_BY_ONE = (
    "def solve(xs):\n"
    "    visited = set()\n"
    "    out = []\n"
    "    for value in xs:\n"
    "        if value not in visited:\n"
    "            out.append(value)\n"
    "    return sorted(out)\n"
)
ORACLE_DEDUP = (
    "def solve(xs):\n"
    "    out = []\n"
    "    for value in sorted(xs):\n"
    "        if not out or out[-1] != value:\n"
    "            out.append(value)\n"
    "    return out\n"
)


class TestDifferentialFuzzing(unittest.TestCase):
    def setUp(self):
        self.vexx = Vexx35Controller()

    def test_matching_candidate_agrees_with_oracle(self):
        result = self.vexx.fuzz_against_oracle(SORT_OK, SORT_OK, "solve")
        self.assertTrue(result["agrees"])
        self.assertEqual(result["checked"], len(Vexx35Controller.BOUNDARY_PROBES))
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(result["errors"], [])

    def test_behavioural_difference_is_reported_with_the_failing_input(self):
        result = self.vexx.fuzz_against_oracle(SORT_OK, ORACLE_DEDUP, "solve")
        self.assertFalse(result["agrees"])
        self.assertTrue(result["mismatches"])
        failing = result["mismatches"][0]
        self.assertIn("input", failing)
        self.assertIn("expected", failing)
        self.assertIn("actual", failing)

    def test_candidate_crash_is_an_error_not_an_agreement(self):
        result = self.vexx.fuzz_against_oracle(
            "def solve(xs):\n    return undefined_name(xs)\n", SORT_OK, "solve"
        )
        self.assertFalse(result["agrees"])
        self.assertTrue(result["errors"])
        self.assertEqual(result["errors"][0]["side"], "candidate")

    def test_broken_oracle_fails_closed(self):
        result = self.vexx.fuzz_against_oracle(
            SORT_OK, "def solve(xs):\n    return xs[99]\n", "solve"
        )
        self.assertFalse(result["agrees"])
        self.assertEqual(result["errors"][0]["side"], "reference")

    def test_explicit_probes_are_used(self):
        result = self.vexx.fuzz_against_oracle(
            SORT_MUTATING, SORT_OK, "solve", inputs=[[[2, 1]]]
        )
        self.assertEqual(result["checked"], 1)


class TestCounterexampleFirstProtocol(unittest.TestCase):
    def test_plan_orders_sweep_fit_falsify_prove(self):
        plan = counterexample_first_plan("n^2 + n + 41 is prime", sweep_n=15, falsify_n=500)
        self.assertEqual(plan.strategy, RESEARCH_STRATEGY)
        self.assertEqual(
            plan.topological_order(), ["sweep", "fit", "falsify", "prove"]
        )
        self.assertEqual(plan.by_id("prove").depends_on, ["falsify"])
        self.assertIn("N=500", plan.by_id("falsify").description)

    def test_plan_rejects_an_inverted_sweep(self):
        with self.assertRaises(DecompositionError):
            counterexample_first_plan("claim", sweep_n=50, falsify_n=10)

    def test_claim_certification_levels(self):
        refuted = certify_universal_claim(40, counterexample=(40, 1681))
        self.assertEqual(refuted["verdict"], "REFUTED")
        self.assertFalse(refuted["may_claim_for_all_n"])

        finite = certify_universal_claim(500)
        self.assertEqual(finite["verdict"], "CONJECTURE_ONLY")
        self.assertFalse(finite["may_claim_for_all_n"])
        self.assertIn("500", finite["explanation"])

        proven = certify_universal_claim(15, identity_verified=True)
        self.assertEqual(proven["verdict"], "PROVEN")
        self.assertTrue(proven["may_claim_for_all_n"])

    def test_selector_routes_conjectures_to_research_strategy(self):
        conjecture_spec = TaskSpec(objective="Is every n^2+n+41 prime?", task_type="research")
        self.assertEqual(StrategySelector().select(conjecture_spec), RESEARCH_STRATEGY)
        plan = decompose_task(conjecture_spec)
        self.assertEqual(plan.strategy, RESEARCH_STRATEGY)
        self.assertEqual(plan.topological_order(), ["sweep", "fit", "falsify", "prove"])

        lit_spec = TaskSpec(objective="Summarize the paper and cite the evidence.", task_type="research")
        self.assertEqual(StrategySelector().select(lit_spec), "retrieve_then_answer")
        self.assertEqual(decompose_task(lit_spec).strategy, "retrieve_then_answer")

    def test_lagrange_hypothesis_synthesis_and_farkas_polytope(self):
        vexx = Vexx35Controller()
        hyp = vexx.synthesize_polynomial_hypothesis(
            [(1, 43), (2, 47), (3, 53)], var_name="n", holdout_points=[(4, 61), (5, 71)]
        )
        self.assertEqual(hyp["expression"], "n**2 + n + 41")
        self.assertEqual(hyp["degree"], 2)
        self.assertTrue(hyp["holdout_verified"])

        # Multi-row Farkas infeasibility check: x + y <= 3 and x + y >= 5
        infeasible = vexx.propagate_rational_polytope(
            [
                {"coeffs": {"x": 1, "y": 1}, "rhs": 3, "op": "<="},
                {"coeffs": {"x": 1, "y": 1}, "rhs": 5, "op": ">="},
            ]
        )
        self.assertFalse(infeasible["feasible"])
        self.assertEqual(infeasible["conflict_rows"], [0, 1])
        self.assertTrue(infeasible["farkas_certificate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
