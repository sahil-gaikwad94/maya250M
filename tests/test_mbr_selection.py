"""12.2.2 tests: execution-based MBR consensus selection."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.vexx import Vexx35Controller


DOUBLE = "def f(xs):\n    return [n * 2 for n in xs]\n"
DOUBLE_VARIANT = "def f(xs):\n    out = []\n    for n in xs:\n        out.append(n * 2)\n    return out\n"
WRONG = "def f(xs):\n    return [n + 1 for n in xs]\n"
MUTATING = "def f(xs):\n    xs.append(9)\n    return xs\n"
BROKEN = "def f(xs)\n    return xs\n"


class TestMBRSelection(unittest.TestCase):
    def setUp(self):
        self.vexx = Vexx35Controller()
        self.probes = [[[3, 1]], [[5]]]

    def test_majority_output_tuple_wins(self):
        result = self.vexx.select_by_mbr([DOUBLE, WRONG, DOUBLE_VARIANT], "f", self.probes)
        self.assertEqual(result["outputs"], ["[6, 2]", "[10]"])
        self.assertEqual(result["votes"], 2)
        self.assertEqual(result["survivors"], 3)
        self.assertTrue(result["consensus"])
        self.assertAlmostEqual(result["agreement"], 2 / 3)
        # The shorter, simpler candidate wins the tie inside the majority group.
        self.assertEqual(result["winner"], DOUBLE)

    def test_policy_and_contract_violations_are_filtered_before_execution(self):
        result = self.vexx.select_by_mbr([BROKEN, DOUBLE], "f", self.probes)
        syntax_row = result["scoreboard"][0]
        self.assertFalse(syntax_row["policy_ok"])
        self.assertFalse(syntax_row["executed"])
        self.assertEqual(result["survivors"], 1)
        # A lone survivor is never promoted to a verified consensus.
        self.assertFalse(result["consensus"])

    def test_input_mutation_candidates_are_not_eligible(self):
        result = self.vexx.select_by_mbr([MUTATING, DOUBLE, DOUBLE_VARIANT], "f", self.probes)
        mutation_row = result["scoreboard"][0]
        self.assertFalse(mutation_row["mutation_free"])
        self.assertEqual(result["survivors"], 2)
        self.assertEqual(result["winner"], DOUBLE)
        self.assertTrue(result["consensus"])

    def test_tie_breaks_on_shortest_ast(self):
        shallow = "def f(xs):\n    return len(xs)\n"
        deep = "def f(xs):\n    if True:\n        return len(xs) + 1\n"
        result = self.vexx.select_by_mbr([deep, shallow], "f", self.probes)
        self.assertEqual(result["winner"], shallow)
        self.assertEqual(result["votes"], 1)
        self.assertFalse(result["consensus"])
        self.assertAlmostEqual(result["agreement"], 0.5)

    def test_empty_probe_set_is_rejected(self):
        with self.assertRaises(ValueError):
            self.vexx.select_by_mbr([DOUBLE], "f", [])



    def test_posterior_log_probs_shift_bayes_risk(self):
        # With uniform prior, DOUBLE and WRONG tie 1-1 on votes, broken by AST depth.
        # With strong posterior log-prob on WRONG, WRONG has lower expected Bayes risk.
        res = self.vexx.select_by_mbr(
            [DOUBLE, WRONG], "f", self.probes, log_probs=[-5.0, -0.1], temperature=1.0
        )
        self.assertEqual(res["winner"], WRONG)
        self.assertLess(res["bayes_risk"], 0.1)
        self.assertGreater(res["expected_utility"], 0.9)

    def test_partial_probe_agreement_minimizes_hamming_bayes_risk(self):
        c1 = "def f(x):\n    return [1, 2, 3][x]\n"
        c2 = "def f(x):\n    return [1, 2, 9][x]\n"
        c3 = "def f(x):\n    return [1, 8, 3][x]\n"
        # All three have distinct 3-probe output tuples (1 vote each), but c1 agrees
        # with c2 on 2/3 probes and with c3 on 2/3 probes -> lowest Bayes risk (2/9 vs 3/9)!
        res = self.vexx.select_by_mbr([c2, c3, c1], "f", [[0], [1], [2]])
        self.assertEqual(res["winner"], c1)
        self.assertAlmostEqual(res["bayes_risk"], 2.0 / 9.0, places=6)

    def test_lexer_state_preserves_backslash_escape_across_token_chunks(self):
        from maya250m.vexx import LexerState, advance_lexer_state

        st1 = advance_lexer_state(LexerState(), 's = "hello\\')
        self.assertEqual(st1.in_str, '"')
        self.assertTrue(st1.escape)
        st2 = advance_lexer_state(st1, '" + "world"')
        # The quote after the trailing backslash was escaped, so " + " opened/closed a string
        # and the final quote closed the string cleanly!
        full = advance_lexer_state(LexerState(), 's = "hello\\" + "world"')
        self.assertEqual(st2.in_str, full.in_str)
        self.assertEqual(st2.valid, full.valid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
