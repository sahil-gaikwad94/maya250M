"""Tests for the five domain batteries: the verifiers must not be vacuous.

Each battery is validated from both sides — a gold solution must pass, and a
plausible-but-wrong solution must fail. A verifier that accepts everything would
otherwise inflate every future score without anyone noticing.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.evaluation.domains import (
    DOMAINS,
    all_tasks,
    completion_syntax_verifier,
    domain_tasks,
    exact_answer_verifier,
    expressions_by_prompt,
    marker_self_consistency_verifier,
)
from maya250m.evaluation.harness import run_tasks
from maya250m.evaluation.identity import MeasurementIdentity, MeasurementStatus
from maya250m.evaluation.symbolic import SymbolicSolver, symbolic_repair_generate


class TestBatteryShape(unittest.TestCase):
    def test_five_domains_unique_tasks_benchmark_identity(self):
        batteries = domain_tasks()
        self.assertEqual(tuple(batteries), DOMAINS)
        tasks = all_tasks()
        self.assertGreaterEqual(len(tasks), 20)
        self.assertEqual(len({task.task_id for task in tasks}), len(tasks))
        for task in tasks:
            self.assertIs(task.identity, MeasurementIdentity.BENCHMARK)
            self.assertTrue(task.prompt.strip())

    def test_unknown_domain_is_rejected(self):
        with self.assertRaises(KeyError):
            domain_tasks("astrology")

    def test_missing_generator_reports_not_run_without_a_value(self):
        report = run_tasks(domain_tasks("maths")["maths"], generate=None, title="maths")
        self.assertIs(report.status, MeasurementStatus.NOT_RUN)
        for measurement in report.measurements:
            self.assertIsNone(measurement.value)


class TestGoldSolutions(unittest.TestCase):
    def test_gold_passes_executable_batteries(self):
        checked = 0
        for domain, tasks in domain_tasks().items():
            for task in tasks:
                gold = task.metadata.get("gold")
                if not gold:
                    continue
                sample = task.prompt + gold if domain in ("coding", "reasoning") else gold
                result = task.verifier(sample)
                self.assertTrue(result.passed, f"{task.task_id}: {result.to_dict()}")
                checked += 1
        self.assertGreaterEqual(checked, 14)

    def test_gold_generator_scores_one(self):
        tasks = domain_tasks("coding")["coding"]
        by_prompt = {task.prompt: task for task in tasks}

        def generate(prompt: str) -> str:
            return prompt + by_prompt[prompt].metadata["gold"]

        report = run_tasks(tasks, generate=generate, title="coding/gold")
        self.assertAlmostEqual(report.require_measured("pass@1"), 1.0)


class TestBrokenSolutions(unittest.TestCase):
    def test_wrong_function_is_rejected(self):
        task = next(t for t in domain_tasks("coding")["coding"] if t.task_id == "coding_add")
        result = task.verifier(task.prompt + "    return a - b\n")
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "tests_failed")

    def test_empty_completion_is_rejected(self):
        task = next(t for t in domain_tasks("coding")["coding"] if t.task_id == "coding_square")
        result = task.verifier(task.prompt)
        self.assertFalse(result.passed)

    def test_forbidden_import_fails_reasoning_task(self):
        task = next(
            t
            for t in domain_tasks("reasoning")["reasoning"]
            if t.task_id == "reasoning_is_even"
        )
        sample = task.prompt + "    import math\n    return n % 2 == 0\n"
        result = task.verifier(sample)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "ast_policy_violation")

    def test_broken_generator_scores_zero(self):
        tasks = domain_tasks("coding")["coding"]
        report = run_tasks(
            tasks, generate=lambda prompt: prompt + "    return None\n", title="coding/broken"
        )
        self.assertAlmostEqual(report.require_measured("pass@1"), 0.0)

    def test_wrong_and_missing_maths_answers_fail(self):
        task = next(
            t for t in domain_tasks("maths")["maths"] if t.task_id == "maths_mul_add"
        )
        self.assertFalse(task.verifier("answer: 399\n").passed)
        self.assertFalse(task.verifier("I am not sure.\n").passed)
        self.assertTrue(task.verifier("answer: 400\n").passed)

    def test_marker_consistency_is_checked(self):
        verifier = marker_self_consistency_verifier()
        self.assertFalse(verifier("<<2 + 2 = 5>>").passed)
        self.assertTrue(verifier("<<2 + 2 = 4>>").passed)
        self.assertTrue(verifier("no marker here").passed)

    def test_exact_answer_rejects_wrong_denominator(self):
        verifier = exact_answer_verifier("2/3")
        self.assertTrue(verifier("answer: 2/3").passed)
        self.assertFalse(verifier("answer: 0.66").passed)
        self.assertTrue(verifier("<< (3/4) * (8/9) = 2/3 >>").passed)


class TestNeuralSyntaxVerifier(unittest.TestCase):
    def test_valid_continuation_passes(self):
        stem = 'def total(items):\n    """Sum every item."""\n'
        verifier = completion_syntax_verifier(stem)
        self.assertTrue(verifier(stem + "    return sum(items)\n").passed)

    def test_broken_or_empty_continuation_fails(self):
        stem = 'def total(items):\n    """Sum every item."""\n'
        verifier = completion_syntax_verifier(stem)
        self.assertFalse(verifier(stem + "    return (((\n").passed)
        self.assertFalse(verifier(stem).passed)
        self.assertFalse(verifier("unrelated text that does not echo the prompt").passed)

    def test_ignoring_the_prompt_does_not_count_as_ability(self):
        """The real failure this battery caught: a memorized unrelated function."""
        stem = 'def total(items):\n    """Sum every item."""\n'
        verifier = completion_syntax_verifier(stem)
        injected = stem + 'def sum_to_26(n: int) -> int:\n    return n * (n + 1) // 2\n'
        result = verifier(injected)
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, "schema_violation")
        self.assertTrue(verifier(stem + "    return sum(items)\n").passed)

    def test_docstring_alone_is_not_an_implementation(self):
        stem = "def last_item(values):\n"
        verifier = completion_syntax_verifier(stem)
        self.assertFalse(verifier(stem + '    """Return the final element."""\n').passed)

    def test_completion_diversity_detects_mode_collapse(self):
        from maya250m.evaluation.domains import completion_diversity

        collapsed = completion_diversity(["same text"] * 6)
        self.assertEqual(collapsed["unique"], 1)
        self.assertTrue(collapsed["collapse_detected"])

        # The measured failure: one template, different constants only.
        numeric = completion_diversity(
            [f"def sum_to_{k}(n: int) -> int:\n    return n * (n + 1) // 2" for k in (21, 26, 32)]
        )
        self.assertEqual(numeric["unique"], 1)
        self.assertTrue(numeric["collapse_detected"])

        varied = completion_diversity(
            ["return x + y", "raise ValueError", "print(name)"]
        )
        self.assertEqual(varied["unique_ratio"], 1.0)
        self.assertFalse(varied["collapse_detected"])


class TestSandboxPolicyAcceptsOrdinaryCode(unittest.TestCase):
    """The battery exposed the policy rejecting correct, safe Python."""

    def test_string_literal_methods_are_allowed(self):
        from maya250m.vexx import Vexx35Controller

        code = (
            'def reverse_words(text: str) -> str:\n'
            '    return " ".join(reversed(text.split()))\n'
        )
        Vexx35Controller._check_ast_policy(code)

    def test_escape_hatches_are_still_blocked(self):
        from maya250m.vexx import Vexx35Controller

        for code in (
            'value = "".__class__\n',
            'value = "{}".format(1)\n',
            'value = "{0.__class__}".format(1)\n',
            'value = "x".encode()\n',
            'import os\n',
        ):
            with self.assertRaises(PermissionError, msg=code):
                Vexx35Controller._check_ast_policy(code)


class TestSymbolicPath(unittest.TestCase):
    def test_solver_derives_every_flagged_expression(self):
        solver = SymbolicSolver()
        for domain in ("maths", "research"):
            for task in domain_tasks(domain)[domain]:
                expression = task.metadata["expression"]
                derived = solver.solve(expression)
                self.assertIsNotNone(derived, expression)
                self.assertTrue(
                    task.verifier(derived).passed, f"{task.task_id} derived {derived!r}"
                )
        self.assertEqual(solver.failures, 0)

    def test_repair_is_scoped_to_flagged_tasks(self):
        coding = domain_tasks("coding")["coding"]
        self.assertEqual(expressions_by_prompt(coding), {})

        calls = []

        def raw(prompt: str) -> str:
            calls.append(prompt)
            return prompt + "    return a - b\n"

        repaired = symbolic_repair_generate(raw, expressions_by_prompt(coding))
        task = next(t for t in coding if t.task_id == "coding_add")
        self.assertEqual(repaired(task.prompt), raw(task.prompt))
        self.assertFalse(task.verifier(repaired(task.prompt)).passed)

    def test_repair_appends_answer_to_maths_attempt(self):
        maths = domain_tasks("maths")["maths"]
        task = next(t for t in maths if t.task_id == "maths_mul_add")
        repaired = symbolic_repair_generate(
            lambda prompt: prompt + "answer: 1\n",
            expressions_by_prompt(maths),
        )
        sample = repaired(task.prompt)
        self.assertIn("answer: 400", sample)
        self.assertTrue(task.verifier(sample).passed)

    def test_unparsable_expression_is_a_miss_not_a_crash(self):
        solver = SymbolicSolver()
        self.assertIsNone(solver.solve("os.system('rm -rf /')"))
        self.assertEqual(solver.failures, 1)
        self.assertEqual(solver.derivations, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
