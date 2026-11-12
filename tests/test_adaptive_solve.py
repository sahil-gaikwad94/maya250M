"""Reasoning-aware inference tests: adaptive depth, abstention, bounded revision."""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.adaptive import (
    SolveConfig,
    generate_adaptive,
    select_loops,
    solve_task,
)
from maya250m.config import get_config
from maya250m.evaluation import MeasurementIdentity, MeasurementStatus, exact_match_verifier
from maya250m.generate import generate_maya
from maya250m.model import MayaModel
from maya250m.reasoning.controller import AdaptiveLoopController
from maya250m.tokenizer import BPETokenizer
from maya250m.understanding import parse_task_spec

HARD_PROMPT = (
    "Fix the bug in maya250m/model.py and maya250m/train.py where prefill and "
    "decode_step disagree about the cache schema. It must preserve backward "
    "compatibility with schema-9 checkpoints and must not add new dependencies."
)
SIMPLE_PROMPT = "Print the number 4."


def _tiny_model(tokenizer: BPETokenizer, seed: int = 0) -> MayaModel:
    torch.manual_seed(seed)
    cfg = get_config("MAYA_TINY_TEST")
    cfg["vocab"] = tokenizer.vocab_size
    cfg["max_ctx"] = 64
    cfg["recurrent_loops"] = 1
    return MayaModel(cfg, tokenizer_hash=tokenizer.tokenizer_hash())


class TestLoopSelection(unittest.TestCase):
    def test_harder_requests_get_at_least_as_many_loops(self):
        controller = AdaptiveLoopController()
        simple_spec = parse_task_spec(SIMPLE_PROMPT)
        hard_spec = parse_task_spec(HARD_PROMPT)
        simple, simple_explain = select_loops(simple_spec, prompt_tokens=8, controller=controller)
        hard, hard_explain = select_loops(hard_spec, prompt_tokens=90, controller=controller)
        self.assertGreaterEqual(simple, 1)
        self.assertLessEqual(simple, 4)
        self.assertGreaterEqual(hard, simple)
        self.assertGreaterEqual(hard, 2)
        self.assertEqual(hard_explain["mode"], "adaptive")
        self.assertGreater(hard_explain["difficulty_score"], simple_explain["difficulty_score"])

    def test_disabled_adaptive_depth_uses_fixed_loops(self):
        spec = parse_task_spec(HARD_PROMPT)
        loops, explain = select_loops(spec, 90, adaptive=True, default_loops=2, disabled={"adaptive_depth"})
        self.assertEqual(loops, 2)
        self.assertEqual(explain["mode"], "fixed")

    def test_prior_failures_increase_depth(self):
        from maya250m.reasoning.state import ReasoningState

        spec = parse_task_spec(HARD_PROMPT)
        controller = AdaptiveLoopController()
        clean, _ = select_loops(spec, 90, controller=controller)
        state = ReasoningState(objective="x")
        state.rejected_approaches = ["tests_failed: boom", "tests_failed: again"]
        retried, _ = select_loops(spec, 90, controller=controller, state=state)
        self.assertGreaterEqual(retried, clean)


class TestGenerationWithLoops(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BPETokenizer()
        cls.model = _tiny_model(cls.tokenizer, seed=4)

    def test_requested_loop_count_is_executed_and_recorded(self):
        one = generate_maya(self.model, self.tokenizer, "x = 1", max_new_tokens=4, num_loops=1)
        three = generate_maya(self.model, self.tokenizer, "x = 1", max_new_tokens=4, num_loops=3)
        self.assertEqual(one.num_loops, 1)
        self.assertEqual(three.num_loops, 3)
        self.assertEqual(len(one.continuation), len(three.continuation))

    def test_default_matches_config_and_is_deterministic(self):
        a = generate_maya(self.model, self.tokenizer, "x = 1", max_new_tokens=4)
        b = generate_maya(self.model, self.tokenizer, "x = 1", max_new_tokens=4)
        self.assertEqual(a.num_loops, int(self.model.recurrent_loops))
        self.assertEqual(a.continuation, b.continuation)

    def test_generate_adaptive_reports_policy_decision(self):
        result, loops, explain = generate_adaptive(
            self.model,
            self.tokenizer,
            HARD_PROMPT,
            config=SolveConfig(max_new_tokens=4, max_loops=3),
        )
        self.assertEqual(result.num_loops, loops)
        self.assertEqual(explain["mode"], "adaptive")
        self.assertGreaterEqual(loops, 2)
        self.assertLessEqual(loops, 3)


class TestSolveTask(unittest.TestCase):
    def test_abstains_instead_of_guessing_on_ambiguous_prompt(self):
        calls = []

        def generator(prompt, loops):
            calls.append(prompt)
            return "something"

        result = solve_task(
            None, None, "make it better somehow", generate_fn=generator, config=SolveConfig()
        )
        self.assertTrue(result.abstained)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(calls, [])
        self.assertEqual(result.verification.failure_category, "ambiguous_request")
        self.assertEqual(result.measurement().value, 0.0)

    def test_one_verifier_driven_revision(self):
        answers = ["41", "42"]
        prompts = []

        def generator(prompt, loops):
            prompts.append(prompt)
            return answers[len(prompts) - 1]

        result = solve_task(
            None,
            None,
            "What is 6 * 7?",
            verifier=exact_match_verifier("42"),
            generate_fn=generator,
            config=SolveConfig(max_attempts=2),
        )
        self.assertEqual(result.attempts, 2)
        self.assertFalse(result.abstained)
        self.assertTrue(result.verified)
        self.assertIn("[REVISION", prompts[1])
        self.assertEqual(result.measurement().value, 1.0)
        self.assertEqual(result.trajectory[0]["correction"], "generalize")

    def test_self_correction_can_be_ablated(self):
        calls = []

        def generator(prompt, loops):
            calls.append(prompt)
            return "wrong"

        result = solve_task(
            None,
            None,
            "What is 6 * 7?",
            verifier=exact_match_verifier("42"),
            generate_fn=generator,
            config=SolveConfig(max_attempts=3),
            disabled={"self_correction"},
        )
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(result.verified)
        self.assertFalse(result.abstained)

    def test_unverified_answer_is_not_a_score(self):
        result = solve_task(
            None,
            None,
            "Explain what a KV cache is.",
            generate_fn=lambda prompt, loops: "a cache of keys and values",
            config=SolveConfig(constraint_check=False),
        )
        self.assertIsNone(result.verification)
        measurement = result.measurement()
        self.assertEqual(measurement.status, MeasurementStatus.NOT_RUN)
        self.assertIsNone(measurement.value)

    def test_measurement_identity_is_carried_through(self):
        result = solve_task(
            None,
            None,
            "What is 6 * 7?",
            verifier=exact_match_verifier("42"),
            generate_fn=lambda prompt, loops: "42",
            config=SolveConfig(identity=MeasurementIdentity.AGENT_TASK),
        )
        self.assertEqual(result.measurement().identity, MeasurementIdentity.AGENT_TASK)

    def test_decomposition_subgoals_and_ablation(self):
        on = solve_task(
            None,
            None,
            HARD_PROMPT,
            generate_fn=lambda prompt, loops: "done",
            config=SolveConfig(constraint_check=False),
        )
        self.assertTrue(on.subgoals)
        off = solve_task(
            None,
            None,
            HARD_PROMPT,
            generate_fn=lambda prompt, loops: "done",
            config=SolveConfig(constraint_check=False),
            disabled={"decomposition"},
        )
        self.assertEqual(off.subgoals, [])

    def test_end_to_end_with_real_model(self):
        tokenizer = BPETokenizer()
        model = _tiny_model(tokenizer, seed=8)
        result = solve_task(
            model,
            tokenizer,
            "Write a Python function add(a, b) that returns the sum of its arguments.",
            verifier=lambda text: __import__("maya250m.verification", fromlist=["ok"]).ok("always"),
            config=SolveConfig(max_new_tokens=4, adaptive_depth=True, max_attempts=1),
        )
        self.assertEqual(result.attempts, 1)
        self.assertGreaterEqual(result.loops_used, 1)
        self.assertLessEqual(result.loops_used, 4)
        self.assertIn("add(a, b)", result.prompt)
        self.assertTrue(result.verified)


if __name__ == "__main__":
    unittest.main(verbosity=2)
