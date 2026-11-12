"""Phase 3 tests: SFT, distillation, DPO and GRPO must really update weights (BUG-02)."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.config import get_config
from maya250m.model import MayaModel
from maya250m.tokenizer import BPETokenizer
from maya250m.training import (
    PreferencePair,
    RolloutGroup,
    StageConfig,
    StagePlan,
    StageResult,
    TrainingDidNotUpdate,
    count_changed_parameters,
    dpo_loss,
    frozen_reference,
    grpo_objective,
    run_distillation_stage,
    run_dpo_stage,
    run_grpo_stage,
    run_sft_stage,
    run_stage_plan,
    snapshot_parameters,
    tokenize_prompt_completion,
    tokenize_sft_text,
)

PROMPT = "# Write a Python function `add` that returns a + b.\n"
CHOSEN = "def add(a, b):\n    return a + b\n"
REJECTED = "def add(a, b):\n    return a - b\n"


def _tiny_model(tokenizer: BPETokenizer, seed: int = 0) -> MayaModel:
    torch.manual_seed(seed)
    cfg = get_config("MAYA_TINY_TEST")
    cfg["vocab"] = tokenizer.vocab_size
    cfg["max_ctx"] = 64
    cfg["recurrent_loops"] = 1
    return MayaModel(cfg, tokenizer_hash=tokenizer.tokenizer_hash())


class TestTokenization(unittest.TestCase):
    def test_prompt_is_masked_and_completion_eos_scored(self):
        tok = BPETokenizer()
        example = tokenize_prompt_completion(tok, PROMPT, CHOSEN, max_len=64)
        prompt_tokens = example.metadata["prompt_tokens"]
        self.assertTrue(all(label == -100 for label in example.labels[:prompt_tokens]))
        self.assertTrue(all(label != -100 for label in example.labels[prompt_tokens:]))
        self.assertEqual(example.supervised_tokens, example.metadata["completion_tokens"])
        self.assertEqual(len(example.input_ids), len(example.labels))

    def test_long_prompt_is_truncated_before_completion(self):
        tok = BPETokenizer()
        before = tokenize_prompt_completion(tok, PROMPT * 20, CHOSEN, max_len=40)
        after = tokenize_prompt_completion(tok, PROMPT * 20, CHOSEN, max_len=40)
        self.assertLessEqual(len(before), 40)
        self.assertEqual(before.supervised_tokens, before.metadata["completion_tokens"])
        self.assertEqual(before.input_ids, after.input_ids)


class TestLossMath(unittest.TestCase):
    def test_dpo_is_ln2_at_reference(self):
        policy_c = torch.tensor([-1.0], requires_grad=True)
        policy_r = torch.tensor([-2.0], requires_grad=True)
        loss, metrics = dpo_loss(policy_c, policy_r, torch.tensor([-1.0]), torch.tensor([-2.0]))
        self.assertAlmostEqual(float(loss.detach()), math.log(2.0), places=6)
        self.assertAlmostEqual(metrics["ref_margin"], 1.0, places=6)
        loss.backward()
        self.assertIsNotNone(policy_c.grad)

    def test_dpo_length_normalisation(self):
        loss, _ = dpo_loss(
            torch.tensor([-10.0]),
            torch.tensor([-10.0]),
            torch.tensor([-10.0]),
            torch.tensor([-10.0]),
            chosen_counts=torch.tensor([10]),
            rejected_counts=torch.tensor([10]),
        )
        self.assertAlmostEqual(float(loss), math.log(2.0), places=6)

    def test_grpo_clipping_bounds_positive_advantage(self):
        # ratio = 100, advantage = +1 -> clipped surrogate is (1 + eps).
        policy_lp = torch.full((1, 4), math.log(100.0), requires_grad=True)
        old_lp = torch.zeros(1, 4)
        ref_lp = torch.zeros(1, 4)
        loss, metrics = grpo_objective(
            policy_lp,
            old_lp,
            ref_lp,
            torch.tensor([1.0]),
            torch.ones(1, 4, dtype=torch.bool),
            clip_eps=0.2,
            beta=0.0,
        )
        self.assertAlmostEqual(float(loss.detach()), -1.2, places=5)
        self.assertGreater(metrics["clip_fraction"], 0.0)
        loss.backward()
        self.assertIsNotNone(policy_lp.grad)

    def test_grpo_zero_advantage_group_contributes_nothing(self):
        policy_lp = torch.zeros(2, 4, requires_grad=True)
        loss, _ = grpo_objective(
            policy_lp,
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            torch.zeros(2),
            torch.ones(2, 4, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            beta=0.0,
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertEqual(float(policy_lp.grad.abs().sum()), 0.0)


class TrainingFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BPETokenizer()
        torch.manual_seed(0)

    def _model(self, seed: int = 0) -> MayaModel:
        return _tiny_model(self.tokenizer, seed=seed)

    def _sft_examples(self):
        return [
            tokenize_sft_text(self.tokenizer, text, max_len=64)
            for text in (
                "def add(a, b):\n    return a + b\n",
                "def mul(a, b):\n    return a * b\n",
                "x = [i * 2 for i in range(3)]\n",
            )
        ]


class TestSFTStage(TrainingFixture):
    def test_sft_updates_parameters(self):
        model = self._model()
        before = snapshot_parameters(model)
        cfg = StageConfig(
            name="sft", lr=5e-3, muon_lr=1e-2, steps=2, batch_size=2, max_len=64,
            track_changes=True,
        )
        result = run_sft_stage(model, self._sft_examples(), cfg)
        result.verify_updated()
        self.assertEqual(result.steps, 2)
        self.assertGreater(result.changed_parameters or 0, 0)
        self.assertEqual(count_changed_parameters(model, before) > 0, True)
        self.assertTrue(all(math.isfinite(loss) for loss in result.losses))
        self.assertGreater(result.grad_norms[0], 0.0)

    def test_sft_rejects_empty_supervision(self):
        model = self._model()
        cfg = StageConfig(name="sft", steps=1, lr=1e-3)
        with self.assertRaises(ValueError):
            run_sft_stage(model, [], cfg)


class TestDistillationStage(TrainingFixture):
    @staticmethod
    def _kl(student, teacher, example) -> float:
        from maya250m.training import distillation_loss

        ids, labels = example.to_tensors()
        with torch.no_grad():
            student_logits, _ = student(ids, labels=None)
            teacher_logits, _ = teacher(ids, labels=None)
        _, metrics = distillation_loss(
            student_logits, teacher_logits, labels, ce_weight=0.0
        )
        return metrics["distill_kl"]

    def test_distillation_moves_student_toward_teacher(self):
        model = self._model(seed=3)
        teacher = frozen_reference(model)
        with torch.no_grad():
            # Perturb the (tied) embedding/unembedding so the KL starts non-zero.
            teacher.embed.weight.add_(0.2 * torch.randn_like(teacher.embed.weight))

        examples = self._sft_examples()
        kl_before = self._kl(model, teacher, examples[0])
        self.assertGreater(kl_before, 0.01)

        cfg = StageConfig(
            name="distill",
            lr=2e-3,
            muon_lr=5e-3,
            steps=20,
            batch_size=1,
            temperature=1.0,
            ce_weight=0.0,
            max_len=64,
            track_changes=True,
        )
        result = run_distillation_stage(model, teacher, [examples[0]], cfg)
        result.verify_updated()
        self.assertGreater(result.changed_parameters or 0, 0)
        self.assertIn("distill_kl", result.metrics)
        self.assertGreater(result.metrics["distill_kl"], 0.0)

        kl_after = self._kl(model, teacher, examples[0])
        self.assertLess(kl_after, 0.9 * kl_before)
        # Teacher must stay frozen.
        teacher_before = teacher.embed.weight.detach().clone()
        self.assertTrue(torch.equal(teacher_before, teacher.embed.weight.detach()))

    def test_teacher_must_differ(self):
        model = self._model()
        with self.assertRaises(ValueError):
            run_distillation_stage(model, model, self._sft_examples(), StageConfig(steps=1))


class TestDPOStage(TrainingFixture):
    def test_dpo_stage_optimises_margin(self):
        model = self._model(seed=5)
        reference = frozen_reference(model)
        before = snapshot_parameters(model)
        pairs = [
            PreferencePair(prompt=PROMPT, chosen=CHOSEN, rejected=REJECTED),
            PreferencePair(prompt=PROMPT + "# keep it simple\n", chosen=CHOSEN, rejected="def add(a, b): return 0"),
        ]
        cfg = StageConfig(
            name="dpo", lr=5e-3, muon_lr=1e-2, steps=4, batch_size=1, beta=0.1, max_len=64,
            track_changes=True,
        )
        result = run_dpo_stage(model, reference, pairs, self.tokenizer, cfg)
        result.verify_updated()
        self.assertGreater(result.changed_parameters or 0, 0)
        self.assertEqual(count_changed_parameters(model, before) > 0, True)
        self.assertIn("reward_accuracy", result.metrics)
        self.assertLess(result.losses[-1], result.losses[0])
        self.assertGreater(result.grad_norms[0], 0.0)

    def test_dpo_requires_frozen_reference(self):
        model = self._model()
        with self.assertRaises(ValueError):
            run_dpo_stage(
                model,
                model,
                [PreferencePair(prompt=PROMPT, chosen=CHOSEN, rejected=REJECTED)],
                self.tokenizer,
                StageConfig(steps=1),
            )

    def test_dpo_rejects_empty_pairs(self):
        model = self._model()
        with self.assertRaises(ValueError):
            run_dpo_stage(model, frozen_reference(model), [], self.tokenizer, StageConfig(steps=1))


class TestGRPOStage(TrainingFixture):
    def _groups(self):
        return [
            RolloutGroup(
                prompt=PROMPT,
                completions=[CHOSEN, REJECTED, "def add(a, b):\n    return 0\n"],
                rewards=[1.0, 0.2, 0.5],
            ),
            # Zero-variance group: must be skipped, never divided by 1e-6.
            RolloutGroup(
                prompt=PROMPT + "# extra\n",
                completions=[CHOSEN, CHOSEN + "\n"],
                rewards=[0.3, 0.3],
            ),
        ]

    def test_grpo_stage_updates_policy(self):
        model = self._model(seed=7)
        reference = frozen_reference(model)
        before = snapshot_parameters(model)
        cfg = StageConfig(
            name="grpo", lr=5e-3, muon_lr=1e-2, steps=2, batch_size=1, beta=0.01, max_len=64,
            track_changes=True,
        )
        result = run_grpo_stage(model, reference, self._groups(), self.tokenizer, cfg)
        result.verify_updated()
        self.assertEqual(result.steps, 2)
        self.assertGreater(result.changed_parameters or 0, 0)
        self.assertEqual(count_changed_parameters(model, before) > 0, True)
        self.assertTrue(all(math.isfinite(loss) for loss in result.losses))
        self.assertEqual(result.metrics.get("skipped_zero_variance_groups"), 1.0)

    def test_all_degenerate_groups_report_zero_steps(self):
        model = self._model(seed=9)
        reference = frozen_reference(model)
        before = snapshot_parameters(model)
        groups = [
            RolloutGroup(prompt=PROMPT, completions=[CHOSEN, REJECTED], rewards=[1.0, 1.0])
        ]
        result = run_grpo_stage(model, reference, groups, self.tokenizer, StageConfig(steps=2, max_len=64))
        self.assertEqual(result.steps, 0)
        self.assertEqual(count_changed_parameters(model, before), 0)


class TestStagePlan(TrainingFixture):
    def test_plan_records_and_fails_closed(self):
        model = self._model(seed=11)
        plan = StagePlan().add(
            "sft",
            examples=self._sft_examples(),
            cfg=StageConfig(name="sft", lr=5e-3, muon_lr=1e-2, steps=1, max_len=64),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stage_report.json")
            results = run_stage_plan(model, plan, record_path=path)
            self.assertEqual(len(results), 1)
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertTrue(payload[0]["parameters_changed"])

    def test_verify_updated_detects_no_change(self):
        result = StageResult(stage="dpo", steps=3, fingerprint_before="abc", fingerprint_after="abc")
        with self.assertRaises(TrainingDidNotUpdate):
            result.verify_updated()


if __name__ == "__main__":
    unittest.main(verbosity=2)
