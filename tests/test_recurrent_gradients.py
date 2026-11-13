"""Proof tests for the recurrent residual path (v3 report section 2).

The v3 audit warns that the loop injection may be dead: if ``core_loop_norm`` were
used as a bare multiplier and initialised to zero, the recurrent output and its
gradient would vanish. In this implementation ``rms_norm_fp32`` applies
``scale = 1 + weight`` when ``zero_centered`` is on, so a zero parameter is the
identity and the path is active. These tests prove that property directly, then
prove gradient flow and forward/prefill/decode parity for R = 1..4.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.config import get_config
from maya250m.model import MayaModel, rms_norm_fp32

SMALL_VOCAB = 512


def _tiny_model(loops: int = 2, seed: int = 0, dtype: torch.dtype = torch.float32) -> MayaModel:
    torch.manual_seed(seed)
    cfg = get_config("MAYA_TINY_TEST")
    cfg["recurrent_loops"] = int(loops)
    cfg["vocab"] = SMALL_VOCAB
    cfg["max_ctx"] = 128
    model = MayaModel(cfg, tokenizer_hash="0" * 64)
    return model.to(dtype) if dtype != torch.float32 else model


def _ids(seq_len: int = 12, batch: int = 1, vocab: int = SMALL_VOCAB) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    return torch.randint(0, vocab, (batch, seq_len), generator=generator)


class TestNormSemantics(unittest.TestCase):
    def test_zero_centered_weight_is_identity(self):
        torch.manual_seed(0)
        x = torch.randn(2, 5, 16)
        w = torch.zeros(16)
        rhs = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        self.assertTrue(torch.allclose(rms_norm_fp32(x, w, 1e-5, True), rhs, atol=1e-6))

    def test_scale_is_one_plus_weight(self):
        x = torch.randn(1, 3, 8)
        w = torch.randn(8)
        expected = rms_norm_fp32(x, torch.zeros(8), 1e-5, True) * (1.0 + w)
        self.assertTrue(torch.allclose(rms_norm_fp32(x, w, 1e-5, True), expected, atol=1e-6))

    def test_non_zero_centered_uses_bare_weight(self):
        x = torch.randn(1, 3, 8)
        w = torch.full((8,), 0.5)
        normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        self.assertTrue(
            torch.allclose(rms_norm_fp32(x, w, 1e-5, False), normalized * 0.5, atol=1e-6)
        )


class TestRecurrentPathIsAlive(unittest.TestCase):
    def test_zero_gate_still_produces_loop_output(self):
        """At init the gate is zero, but loop 1 must differ from the prelude output."""
        model = _tiny_model(loops=2).eval()
        self.assertLess(float(model.core_loop_gate.detach().abs().max()), 1e-9)
        ids = _ids(seq_len=8)
        with torch.no_grad():
            two, _ = model(ids, num_loops=2)
            one, _ = model(ids, num_loops=1)
        self.assertGreater(float((two - one).abs().max()), 1e-6)

    def test_gate_changes_the_result(self):
        model = _tiny_model(loops=2).eval()
        ids = _ids(seq_len=8)
        with torch.no_grad():
            base, _ = model(ids, num_loops=3)
            model.core_loop_gate.fill_(0.1)
            shifted, _ = model(ids, num_loops=3)
            model.core_loop_gate.fill_(-0.1)
            negative, _ = model(ids, num_loops=3)
        self.assertGreater(float((shifted - base).abs().max()), 1e-8)
        self.assertGreater(float((negative - shifted).abs().max()), 1e-8)

    def test_gradients_flow_through_gate_norm_and_core_blocks(self):
        model = _tiny_model(loops=2)
        model.train()
        ids = _ids(seq_len=10)
        logits, _ = model(ids, num_loops=2)
        loss = logits.float().pow(2).mean()
        loss.backward()

        gate_grad = model.core_loop_gate.grad
        norm_grad = model.core_loop_norm.grad
        self.assertIsNotNone(gate_grad)
        self.assertIsNotNone(norm_grad)
        self.assertGreater(float(gate_grad.abs().max()), 1e-12)
        self.assertGreater(float(norm_grad.abs().max()), 1e-10)
        self.assertTrue(torch.isfinite(gate_grad).all() and torch.isfinite(norm_grad).all())

        core_block = model.blocks[model.prelude_layers]
        grads = [p.grad for p in core_block.parameters() if p.grad is not None]
        self.assertTrue(grads, "no gradient reached the first core block")
        self.assertGreater(max(float(g.abs().max()) for g in grads), 1e-12)

    def test_loop_zero_perturbation_changes_later_loop_output(self):
        model = _tiny_model(loops=2).eval()
        ids = _ids(seq_len=8)
        with torch.no_grad():
            base, _ = model(ids, num_loops=2)
            # Perturb the prelude representation that feeds the recurrent core.
            model.blocks[0].n1.add_(0.5)
            perturbed, _ = model(ids, num_loops=2)
        self.assertGreater(float((perturbed - base).abs().max()), 1e-6)


class TestForwardPrefillDecodeParity(unittest.TestCase):
    """forward() == prefill + token-by-token decode() for R = 1..4, with numbers."""

    def _assert_parity(self, loops: int, mode: str = "train", batch: int = 1) -> None:
        model = _tiny_model(loops=loops, seed=11)
        model.train() if mode == "train" else model.eval()
        ids = _ids(seq_len=12, batch=batch)

        with torch.no_grad():
            full_logits, _ = model(ids, num_loops=loops)
            prefill_logits, cache, info = model.prefill(ids, num_loops=loops)
            self.assertEqual(int(info["num_loops"]), loops)
            self.assertEqual(int(cache["num_loops"]), loops)

            ref = full_logits[:, -1, :]
            max_abs = float((prefill_logits - ref).abs().max())
            denom = ref.abs().clamp(min=1e-6)
            max_rel = float(((prefill_logits - ref).abs() / denom).max())
            self.assertLess(max_abs, 1e-4, f"R={loops} prefill max_abs={max_abs:.3e}")
            self.assertLess(max_rel, 1e-3, f"R={loops} prefill max_rel={max_rel:.3e}")

            # Continue decoding a few tokens and compare with the full-sequence call.
            extra = _ids(seq_len=4, batch=batch, vocab=SMALL_VOCAB)
            step_cache = cache
            produced = []
            for step in range(extra.shape[1]):
                token = extra[:, step : step + 1]
                step_logits, step_cache = model.decode_step(token, step_cache)
                produced.append(step_logits)
            decoded = torch.stack(produced, dim=1)  # [B, steps, vocab]
            extended = torch.cat([ids, extra], dim=1)
            extended_logits, _ = model(extended, num_loops=loops)
            reference = extended_logits[:, ids.shape[1] :, :]
            delta = float((decoded - reference).abs().max())
            self.assertLess(delta, 1e-4, f"R={loops} decode max_abs={delta:.3e}")

    def test_parity_r1(self):
        self._assert_parity(1)

    def test_parity_r2(self):
        self._assert_parity(2)

    def test_parity_r3(self):
        self._assert_parity(3)

    def test_parity_r4(self):
        self._assert_parity(4)

    def test_parity_batch_two(self):
        self._assert_parity(2, batch=2)

    def test_parity_eval_mode(self):
        self._assert_parity(3, mode="eval")

    def test_document_boundary_parity(self):
        """Doc-boundary resets must behave identically in both paths."""
        model = _tiny_model(loops=2, seed=5).eval()
        ids = _ids(seq_len=10)
        doc_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
        with torch.no_grad():
            full_logits, _ = model(ids, doc_ids=doc_ids, num_loops=2)
            prefill_logits, _, _ = model.prefill(ids, doc_ids=doc_ids, num_loops=2)
        self.assertLess(float((prefill_logits - full_logits[:, -1, :]).abs().max()), 1e-4)


class TestMixedPrecision(unittest.TestCase):
    def test_bfloat16_parity_within_dtype_tolerance(self):
        torch.manual_seed(0)
        model = _tiny_model(loops=2, seed=7).to(torch.bfloat16).eval()
        ids = _ids(seq_len=10)
        with torch.no_grad():
            full_logits, _ = model(ids, num_loops=2)
            prefill_logits, _, _ = model.prefill(ids, num_loops=2)
            delta = float((prefill_logits.float() - full_logits[:, -1, :].float()).abs().max())
        # bf16 has ~3 decimal digits; the contract is tighter than 5e-2 logit units.
        self.assertLess(delta, 5e-2, f"bf16 parity delta {delta:.3e}")
        self.assertGreater(float(prefill_logits.float().abs().max()), 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
