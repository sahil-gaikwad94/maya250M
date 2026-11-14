"""Phase 0A regression suite: recurrent inference parity and core-loop gradients.

Verifies the exit gate for Milestone 1:
    ``forward()``, ``prefill()`` and token-by-token ``decode_step()`` execute the
    exact same mathematical function for ``num_loops = 1, 2, 3`` and
    ``batch_size = 1, 2`` (max logit difference < 1e-4).

Also covers:
* ``BUG-06`` — the loop-injection gate must not zero out Loop 0 at init, so
  ``core_loop_norm`` / ``core_loop_gate`` must receive non-zero gradients on the
  very first backward pass.
* ``BUG-07`` — per-horizon MTP losses must be exposed individually and the pooled
  MTP loss must equal the mean of the active horizons.
* ``BUG-07`` — microbatch gradient accumulation must be numerically identical to a
  single concatenated batch even when MTP horizons have unequal valid counts.
"""

from __future__ import annotations

import os
import sys
import unittest

try:  # PyTorch is optional on CPU-only shards; tensor tests are skipped without it.
    import torch
except ImportError:  # pragma: no cover - exercised on torch-free nodes
    torch = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if torch is not None:
    from maya250m import (
    BPETokenizer,
    MayaModel,
    StreamPackedDataset,
    get_config,
    train_maya,
    )
    from maya250m.model import (
        clone_hybrid_cache,
        create_empty_hybrid_cache_v2,
        truncate_hybrid_cache_at_token,
    )

SMALL_VOCAB = 512


def _tiny_model(loops: int = 1, seed: int = 0) -> MayaModel:
    """Deterministic MAYA_TINY_TEST model with a compact vocabulary."""
    torch.manual_seed(seed)
    cfg = get_config("MAYA_TINY_TEST")
    cfg["recurrent_loops"] = int(loops)
    cfg["vocab"] = SMALL_VOCAB
    cfg["max_ctx"] = 128
    return MayaModel(cfg, tokenizer_hash="0" * 64)


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestRecurrentParity(unittest.TestCase):
    def _assert_parity(self, loops: int, batch_size: int) -> None:
        model = _tiny_model(loops=loops, seed=11)
        model.eval()

        torch.manual_seed(1234)
        n, prefix = 12, 8
        ids = torch.randint(1, SMALL_VOCAB, (batch_size, n))

        with torch.no_grad():
            full_logits, _ = model(ids)
            head_logits, bundle, info = model.prefill(ids[:, :prefix])

            self.assertEqual(bundle["schema_version"], 3)
            self.assertEqual(bundle["num_loops"], loops)
            self.assertEqual(info["effective_len"], prefix)
            self.assertEqual(bundle["position"], prefix)
            self.assertEqual(len(bundle["core_loops"]), loops)

            diff = (head_logits - full_logits[:, prefix - 1, :]).abs().max().item()
            self.assertLess(diff, 1e-4, f"prefill drift {diff:.3e} (loops={loops})")

            cache = bundle
            for t in range(prefix, n):
                step_logits, cache = model.decode_step(ids[:, t : t + 1], cache)
                step_diff = (step_logits - full_logits[:, t, :]).abs().max().item()
                self.assertLess(
                    step_diff,
                    1e-4,
                    f"decode_step drift {step_diff:.3e} at t={t} (loops={loops})",
                )
            self.assertEqual(cache["position"], n)

    def test_parity_loops_1_batch_1(self):
        self._assert_parity(loops=1, batch_size=1)

    def test_parity_loops_1_batch_2(self):
        self._assert_parity(loops=1, batch_size=2)

    def test_parity_loops_2_batch_1(self):
        self._assert_parity(loops=2, batch_size=1)

    def test_parity_loops_2_batch_2(self):
        self._assert_parity(loops=2, batch_size=2)

    def test_parity_loops_3_batch_2(self):
        self._assert_parity(loops=3, batch_size=2)

    def test_core_loop_gate_receives_gradient_at_init(self):
        """BUG-06: Loop 0's computation must not be multiplied by zero at init."""
        model = _tiny_model(loops=2, seed=3)
        torch.manual_seed(5)
        ids = torch.randint(1, SMALL_VOCAB, (2, 10))
        labels = ids.clone()

        _logits, loss, _comp = model(ids, labels=labels, return_components=True)
        loss.backward()

        self.assertIsNotNone(model.core_loop_norm.grad)
        self.assertGreater(
            float(model.core_loop_norm.grad.abs().sum().item()),
            0.0,
            "core_loop_norm gradient is identically zero (BUG-06 regression)",
        )
        self.assertIsNotNone(model.core_loop_gate.grad)

    def test_mtp_per_horizon_losses_exposed(self):
        """BUG-07: every MTP horizon reports its own CE and the pooled loss is their mean."""
        model = _tiny_model(loops=1, seed=7)
        model.eval()
        torch.manual_seed(9)
        ids = torch.randint(1, SMALL_VOCAB, (2, 12))
        labels = ids.clone()

        with torch.no_grad():
            _, _, comp = model(
                ids, labels=labels, mtp_weight=0.2, return_components=True
            )

        losses = comp["mtp_horizon_losses"]
        counts = comp["mtp_horizon_counts"]
        self.assertEqual(len(losses), model.mtp_depth)
        self.assertEqual(len(counts), model.mtp_depth)

        active = [l for l in losses if l is not None]
        self.assertTrue(active, "expected at least one active MTP horizon")
        expected = torch.stack(active).mean().item()
        self.assertAlmostEqual(comp["mtp_loss"].item(), expected, places=6)

    def test_grad_accum_matches_concatenated_batch_with_mtp(self):
        """BUG-07: per-horizon normalization makes grad accumulation exact."""
        texts = [
            f"def f_{i}(x: int) -> int:\n    return x + {i}\n" + "pass\n" * (i % 4)
            for i in range(24)
        ]
        tok = BPETokenizer.train(texts, vocab_size=384)
        dataset = StreamPackedDataset(tok, texts, seq_len=32, seed=42)

        def run(batch_size: int, grad_accum: int) -> MayaModel:
            torch.manual_seed(17)
            cfg = get_config("MAYA_TINY_TEST")
            cfg["vocab"] = tok.vocab_size
            model = MayaModel(cfg, tokenizer_hash=tok.tokenizer_hash())
            train_maya(
                model,
                dataset,
                batch_size=batch_size,
                grad_accum_steps=grad_accum,
                max_steps=1,
                mtp_weight=0.2,
                intermediate_supervision_weight=0.0,
                seed=42,
            )
            return model

        concat = run(batch_size=2, grad_accum=1)
        accumulated = run(batch_size=1, grad_accum=2)

        worst = 0.0
        for p_concat, p_accum in zip(concat.parameters(), accumulated.parameters()):
            diff = (p_concat.detach() - p_accum.detach()).abs().max().item()
            worst = max(worst, diff)
        self.assertLess(
            worst, 1e-5, f"grad-accum drift {worst:.3e} exceeds exactness budget"
        )

    def test_v2_cache_helpers_support_both_schemas(self):
        empty = create_empty_hybrid_cache_v2(2, 1, 2, 1, 1, "hash")
        self.assertEqual(len(empty["core_loops"]), 2)
        self.assertEqual(len(empty["core_loops"][0]), 2)

        kv = {"k": torch.ones(1, 4, 10, 8), "v": torch.ones(1, 4, 10, 8)}
        empty["position"] = 10
        empty["prelude"][0] = {kk: vv.clone() for kk, vv in kv.items()}
        empty["core_loops"][1][0] = {kk: vv.clone() for kk, vv in kv.items()}

        cloned = clone_hybrid_cache(empty)
        truncated = truncate_hybrid_cache_at_token(cloned, 4)

        self.assertEqual(truncated["prelude"][0]["k"].shape[2], 4)
        self.assertEqual(truncated["core_loops"][1][0]["k"].shape[2], 4)
        self.assertEqual(truncated["position"], 4)
        # the original cache must be untouched (deep copy)
        self.assertEqual(empty["prelude"][0]["k"].shape[2], 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
