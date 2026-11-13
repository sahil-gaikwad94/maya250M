"""v3 P0.3: cache identity, snapshot/replay rewind, and exact memory accounting."""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m.cache import (
    CACHE_SCHEMA_VERSION,
    CacheIdentityError,
    CacheRewindError,
    CacheSnapshotStore,
    build_identity,
    cache_memory_ledger,
    cache_memory_table,
    rewind_cache,
    stamp_bundle,
    validate_bundle,
)
from maya250m.config import get_config
from maya250m.model import MayaModel, create_empty_hybrid_cache_v3

SMALL_VOCAB = 512


def _tiny_model(loops: int = 2, seed: int = 0, checkpoint: str = "ckpt-A") -> MayaModel:
    torch.manual_seed(seed)
    cfg = get_config("MAYA_TINY_TEST")
    cfg["vocab"] = SMALL_VOCAB
    cfg["max_ctx"] = 128
    cfg["recurrent_loops"] = int(loops)
    model = MayaModel(cfg, tokenizer_hash="tok-hash-1")
    model.checkpoint_artifact_id = checkpoint
    return model


def _ids(seq_len: int = 12, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, SMALL_VOCAB, (1, seq_len), generator=generator)


class TestCacheIdentity(unittest.TestCase):
    def test_prefill_stamps_schema3_metadata(self):
        model = _tiny_model()
        ids = _ids(8)
        with torch.no_grad():
            _logits, bundle, _info = model.prefill(ids, num_loops=2, tokenizer_hash="tok-H")
        self.assertEqual(bundle["schema_version"], CACHE_SCHEMA_VERSION)
        self.assertEqual(bundle["num_loops"], 2)
        self.assertEqual(bundle["batch_size"], 1)
        self.assertEqual(bundle["position"], 8)
        self.assertEqual(bundle["dtype"], "torch.float32")
        self.assertEqual(bundle["device_type"], "cpu")
        self.assertEqual(bundle["tokenizer_hash"], "tok-H")
        self.assertEqual(bundle["checkpoint_sha256"], "ckpt-A")
        self.assertEqual(bundle["config_hash"], model.config_hash())
        self.assertIn("doc_state", bundle)
        self.assertEqual(validate_bundle(bundle), 3)

    def test_identity_mismatches_are_rejected(self):
        model = _tiny_model()
        ids = _ids(6)
        with torch.no_grad():
            _logits, bundle, _info = model.prefill(ids, num_loops=2, tokenizer_hash="tok-H")

        good = build_identity(model, batch_size=1, num_loops=2, dtype=torch.float32)
        good = type(good)(**{**good.to_dict(), "tokenizer_hash": "tok-H"})
        validate_bundle(bundle, good)

        for field, value in (
            ("config_hash", "0" * 64),
            ("batch_size", 2),
            ("num_loops", 3),
            ("dtype", "torch.bfloat16"),
            ("device_type", "cuda"),
            ("tokenizer_hash", "other-tokenizer"),
            ("checkpoint_sha256", "ckpt-B"),
        ):
            payload = good.to_dict()
            payload[field] = value
            with self.assertRaises(CacheIdentityError, msg=f"{field} not rejected"):
                validate_bundle(bundle, type(good)(**payload))

    def test_checkpoint_change_invalidates_cache(self):
        model = _tiny_model()
        ids = _ids(5)
        with torch.no_grad():
            _logits, bundle, _info = model.prefill(ids, num_loops=2)
        model.checkpoint_artifact_id = "ckpt-B"
        fresh_identity = build_identity(model, batch_size=1, num_loops=2, dtype=torch.float32)
        with self.assertRaises(CacheIdentityError):
            validate_bundle(bundle, fresh_identity)

    def test_decode_rejects_wrong_batch_size(self):
        model = _tiny_model().eval()
        ids = _ids(5)
        with torch.no_grad():
            _logits, bundle, _info = model.prefill(ids, num_loops=2)
            with self.assertRaises(CacheIdentityError):
                model.decode_step(torch.tensor([[1], [2]]), bundle)

    def test_empty_v3_cache_is_valid_and_stamped(self):
        model = _tiny_model()
        cache = create_empty_hybrid_cache_v3(model, num_loops=2, batch_size=1)
        self.assertEqual(cache["schema_version"], CACHE_SCHEMA_VERSION)
        self.assertEqual(cache["position"], 0)
        self.assertEqual(len(cache["core_loops"]), 2)
        validate_bundle(cache)

    def test_stamp_preserves_tensors_by_reference(self):
        model = _tiny_model()
        ids = _ids(4)
        with torch.no_grad():
            _logits, bundle, _info = model.prefill(ids, num_loops=2)
        identity = build_identity(model, batch_size=1, num_loops=2, dtype=torch.float32)
        stamped = stamp_bundle(bundle, identity)
        self.assertIs(stamped["prelude"], bundle["prelude"])


class TestRewind(unittest.TestCase):
    def _setup(self):
        model = _tiny_model(loops=2, seed=3).eval()
        ids = _ids(12, seed=11)
        with torch.no_grad():
            _logits, full, _info = model.prefill(ids, num_loops=2)
        return model, ids, full

    def test_rewind_to_final_position_is_identity(self):
        model, _ids_tensor, full = self._setup()
        store = CacheSnapshotStore(interval=4)
        store.snapshot(full, force=True)
        restored = rewind_cache(full, full["position"], store, model)
        self.assertEqual(restored["position"], full["position"])
        with torch.no_grad():
            a, _ = model.decode_step(torch.tensor([[5]]), full)
            b, _ = model.decode_step(torch.tensor([[5]]), restored)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_rewind_to_middle_matches_prefix_cache(self):
        model, ids, full = self._setup()
        store = CacheSnapshotStore(interval=4)
        with torch.no_grad():
            _logits, prefix, _info = model.prefill(ids[:, :6], num_loops=2)
        store.snapshot(prefix, force=True)

        restored = rewind_cache(full, 6, store, model, replay_ids=ids[:, 6:])
        self.assertEqual(restored["position"], 6)
        nxt = torch.tensor([[9]])
        with torch.no_grad():
            expected, _ = model.decode_step(nxt, prefix)
            actual, _ = model.decode_step(nxt, restored)
        self.assertLess(float((expected - actual).abs().max()), 1e-5)

    def test_rewind_to_zero_and_full_replay(self):
        model, ids, full = self._setup()
        store = CacheSnapshotStore(interval=64)
        empty = create_empty_hybrid_cache_v3(model, num_loops=2, batch_size=1)
        store.snapshot(empty, position=0, force=True)

        restored = rewind_cache(full, 0, store, model, replay_ids=ids)
        self.assertEqual(restored["position"], 0)
        replayed = rewind_cache(full, ids.shape[1], store, model, replay_ids=ids)
        nxt = torch.tensor([[3]])
        with torch.no_grad():
            expected, _ = model.decode_step(nxt, full)
            actual, _ = model.decode_step(nxt, replayed)
        self.assertLess(float((expected - actual).abs().max()), 1e-5)

    def test_rewind_across_document_boundary(self):
        model = _tiny_model(loops=2, seed=4).eval()
        ids = _ids(10, seed=5)
        doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
        with torch.no_grad():
            _logits, full, _info = model.prefill(ids, doc_ids=doc_ids, num_loops=2)
            _logits, midpoint, _info = model.prefill(
                ids[:, :5], doc_ids=doc_ids[:, :5], num_loops=2
            )
        store = CacheSnapshotStore(interval=5)
        store.snapshot(midpoint, force=True)
        restored = rewind_cache(full, 5, store, model, replay_ids=ids[:, 5:])
        nxt = torch.tensor([[2]])
        with torch.no_grad():
            expected, _ = model.decode_step(nxt, midpoint)
            actual, _ = model.decode_step(nxt, restored)
        self.assertLess(float((expected - actual).abs().max()), 1e-5)

    def test_rewind_without_snapshot_or_replay_is_refused(self):
        model, ids, full = self._setup()
        empty_store = CacheSnapshotStore(interval=4)
        with self.assertRaises(CacheRewindError):
            rewind_cache(full, 4, empty_store, model, replay_ids=ids)

        store = CacheSnapshotStore(interval=4)
        with torch.no_grad():
            _logits, prefix, _info = model.prefill(ids[:, :4], num_loops=2)
        store.snapshot(prefix, force=True)
        with self.assertRaises(CacheRewindError):
            rewind_cache(full, 8, store, model)  # needs replay
        with self.assertRaises(CacheRewindError):
            rewind_cache(full, 8, store, model, replay_ids=ids[:, 9:])  # too few tokens
        with self.assertRaises(CacheRewindError):
            rewind_cache(full, 20, store, model)  # forward is not a rewind

    def test_all_core_loop_passes_are_restored(self):
        model, ids, full = self._setup()
        store = CacheSnapshotStore(interval=4)
        with torch.no_grad():
            _logits, prefix, _info = model.prefill(ids[:, :4], num_loops=2)
        store.snapshot(prefix, force=True)
        restored = rewind_cache(full, 4, store, model, replay_ids=ids[:, 4:])
        self.assertEqual(len(restored["core_loops"]), 2)
        for pass_index, (expected_pass, actual_pass) in enumerate(
            zip(prefix["core_loops"], restored["core_loops"])
        ):
            for expected_layer, actual_layer in zip(expected_pass, actual_pass):
                for key in ("k", "v", "conv", "state"):
                    if key in expected_layer:
                        self.assertTrue(
                            torch.equal(expected_layer[key], actual_layer[key]),
                            f"loop pass {pass_index} key {key} diverged",
                        )


class TestMemoryLedger(unittest.TestCase):
    def test_ledger_scales_with_loops_and_context(self):
        model = _tiny_model(loops=2)
        rows = cache_memory_table(model, loop_counts=(1, 2, 3, 4), ctx=64)
        self.assertEqual([row["loops"] for row in rows], [1, 2, 3, 4])
        for row in rows:
            self.assertGreater(row["kv_cache_bytes"], 0)
            self.assertGreater(row["recurrent_state_bytes"], 0)
            self.assertEqual(
                row["combined_bytes"], row["kv_cache_bytes"] + row["recurrent_state_bytes"]
            )
        kv = [row["kv_cache_bytes"] for row in rows]
        self.assertEqual(kv, sorted(kv))
        self.assertGreater(kv[-1], kv[0])

        doubled = cache_memory_ledger(model, loops=2, ctx=64)
        single = cache_memory_ledger(model, loops=2, ctx=32)
        self.assertEqual(doubled["kv_cache_bytes"], 2 * single["kv_cache_bytes"])
        self.assertEqual(doubled["recurrent_state_bytes"], single["recurrent_state_bytes"])

    def test_ledger_matches_manual_tensor_sum(self):
        model = _tiny_model(loops=3).eval()
        with torch.no_grad():
            _logits, bundle, _info = model.prefill(torch.zeros((1, 1), dtype=torch.long), num_loops=3)
        manual_state = 0
        # Core-loop state exists once per pass, so it must be counted R times.
        for section, multiplier in (
            (bundle["prelude"], 1),
            (bundle["coda"], 1),
            (bundle["core_loops"][0], 3),
        ):
            for entry in section:
                if not isinstance(entry, dict):
                    continue
                for key, value in entry.items():
                    if key in ("k", "v"):
                        continue
                    if isinstance(value, torch.Tensor):
                        manual_state += multiplier * value.numel() * value.element_size()
        ledger = cache_memory_ledger(model, loops=3, ctx=128)
        self.assertEqual(ledger["recurrent_state_bytes"], manual_state)
        self.assertIn("core_gdn_layers", ledger["components"])
        self.assertGreater(ledger["components"]["core_full_attention_layers"], 0)

    def test_snapshot_store_is_bounded(self):
        model = _tiny_model()
        ids = _ids(10)
        store = CacheSnapshotStore(interval=2, max_snapshots=3)
        for k in range(2, 11, 2):
            with torch.no_grad():
                _logits, bundle, _info = model.prefill(ids[:, :k], num_loops=2)
            store.snapshot(bundle)
        self.assertLessEqual(len(store.snapshots), 3)
        self.assertGreater(store.memory_bytes(), 0)
        self.assertIsNotNone(store.floor(9))
        store.clear()
        self.assertEqual(store.memory_bytes(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
