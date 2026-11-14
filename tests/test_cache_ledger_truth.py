"""Reconcile the published ledger with the model's real cache and the README.

The README once advertised half the real 250M KV cache and a "zero extra KV
cache" loop claim. Numbers like that are claims, so they are pinned here against
three independent things: config arithmetic, a measured prefill through
``maya250m.cache.cache_memory_ledger``, and the README table itself.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:  # PyTorch is optional on CPU-only shards; tensor tests are skipped without it.
    import torch
except ImportError:  # pragma: no cover - exercised on torch-free nodes
    torch = None

from maya250m import get_config

if torch is not None:
    from maya250m import MayaModel
    from maya250m.cache import cache_memory_ledger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

with open(os.path.join(PROJECT_ROOT, "metrics_maya250m.json"), encoding="utf-8") as _f:
    LEDGER = json.load(_f)

with open(os.path.join(PROJECT_ROOT, "README.md"), encoding="utf-8") as _f:
    README = _f.read()


def _first_int(cell: str) -> int:
    match = re.search(r"\d[\d,]*", cell)
    if match is None:
        raise AssertionError(f"no integer in table cell {cell!r}")
    return int(match.group(0).replace(",", ""))


def _schedule(cfg: dict) -> list:
    return list(cfg["hybrid_schedule"] or ["full"] * int(cfg["layers"]))


class TestLedgerArithmetic(unittest.TestCase):
    def test_cache_columns_follow_from_the_config(self):
        for preset, row in LEDGER.items():
            cfg = get_config(preset)
            schedule = _schedule(cfg)
            head_dim = int(cfg["d"]) // int(cfg["n_heads"])
            per_full_layer = 2 * int(cfg["n_kv_heads"]) * head_dim * int(cfg["max_ctx"]) * 2
            n_full = sum(1 for layer in schedule if layer == "full")
            n_gdn = sum(1 for layer in schedule if layer == "gdn")

            self.assertEqual(row["kv_cache_bytes_bf16"], n_full * per_full_layer, preset)
            self.assertEqual(
                row["dense_counterfactual_kv_bytes_bf16"],
                int(cfg["layers"]) * per_full_layer,
                preset,
            )

            gdn_head_dim = int(cfg["d"]) // int(cfg["gdn_heads"])
            matrix = int(cfg["gdn_heads"]) * gdn_head_dim**2 * 4
            conv = 3 * int(cfg["d"]) * (int(cfg["conv_kernel"]) - 1) * 4
            self.assertEqual(row["gdn_state_bytes_fp32"], n_gdn * matrix, preset)
            self.assertEqual(row["gdn_conv_state_bytes_fp32"], n_gdn * conv, preset)
            self.assertEqual(
                row["recurrent_state_bytes_fp32"], n_gdn * (matrix + conv), preset
            )

    def test_auxiliary_memory_head_is_marked_untrained(self):
        for preset, row in LEDGER.items():
            self.assertFalse(row["auxiliary_memory_head_attached"], preset)
            self.assertEqual(row["auxiliary_memory_head_status"], "auxiliary_untrained")
            self.assertEqual(
                row["trainable_total_with_auxiliary"],
                row["total_parameters"] + row["auxiliary_memory_head_parameters"],
                preset,
            )

    def test_loop_scaling_only_touches_the_core(self):
        for preset, row in LEDGER.items():
            cfg = get_config(preset)
            schedule = _schedule(cfg)
            head_dim = int(cfg["d"]) // int(cfg["n_heads"])
            per_full_layer = 2 * int(cfg["n_kv_heads"]) * head_dim * int(cfg["max_ctx"]) * 2
            prelude = int(cfg["prelude_layers"])
            core = schedule[prelude : prelude + int(cfg["core_layers"])]
            core_kv = sum(1 for layer in core if layer == "full") * per_full_layer

            for loops in (1, 2, 3, 4):
                expected = row["kv_cache_bytes_bf16"] + (loops - 1) * core_kv
                self.assertEqual(
                    row["kv_cache_bytes_bf16_by_loops"][str(loops)], expected, preset
                )
            self.assertEqual(row["kv_cache_bytes_bf16_by_loops"]["1"], row["kv_cache_bytes_bf16"])


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestMeasuredCache(unittest.TestCase):
    """A measured prefill must reproduce the ledger, loop by loop."""

    def test_measured_cache_matches_the_ledger(self):
        for preset in ("MAYA_TINY_TEST", "MAYA_50M_PILOT"):
            row = LEDGER[preset]
            cfg = get_config(preset)
            model = MayaModel(cfg, tokenizer_hash="0" * 64)
            for loops in (1, 2, 3):
                measured = cache_memory_ledger(
                    model, ctx=row["max_ctx"], loops=loops, kv_dtype=torch.bfloat16
                )
                self.assertEqual(
                    measured["kv_cache_bytes"],
                    row["kv_cache_bytes_bf16_by_loops"][str(loops)],
                    f"{preset} loops={loops}",
                )
                self.assertEqual(
                    measured["recurrent_state_bytes"],
                    row["recurrent_state_bytes_fp32_by_loops"][str(loops)],
                    f"{preset} loops={loops}",
                )


class TestReadmeTable(unittest.TestCase):
    """The README table is a claim; it must equal the generated ledger."""

    def _rows(self):
        rows = {}
        for line in README.splitlines():
            match = re.match(r"\|\s*`(MAYA_[A-Z0-9_]+)`\s*\|(.+)\|\s*$", line)
            if match is None:
                continue
            rows[match.group(1)] = [cell.strip() for cell in match.group(2).split("|")]
        return rows

    def test_table_matches_ledger(self):
        rows = self._rows()
        self.assertIn("MAYA_250M", rows)
        for preset, cells in rows.items():
            row = LEDGER[preset]
            self.assertEqual(len(cells), 6, f"{preset} row has {len(cells)} cells")
            self.assertEqual(_first_int(cells[0]), row["total_parameters"], preset)
            self.assertEqual(_first_int(cells[1]), row["active_inference_parameters"], preset)
            self.assertEqual(_first_int(cells[3]), row["kv_cache_bytes_bf16"], preset)
            self.assertEqual(
                _first_int(cells[4]), row["dense_counterfactual_kv_bytes_bf16"], preset
            )
            self.assertEqual(_first_int(cells[5]), row["recurrent_state_bytes_fp32"], preset)

    def test_readme_does_not_repeat_the_zero_kv_loop_claim(self):
        self.assertNotIn("zero extra parameters or KV cache", README)
        self.assertIn("kv_cache_bytes_bf16_by_loops", README)

    def test_model_docstring_matches_the_loop_cache_ledger(self):
        """P0.10: the module docstring may not claim a zero-KV recurrent loop."""
        with open(
            os.path.join(PROJECT_ROOT, "maya250m", "model.py"), encoding="utf-8"
        ) as handle:
            source = handle.read()
        self.assertNotIn("without increasing parameter count or KV cache", source)
        self.assertIn("+128 MiB", source)
        self.assertIn("402,653,184", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
