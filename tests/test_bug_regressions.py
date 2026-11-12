"""Phase 0B regression suite for BUG-05, BUG-08, BUG-09 and BUG-10.

* BUG-05: WY chunkwise scan must preserve a non-zero initial GDN state on the very
  first chunk when no ``prev_doc`` is supplied (prefill path).
* BUG-08: ``single_row_bound_contractor`` must recursively rewind both v1 flat list
  caches and v2 structured multi-loop caches.
* BUG-09: ``extract_ast_symbol_table`` must see multi-line signatures, ``async def``
  coroutines and class-scoped methods.
* BUG-10: Vexx CAS / ``<experiment>`` splice loops must never overrun the generation
  budget or the model context window.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m import BPETokenizer, Vexx35Controller, generate_maya
from tests.test_maya250m import reference_recurrent_gdn_numpy


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:  # pragma: no cover
        return None
    return torch


class TestPhase0BRegressions(unittest.TestCase):
    def test_bug05_first_chunk_preserves_initial_state(self):
        torch = _torch()
        if torch is None:
            self.skipTest("PyTorch not installed")
        from maya250m.model import GatedDeltaNetLayer

        rng = np.random.default_rng(0)
        b, n, h, hd = 2, 32, 4, 16
        q = torch.from_numpy(rng.standard_normal((b, n, h, hd)))
        q = q / q.norm(dim=-1, keepdim=True)
        k = torch.from_numpy(rng.standard_normal((b, n, h, hd)))
        k = k / k.norm(dim=-1, keepdim=True)
        v = torch.from_numpy(rng.standard_normal((b, n, h, hd)))
        g = torch.from_numpy(rng.uniform(0.75, 0.99, size=(b, n, h)))
        beta = torch.from_numpy(rng.uniform(0.1, 1.95, size=(b, n, h)))
        S0 = torch.from_numpy(rng.standard_normal((b, h, hd, hd)) * 0.1).float()
        doc_ids = torch.from_numpy(
            np.array(
                [
                    [
                        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                        -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                    ]
                ],
                dtype=np.int64,
            )
        )

        ref_out, ref_state = reference_recurrent_gdn_numpy(
            q.numpy(),
            k.numpy(),
            v.numpy(),
            g.numpy(),
            beta.numpy(),
            S0.numpy(),
            doc_ids=doc_ids.numpy(),
            prev_doc=None,
        )

        layer = GatedDeltaNetLayer(
            d=64, n_heads=4, conv_kernel=4, beta_scale=2.0, chunk_size=8
        )
        layer.eval()
        out, state = layer._chunk_parallel_scan(
            q, k, v, g, beta, S0, doc_ids=doc_ids, prev_doc=None
        )
        out = out.detach().numpy()
        state = state.detach().numpy()

        self.assertLess(float(np.max(np.abs(ref_out - out))), 1e-5)
        self.assertLess(float(np.max(np.abs(ref_state - state))), 1e-5)

    def test_bug08_contractor_rewinds_v2_structured_cache(self):
        torch = _torch()
        if torch is None:
            self.skipTest("PyTorch not installed")
        from maya250m.model import create_empty_hybrid_cache_v2

        vexx = Vexx35Controller()
        bundle = create_empty_hybrid_cache_v2(
            num_loops=2,
            prelude_layers=1,
            core_layers=1,
            coda_layers=1,
            batch_size=1,
            config_hash="stub",
        )
        bundle["position"] = 10
        bundle["prelude"][0] = {
            "k": torch.ones(1, 4, 10, 8),
            "v": torch.ones(1, 4, 10, 8),
        }
        bundle["core_loops"][1][0] = {
            "k": torch.ones(1, 4, 10, 8),
            "v": torch.ones(1, 4, 10, 8),
        }
        bundle["coda"][0] = {
            "state": torch.ones(1, 4, 8, 8),
            "conv": torch.zeros(1, 3, 24),
        }

        vexx.save_checkpoint(
            "x", token_idx=4, hybrid_cache=bundle, entropy=1.0, proposed_val=2.0
        )
        idx, rewound, bounds = vexx.single_row_bound_contractor(
            {"x"}, {"coeffs": {"x": 1}, "rhs": 3}, {"x": 2.0}
        )

        self.assertEqual(idx, 4)
        self.assertIsInstance(rewound, dict)
        self.assertEqual(rewound["prelude"][0]["k"].shape[2], 4)
        self.assertEqual(rewound["core_loops"][1][0]["k"].shape[2], 4)
        self.assertEqual(rewound["position"], 4)
        # non-sliceable GDN state is carried through untouched
        self.assertEqual(rewound["coda"][0]["state"].shape, (1, 4, 8, 8))
        # exact rational bound 3/1 widened outward by one ULP; no declared lower bound
        self.assertEqual(bounds[0], -float("inf"))
        self.assertEqual(bounds[1], np.nextafter(3.0, np.inf))

    def test_bug09_multiline_async_and_class_scoped_signatures(self):
        code = (
            "def ok(a, b=2):\n"
            "    return a + b\n"
            "\n"
            "async def fetch(url, timeout=1.0, *args):\n"
            "    return url\n"
            "\n"
            "class C:\n"
            "    def method(self, x):\n"
            "        return x\n"
        )
        table = Vexx35Controller.extract_ast_symbol_table(code)
        self.assertEqual(table["ok"]["min_arity"], 1)
        self.assertEqual(table["ok"]["max_arity"], 2)
        self.assertIsNone(table["fetch"]["max_arity"])
        # url required, timeout has a default, *args makes arity unbounded
        self.assertEqual(table["fetch"]["min_arity"], 1)
        self.assertEqual(table["method"]["min_arity"], 1)
        self.assertEqual(table["method"]["max_arity"], 1)

        # In-flight, syntactically incomplete multi-line signature.
        partial = "def compute(\n    a: int,\n    b: int,\n    c: int = 0,\n) -> int:"
        partial_table = Vexx35Controller.extract_ast_symbol_table(partial)
        self.assertIn("compute", partial_table)
        self.assertEqual(partial_table["compute"]["min_arity"], 2)
        self.assertEqual(partial_table["compute"]["max_arity"], 3)

    def test_bug10_cas_splice_respects_generation_budget(self):
        torch = _torch()
        if torch is None:
            self.skipTest("PyTorch not installed")

        tok = BPETokenizer()
        eq_id = tok.token_to_id["="]

        class _StubModel:
            max_ctx = 64

            def eval(self):
                return self

            def prefill(self, input_ids, doc_ids=None, num_loops=None):
                b, n = input_ids.shape
                logits = torch.full((b, tok.vocab_size), -1e4)
                logits[:, eq_id] = 0.0
                bundle = {
                    "schema_version": 2,
                    "num_loops": 1,
                    "position": int(n),
                    "batch_size": b,
                    "config_hash": "stub",
                    "prelude": [],
                    "core_loops": [],
                    "coda": [],
                }
                info = {
                    "truncated": False,
                    "tokens_dropped": 0,
                    "effective_len": int(n),
                    "num_loops": 1,
                }
                return logits, bundle, info

            def decode_step(self, token_id, cache_bundle, seq_start=None, return_hidden=False):
                n = int(token_id.shape[1])
                logits = torch.full((1, tok.vocab_size), -1e4)
                logits[:, eq_id] = 0.0
                bundle = dict(cache_bundle)
                bundle["position"] = int(bundle["position"]) + n
                return logits[:, -1, :], bundle

        class _SplicingVexx:
            def apply_in_flight_mask(
                self,
                logits,
                prefix_text,
                vocab_map,
                expected_return_type=None,
                remaining_token_budget=256,
                mode="python",
            ):
                return logits

            def crane_phase(self, text):
                return "UNCONSTRAINED_THINK"

            def crane_cas_evaluate(self, expr):
                return "1234567890123456789012345678901234567890"

        result = generate_maya(
            _StubModel(),
            tok,
            "x <<2+2=",
            max_new_tokens=1,
            vexx=_SplicingVexx(),
            mode="think",
        )
        self.assertLessEqual(result.tokens_generated, 1)
        self.assertEqual(result.tokens_generated, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
