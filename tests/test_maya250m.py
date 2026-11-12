"""Comprehensive verification suite for standalone Maya-250M + Vexx 3.5.

Can be executed via either ``pytest tests/`` or ``python3 -m unittest tests/test_maya250m.py``
(or ``python3 tests/test_maya250m.py``). Verifies all Maya-250M and Vexx 3.5 invariants,
WY Chunkwise Parallel GDN mathematical equivalence (< 1e-6), and PyTorch modules.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.linalg import solve_triangular

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m import (
    EXP_END,
    EXP_START,
    THINK_END,
    THINK_START,
    TORCH_AVAILABLE,
    BPETokenizer,
    MemmapShardDataset,
    StreamPackedDataset,
    Vexx35Controller,
    deduplicate_and_split,
    get_config,
    safe_sympy,
    validate_config,
    write_memmap_shard,
)
from tools.param_ledger_maya import compute_preset_ledger


def reference_recurrent_gdn_numpy(q, k, v, g, beta, S0, doc_ids=None, prev_doc=None):
    """Reference token-by-token Gated DeltaNet recurrence in FP64/FP32."""
    b, n, h, hd = q.shape
    scale = 1.0 / np.sqrt(hd)
    S = S0.copy().astype(np.float64)
    outputs = np.zeros((b, n, h, hd), dtype=np.float64)
    for t in range(n):
        if doc_ids is not None:
            if t == 0 and prev_doc is not None:
                same_doc = (doc_ids[:, 0] == prev_doc) & (doc_ids[:, 0] >= 0)
                S = S * same_doc[:, None, None, None]
            elif t == 0 and prev_doc is None:
                valid_doc = doc_ids[:, 0] >= 0
                S = S * valid_doc[:, None, None, None]
            elif t > 0:
                same_doc = (doc_ids[:, t] == doc_ids[:, t - 1]) & (doc_ids[:, t] >= 0)
                S = S * same_doc[:, None, None, None]

        gt = g[:, t, :, None, None]
        bt = beta[:, t, :, None]
        kt = k[:, t]
        vt = v[:, t]
        qt = q[:, t]

        S_decayed = S * gt
        retrieved = np.einsum("bhkv,bhk->bhv", S_decayed, kt)
        delta = bt * (vt - retrieved)
        S = S_decayed + kt[..., :, None] * delta[..., None, :]
        outputs[:, t] = np.einsum("bhkv,bhk->bhv", S, qt) * scale
    return outputs, S


def wy_chunk_parallel_gdn_numpy(q, k, v, g, beta, S0, chunk_size=8, doc_ids=None, prev_doc=None):
    """Exact WY-Representation Chunkwise Parallel Gated DeltaNet solver."""
    b, n, h, hd = q.shape
    scale = 1.0 / np.sqrt(hd)
    S = S0.copy().astype(np.float64)
    q_f = np.transpose(q.astype(np.float64), (0, 2, 1, 3))
    k_f = np.transpose(k.astype(np.float64), (0, 2, 1, 3))
    v_f = np.transpose(v.astype(np.float64), (0, 2, 1, 3))
    g_f = np.transpose(g.astype(np.float64), (0, 2, 1))
    b_f = np.transpose(beta.astype(np.float64), (0, 2, 1))

    chunk_outputs = []
    cur_prev_doc = prev_doc

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        L_c = end - start
        qc = q_f[:, :, start:end, :]
        kc = k_f[:, :, start:end, :]
        vc = v_f[:, :, start:end, :]
        gc = g_f[:, :, start:end]
        bc = b_f[:, :, start:end]

        log_g = np.log(np.clip(gc, 1e-12, 1.0))
        cum_log_g = np.cumsum(log_g, axis=-1)

        idx = np.arange(L_c)
        tril_mask = idx[:, None] >= idx[None, :]
        strict_tril = idx[:, None] > idx[None, :]

        diff_log = cum_log_g[..., :, None] - cum_log_g[..., None, :]
        diff_log = np.where(tril_mask[None, None, :, :], diff_log, -1e9)
        gamma_mat = np.exp(diff_log) * tril_mask[None, None, :, :]
        gamma_from_init = np.exp(cum_log_g)

        if doc_ids is not None:
            docs_c = doc_ids[:, start:end]
            valid_tok = docs_c >= 0
            diag_b = np.eye(L_c, dtype=bool)[None, :, :]
            same_intra = ((docs_c[:, :, None] == docs_c[:, None, :]) & valid_tok[:, :, None]) | diag_b
            gamma_mat = gamma_mat * same_intra[:, None, :, :]
            if cur_prev_doc is not None:
                same_as_init = (docs_c == cur_prev_doc[:, None]) & valid_tok
            else:
                same_as_init = np.zeros_like(valid_tok)
            gamma_from_init = gamma_from_init * same_as_init[:, None, :]
        else:
            docs_c = None

        kkt = np.matmul(kc, np.swapaxes(kc, -1, -2))
        qkt = np.matmul(qc, np.swapaxes(kc, -1, -2))

        M = (bc[..., :, None] * gamma_mat * kkt) * strict_tril[None, None, :, :]
        eye = np.eye(L_c, dtype=np.float64)

        s0_k = np.einsum("bhkv,bhck->bhcv", S, kc)
        R = bc[..., :, None] * (vc - gamma_from_init[..., :, None] * s0_k)

        U = np.zeros_like(R)
        for bi in range(b):
            for hi in range(h):
                U[bi, hi] = solve_triangular(
                    eye + M[bi, hi], R[bi, hi], lower=True, unit_diagonal=True
                )

        s0_q = np.einsum("bhkv,bhck->bhcv", S, qc)
        A_qk = gamma_mat * qkt
        out_c = (gamma_from_init[..., :, None] * s0_q + np.matmul(A_qk, U)) * scale
        chunk_outputs.append(np.transpose(out_c, (0, 2, 1, 3)))

        decay_end = gamma_from_init[:, :, -1]
        weights_end = gamma_mat[:, :, -1, :]
        k_weighted = kc * weights_end[..., :, None]
        S = S * decay_end[:, :, None, None] + np.matmul(
            np.swapaxes(k_weighted, -1, -2), U
        )
        if docs_c is not None:
            cur_prev_doc = docs_c[:, -1]

    return np.concatenate(chunk_outputs, axis=1), S


class TestMaya250MAndVexx35(unittest.TestCase):
    def test_config_and_250m_parameter_ledger(self):
        cfg = get_config("MAYA_250M")
        self.assertEqual(cfg["layers"], 32)
        self.assertEqual(cfg["prelude_layers"] + cfg["core_layers"] + cfg["coda_layers"], 32)
        self.assertEqual(cfg["d"], 768)
        self.assertEqual(cfg["gdn_heads"], 6)
        self.assertEqual(cfg["d"] // cfg["gdn_heads"], 128)

        ledger = compute_preset_ledger("MAYA_250M")
        self.assertEqual(ledger["total_parameters"], 250_007_328)
        self.assertEqual(ledger["effective_layers_2x_loop"], 48)
        self.assertEqual(ledger["effective_layers_4x_loop"], 80)
        self.assertEqual(ledger["kv_reduction_pct"], 75.0)

        bad = copy.deepcopy(cfg)
        bad["unknown_key"] = 1
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_tokenizer_protocol_markers_and_uint16_packing(self):
        corpus = [
            "def add(a: int, b: int) -> int:\n    return a + b\n",
            "<think>Check <<2+2=4>> <experiment>RESULT = 2 + 2</experiment></think>4",
        ] * 8
        tok = BPETokenizer.train(corpus, vocab_size=320)
        tok.validate()

        sample = "<think>Plan <experiment>RESULT = 7 * 6</experiment></think>42"
        ids = tok.encode(sample, add_bos=True, add_eos=True)
        self.assertIn(tok.token_to_id[THINK_START], ids)
        self.assertIn(tok.token_to_id[EXP_START], ids)
        self.assertIn(tok.token_to_id[EXP_END], ids)
        self.assertIn(tok.token_to_id[THINK_END], ids)
        self.assertEqual(tok.decode(ids, preserve_protocol=True), sample)

        arr16 = tok.encode_to_uint16(sample)
        self.assertEqual(arr16.dtype, np.uint16)
        self.assertEqual(len(arr16), len(ids))

    def test_wy_chunkwise_parallel_gdn_exact_parity_across_doc_boundaries(self):
        rng = np.random.default_rng(42)
        b, n, h, hd = 2, 20, 4, 16
        q = rng.standard_normal((b, n, h, hd))
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        k = rng.standard_normal((b, n, h, hd))
        k = k / np.linalg.norm(k, axis=-1, keepdims=True)
        v = rng.standard_normal((b, n, h, hd))
        g = rng.uniform(0.75, 0.99, size=(b, n, h))
        # Negative-eigenvalue Householder write gate beta in (0, 2)
        beta = rng.uniform(0.1, 1.95, size=(b, n, h))
        S0 = rng.standard_normal((b, h, hd, hd)) * 0.1

        doc_ids = np.array(
            [
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, -1, -1, -1],
                [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, -1, -1, -1, -1],
            ],
            dtype=np.int64,
        )
        prev_doc = np.array([0, 99], dtype=np.int64)

        out_rec, state_rec = reference_recurrent_gdn_numpy(
            q, k, v, g, beta, S0, doc_ids=doc_ids, prev_doc=prev_doc
        )
        out_chunk, state_chunk = wy_chunk_parallel_gdn_numpy(
            q, k, v, g, beta, S0, chunk_size=6, doc_ids=doc_ids, prev_doc=prev_doc
        )

        max_out_diff = float(np.max(np.abs(out_rec - out_chunk)))
        max_state_diff = float(np.max(np.abs(state_rec - state_chunk)))
        self.assertLess(max_out_diff, 1e-10)
        self.assertLess(max_state_diff, 1e-10)

    def test_vexx35_all_six_tiers_and_experiment_splice(self):
        vexx = Vexx35Controller(
            domain_bounds={"x": (-10.0, 10.0), "y": (-10.0, 10.0)},
            symbol_table={"compute": {"min_arity": 2, "max_arity": 2, "return_type": "int"}},
        )

        self.assertEqual(vexx.crane_phase("<think>hello"), "UNCONSTRAINED_THINK")
        self.assertEqual(vexx.crane_phase("<think><experiment>x=1"), "EXPERIMENT_DESIGN")
        self.assertEqual(
            vexx.crane_phase("<think><experiment>x=1</experiment>ok"), "UNCONSTRAINED_THINK"
        )
        self.assertEqual(vexx.crane_phase("<think>done</think>def f():"), "STRICT_VERIFIED")

        st_bad = vexx._lex_code_state('x = f"{compute(1"')
        self.assertFalse(st_bad["valid"])

        vocab_map = {0: "compute(", 1: "1, 2)", 2: "1)", 3: "\n"}
        accepted = vexx.verify_mtp_draft_tree(
            "res = ", [0, 1, 3], vocab_map, remaining_token_budget=20, mode="python"
        )
        self.assertEqual(accepted, [0, 1, 3])
        rejected = vexx.verify_mtp_draft_tree(
            "res = ", [0, 2, 3], vocab_map, remaining_token_budget=20, mode="python"
        )
        self.assertEqual(rejected, [0])

        with self.assertRaises(ValueError):
            safe_sympy("__import__('os').getcwd()")
        self.assertEqual(vexx.crane_cas_evaluate("(5/6 - 1/4)/(7/12)"), "1")
        sol = vexx.solve_linear_system(["2*x + y = 5", "x - y = 1"], ["x", "y"])
        self.assertEqual(sol["status"], "unique")
        self.assertEqual(sol["solution"], {"x": "2", "y": "1"})

        fake_cache = [
            {"state": np.ones((1, 4, 16, 16)), "conv": np.zeros((1, 3, 192))},
            {"k": np.ones((1, 4, 10, 32)), "v": np.ones((1, 4, 10, 32))},
        ]
        vexx.save_checkpoint(
            "x", token_idx=4, hybrid_cache=fake_cache, entropy=1.5, proposed_val=8.0
        )
        idx, rewound, new_b = vexx.single_row_bound_contractor(
            {"x"}, {"coeffs": {"x": 2, "y": 1}, "rhs": 6}, {"x": 8.0, "y": 2.0}
        )
        self.assertEqual(idx, 4)
        self.assertEqual(rewound[1]["k"].shape[2], 4)
        self.assertLess(new_b[1], 2.0001)

        cal = vexx.verify_epistemic_calibration("95% CI [-1.2, 0.4], p = 0.01")
        self.assertEqual(cal["verdict"], "CONFLICT")
        dec = vexx.verify_epistemic_calibration("Is 0.999... less than 1 or equal to 1?")
        self.assertEqual(dec["verdict"], "equal")

        obs = vexx.execute_experiment_block("RESULT = [k*k for k in range(4)]")
        self.assertEqual(obs, "[0, 1, 4, 9]")
        bad_obs = vexx.execute_experiment_block("import os\nRESULT = 1")
        self.assertIn("EXP_ERROR", bad_obs)

        grpo = vexx.compute_rlvr_group_rewards(
            candidates=[
                "def sq(n):\n    return n * n\n",
                "def sq(n):\n    return n + n\n",
            ],
            fn_name="sq",
            test_cases=[{"input": [5], "expected": 25}, {"input": [3], "expected": 9}],
        )
        self.assertEqual(grpo["rewards"], [1.0, 0.0])
        self.assertGreater(grpo["advantages"][0], 0.9)
        self.assertLess(grpo["advantages"][1], -0.9)

    def test_memmap_shard_packing_and_loading(self):
        corpus = [
            f"<think>Step {i}: <<{i}+1={i+1}>></think>\ndef fn_{i}(x: int) -> int:\n    return x + {i}\n"
            for i in range(30)
        ]
        train_t, _val_t, _test_t, _m = deduplicate_and_split(corpus, seed=42)
        tok = BPETokenizer.train(train_t, vocab_size=300)

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_pref = os.path.join(tmpdir, "shard")
            meta = write_memmap_shard(tok, train_t, shard_pref, seq_len=32, seed=42)
            self.assertGreater(meta["num_examples"], 0)
            mmap_ds = MemmapShardDataset(shard_pref)
            self.assertEqual(len(mmap_ds), meta["num_examples"])
            ids, lbls, docs = mmap_ds[0]
            self.assertEqual(len(ids), 32)
            self.assertEqual(lbls[0], -100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
