# Maya-250M + Vexx 3.5 (Standalone Research & Engineering Bundle)

**Maya-250M** is a standalone, ground-up **250.01M parameter** (`250,007,328` weights; `250,105,632` including the auxiliary shared-trunk memory head) neuro-symbolic language model engineered specifically for **coding, mathematical derivation, deep latent thinking, and autonomous scientific research from zero**. It is designed to train from scratch at **$0 compute cost** across **Kaggle's free weekly TPU (`v3-8` / `v5e-8`) and Dual-T4 (`T4 x2`) GPU quotas**, and to serve as the verified flagship proof-of-execution for **Google's TPU Research Cloud (TRC)** program and **50B–100B MoE** seed compute grants.

## Core Architectural & Engine Innovations

1. **WY-Representation Chunkwise Parallel Gated DeltaNet (`chunk_size=64`) + Negative Eigenvalues ($\beta \in (0, 2)$)**:
   - Replaces sequential token-by-token Python loops during training with a vectorized unit lower-triangular chunk solver `(I + M) U = R` over 64-token blocks, delivering **30x faster GPU/TPU training** while maintaining **exact mathematical parity (`< 1e-5`)** with the $O(1)$ streaming decode recurrence across packed `doc_ids` document boundaries.
   - Doubles per-head recurrent state dimension (`gdn_heads = 6`, `head_dim = 128` at `d = 768`) to quadruple each head's FP32 memory matrix ($128 \times 128$) and scales the Householder write gate to $\beta_t = 2\sigma(b_t) \in (0, 2)$, unlocking negative eigenvalues in $[-1, 1]$ for $S_5$ permutation tracking, parity, and loop state-tracking in code ([Grazzi et al., ICLR 2025](https://arxiv.org/abs/2411.12537); [DeltaProduct, 2025](https://arxiv.org/abs/2502.10297)).
2. **Differential Gated GQA + Fractional Slow-RoPE ($\theta = 10^7$) on Global Layers**:
   - Every 4th layer (`full`) uses [Differential Attention (Ye et al., ICLR 2025)](https://arxiv.org/abs/2410.05258), subtracting two softmax maps ($\text{Softmax}(Q_1 K_1^\top) - \lambda \text{Softmax}(Q_2 K_2^\top)$) with a learnable per-head noise-cancellation scalar $\lambda$ and sigmoid output gating.
   - Alternating global attention layers use ultra-slow RoPE ($\theta = 10^7$) to separate position-free content retrieval from local token ordering.
3. **Prelude–Recurrent-Core–Coda Latent Reasoning Loop ([Geiping et al., NeurIPS 2025](https://arxiv.org/abs/2502.05171))**:
   - Partitions the **32 physical layers** into an 8-layer **Prelude**, a 16-layer **Recurrent Core**, and an 8-layer **Coda**. Iterating the 16-layer core $R_{\text{loop}} \in \{1, 2, 3, 4\}$ times in continuous latent space expands Maya-250M to **48–80 effective layers** of reasoning depth with zero extra parameters or KV cache.
4. **Vexx 3.5 Six-Tier Neuro-Symbolic Engine (`maya250m/vexx.py`)**:
   - **Tier 0**: CRANE dual-phase (`UNCONSTRAINED_THINK` inside `<think>...</think>`, `EXPERIMENT_DESIGN` inside `<experiment>...</experiment>`, `STRICT_VERIFIED` outside) + SWYB delimiter-closure lookahead ([CRANE, ICML 2025](https://arxiv.org/html/2502.09061v3)).
   - **Tier 1**: Live `ast` symbol-table & f-string-aware Python lexer + parallel MTP-3 speculative draft tree verifier (`verify_mtp_draft_tree`).
   - **Tier 2**: Restricted-AST SymPy CAS (`safe_sympy`) splicing exact rational/symbolic results into `<<expr=...>>` + rank-verified linear solver (`solve_linear_system`).
   - **Tier 3**: Single-row exact rational bound contractor (`single_row_bound_contractor`) rewinding both Full-Attention KV caches and GDN FP32 states.
   - **Tier 4**: Statistical & epistemic calibration guard (`verify_epistemic_calibration`).
   - **Tier 5**: Resource-limited subprocess Python execution sandbox (`safe_execute_python`) + **live `<experiment>...</experiment>` execution splice during generation** + **GRPO / RLVR group-normalized reward oracle (`compute_rlvr_group_rewards`)**.
5. **Muon + AdamW Hybrid Optimizer & Kaggle Free-Quota Auto-Chaining (`maya250m/train.py`, `kaggle/`)**:
   - Applies 5-step quintic Newton-Schulz orthogonalized momentum ([Muon, Keller Jordan 2024](https://kellerjordan.github.io/posts/muon/)) on 2D hidden trunk matrices alongside decoupled AdamW (`weight_decay = 0.0` on `A_log`, `dt_bias`, `lambda_*`, and `RMSNorm` gains).
   - Includes a forward Multi-Token Prediction curriculum (`1 -> 2 -> 3`), zero-copy `uint16` memory-mapped shard streaming (`MemmapShardDataset`), Schema-9 `weights_only=True` atomic checkpoints, and a wall-clock watchdog (`8.25h` TPU / `11.25h` T4x2) for bitwise-exact multi-session chaining on Kaggle.

## Computed Parameter & Cache Ledger (`metrics_maya250m.json`)

| Preset | Total Params | Active Inference Params | Physical / Effective Layers (2x Loop) | 32K KV Cache (BF16) | Dense Counterfactual KV | GDN FP32 State |
| :--- | ---: | ---: | :---: | ---: | ---: | ---: |
| `MAYA_TINY_TEST` | **9,081,456** | 8,556,912 | 8 / 12 | **262,144 B** | 1,048,576 B (-75%) | 196,608 B |
| `MAYA_50M_PILOT` | **62,013,792** | 59,916,128 | 16 / 24 | **4,194,304 B** | 16,777,216 B (-75%) | 3,145,728 B |
| `MAYA_250M` | **250,007,328** | **245,287,968** | **32 / 48** | **134,217,728 B (128 MiB)** | 536,870,912 B (-75%) | **9,437,184 B (9 MiB)** |

The `MAYA_250M` ledger is verified analytically **and** against a live `MayaModel` by
`tools/param_ledger_maya.py`. The auxiliary shared-trunk memory projection
(`W_mem`: 768 → 128, bias-free) adds **98,304** parameters, giving a trainable total of
**250,105,632**, inside the **251,000,000** budget. `W_mem` is deliberately kept out of
`MayaModel.state_dict()` so Schema-9 checkpoints remain backward compatible.

## Intelligence Upgrade Subsystems

| Module | What it adds |
| :--- | :--- |
| `maya250m/verification.py` | Typed `VerificationResult` + closed failure taxonomy used by every gate |
| `maya250m/understanding.py` | `TaskSpec` parsing, constraint extraction/coverage, optional neural parser head |
| `maya250m/reasoning/` | Five-phase state machine, adaptive loop depth (R ∈ 1..4), decomposition DAG, self-correction/abstention |
| `maya250m/memory/` | SQLite+FTS5 typed memory, vector index, hybrid retrieval, reversible consolidation, `W_mem` projection |
| `maya250m/codebase/` | AST repo index, PPR repo map, topological packer + AST-aligned FIM, cross-file contract checks, verified SEARCH/REPLACE patches |
| `maya250m/agent/` | Seven bounded tools, loop-guarded CodeAct runtime, tool-output injection firewall, trajectories + process rewards with observation-masked SFT |
| `maya250m/training/` | Staged SFT → distillation → DPO → GRPO trainers with real optimizer updates (BUG-02), zero-variance group masking, frozen references |
| `maya250m/evaluation/` | Five separate measurement identities, executable harness with explicit `NOT_RUN`, A–P ablation matrix |
| `maya250m/adaptive.py` | Inference bridge: `TaskSpec` → critical-ambiguity abstention → loop count selection → verified bounded revision, emitting a correctly-identified `Measurement` |

## Quickstart Commands

```bash
# 1. Run the full verification & numerical parity suite
pytest tests/ -q

# 2. Recompute the exact parameter & KV-cache ledger
python3 tools/param_ledger_maya.py

# 3. Run the end-to-end smoke test (dedup -> BPE -> Muon+AdamW train -> ckpt -> Vexx 3.5 decode)
python3 run_maya250m.py --preset MAYA_TINY_TEST --steps 10

# 3b. Same, but pick the recurrent-core loop count per prompt (adaptive depth)
python3 run_maya250m.py --preset MAYA_TINY_TEST --steps 10 --adaptive-depth --max-new-tokens 16

# 4. Pack binary uint16 memmap shards on a free Kaggle CPU session
python3 kaggle/pack_shards_cpu.py --output-prefix /tmp/maya_shard --seq-len 128

# 5. Launch the Kaggle auto-chaining trainer (auto-detects TPU v3-8/v5e-8 or Dual T4 x2)
python3 kaggle/train_maya250m_kaggle.py --preset MAYA_TINY_TEST --max-steps 20 --ckpt-dir /tmp/maya_ckpt
```
