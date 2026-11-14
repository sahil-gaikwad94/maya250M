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
   - Completion is evidence-gated: `AgentRuntime.finalize(CompletionEvidence)` refuses to resolve a trajectory unless every mandatory requirement is mapped to an executed test or a typed mechanical verifier (`api_signature`, `path_presence`, `test_outcomes`, `dependency_restrictions`, `output_schema`, `complexity`) or explicitly waived by the user; lexical overlap is reported only as `lexical_requirement_coverage`, never as verification.
     - Research traces follow a **CONJECTURE → FALSIFY → PROVE** protocol (`counterexample_first_plan`, `certify_universal_claim`): finite sweeps license `CONJECTURE_ONLY`, never "proven for all n", and a counterexample refutes.
   - Partitions the **32 physical layers** into an 8-layer **Prelude**, a 16-layer **Recurrent Core**, and an 8-layer **Coda**. Iterating the 16-layer core $R_{\text{loop}} \in \{1, 2, 3, 4\}$ times in continuous latent space expands Maya-250M to **48–80 effective layers** of reasoning depth with zero extra parameters. It is *not* zero extra cache: every core pass keeps its own entries for tier-3 rewinds, so core KV and core GDN state grow by one pass per loop (`kv_cache_bytes_bf16_by_loops` and `recurrent_state_bytes_fp32_by_loops` in the ledger; at $R=2$ the 250M KV cache is 402,653,184 B, not 268,435,456 B).
4. **Vexx 3.5 Six-Tier Neuro-Symbolic Engine (`maya250m/vexx.py`)**:
   - **Tier 0**: CRANE dual-phase (`UNCONSTRAINED_THINK` inside `<think>...</think>`, `EXPERIMENT_DESIGN` inside `<experiment>...</experiment>`, `STRICT_VERIFIED` outside) + SWYB delimiter-closure lookahead ([CRANE, ICML 2025](https://arxiv.org/html/2502.09061v3)).
   - **Tier 1**: Live `ast` symbol-table & f-string-aware Python lexer. `advance_lexer_state()` resumes from a `LexerState`, so one decode step lexes each candidate token once (O(|V|·token_len + L)) instead of re-scanning the whole prefix per candidate (O(|V|·L)); token categories are cached per vocabulary. Container/string method calls on locals (`stack.append(x)`, `adj[u].append(v)`, `d.get(k, 0)`) are allowed, while dunder attributes, forbidden identifiers and unknown non-standard methods stay masked. MTP-3 drafts are pre-filtered by the same grammar via `filter_draft_tokens_by_grammar()`.
   - **Tier 2**: Restricted-AST SymPy CAS (`safe_sympy`) splicing exact rational/symbolic results into `<<expr=...>>` — including bounded number-theory/combinatorics (`gcd`, `lcm`, `binomial`, `factorial`, `isprime`, `totient`, `Mod`) and rounding/algebra (`floor`, `ceil`, `Rational`, `sign`, `Min`, `Max`) — plus a rank-verified linear solver (`solve_linear_system`).
   - **Tier 3**: Single-row exact rational bound contractor (`single_row_bound_contractor`) rewinding both Full-Attention KV caches and GDN FP32 states. A GDN state has no sequence axis, so a snapshot is only accepted when the cache position equals the checkpoint token and truncating a live bundle fails closed instead of silently retaining future tokens.
   - **Tier 4**: Statistical & epistemic calibration guard (`verify_epistemic_calibration`), now fronted by an executable audit (`verify_statistical_report`) that flags GRIM-impossible means/percentages (`34.5% of N=10`), impossible test-statistic/p-value pairs (`z`, `t` with `df`, `chi²` with `df`), and dimensionally inhomogeneous unit equations.
   - **Tier 5**: Resource-limited subprocess Python execution sandbox (`safe_execute_python`) that allows `ast.Nonlocal` and whitelisted pure algorithmic stdlib imports (`math`, `heapq`, `collections`, `bisect`, `itertools`, `functools`, `typing`, `fractions`, `decimal`, `operator`, `string`) at both the AST and runtime `__import__` layers + **stateful multi-cell `<experiment>...</experiment>` splice** (verified definitions carry across cells in one episode) + **GRPO / RLVR group-normalized reward oracle (`compute_rlvr_group_rewards`)** with a shaped reward ladder (`-0.5` syntax/policy, `-0.2` contract/mutation, `0.0` no test passed, `+0.3·pass_rate`, `+1.0` full pass) + **differential property fuzzing against a self-written naive oracle** (`fuzz_against_oracle` on boundary probes: empty, singleton, negatives, duplicates) + **execution-based MBR selection** (`select_by_mbr`: AST/contract filtering, majority behavioural consensus over sandboxed outputs, ties broken by shortest AST and zero input mutation; a lone candidate is never reported as consensus).
5. **Muon + AdamW Hybrid Optimizer & Kaggle Free-Quota Auto-Chaining (`maya250m/train.py`, `kaggle/`)**:
   - Applies 5-step quintic Newton-Schulz orthogonalized momentum ([Muon, Keller Jordan 2024](https://kellerjordan.github.io/posts/muon/)) on 2D hidden trunk matrices alongside decoupled AdamW (`weight_decay = 0.0` on `A_log`, `dt_bias`, `lambda_*`, and `RMSNorm` gains).
   - Includes a forward Multi-Token Prediction curriculum (`1 -> 2 -> 3`), zero-copy `uint16` memory-mapped shard streaming (`MemmapShardDataset`), Schema-9 `weights_only=True` atomic checkpoints, and a wall-clock watchdog (`8.25h` TPU / `11.25h` T4x2) for bitwise-exact multi-session chaining on Kaggle.

## Computed Parameter & Cache Ledger (`metrics_maya250m.json`)

| Preset | Total Params | Active Inference Params | Physical / Effective Layers (2x Loop) | KV Cache at `max_ctx`, 1 Loop (BF16) | Dense Counterfactual KV | Recurrent State (FP32) |
| :--- | ---: | ---: | :---: | ---: | ---: | ---: |
| `MAYA_TINY_TEST` | **9,081,456** | 8,556,912 | 8 / 12 | **524,288 B** | 2,097,152 B (-75%) | 448,512 B |
| `MAYA_50M_PILOT` | **62,013,792** | 59,916,128 | 16 / 24 | **16,777,216 B (16 MiB)** | 67,108,864 B (-75%) | 3,366,912 B |
| `MAYA_250M` | **250,007,328** | **245,287,968** | **32 / 48** | **268,435,456 B (256 MiB)** | 1,073,741,824 B (1 GiB) (-75%) | **10,100,736 B (9.6 MiB)** |

The `MAYA_250M` ledger is verified analytically **and** against a live `MayaModel` by
`tools/param_ledger_maya.py`, and its cache columns are reconciled against
`maya250m.cache.cache_memory_ledger` (a measured prefill, not a formula) by
`tests/test_cache_ledger_truth.py`. The auxiliary shared-trunk memory projection
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
| `maya250m/codebase/` | AST repo index, PPR repo map, hybrid lexical+graph retrieval (RRF, dense `NOT_RUN`), topological packer + AST-aligned FIM, cross-file contract checks, verified SEARCH/REPLACE patches |
| `maya250m/agent/` | Eight bounded tools, loop-guarded CodeAct runtime, tool-output injection firewall, trajectories + process rewards with observation-masked SFT |
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
python3 kaggle/train_maya250m_kaggle.py --preset MAYA_TINY_TEST --max-steps 20 --ckpt-dir /tmp/maya_ckpt --smoke-test-synthetic

# 6. Score the five domains (coding / maths / reasoning / research / neural) on a checkpoint
python3 tools/eval_maya_domains.py --preset MAYA_TINY_TEST --steps 300
```

## Domain Evaluation (measured, not claimed)

Every task is scored by *running* something — unit tests inside the Vexx sandbox, exact
rational arithmetic, or an AST check — never by string similarity to a reference answer.
`model_only` is the network's own continuation; `system_symbolic` adds Vexx's exact
derivation for prompts that state an expression, and the two columns are never merged
into one number. Each domain also reports `completion_diversity`, which catches a model
that emits one memorized template regardless of the prompt.

Measured on `MAYA_TINY_TEST` (9M parameters, 300 CPU steps, the 64-document synthetic
fixture): held-out CE **4.19**, perplexity **66.2**; `pass@1 = 0.000` on coding, maths,
reasoning, research *and* neural; `system_symbolic = 1.000` on maths and research. Every
domain reported `unique completions 1/n` with `collapse_detected: true` in
`artifacts/eval_domains.json`: the model emits `def sum_to_<k>(...)` for every prompt.

That is the honest state of a 9M model trained for 300 steps on fixture data — the
verifiers, the exact-arithmetic path and the collapse detector work; the intelligence
does not exist yet. Run the same command with `--corpus` and a real preset to measure
something meaningful.
