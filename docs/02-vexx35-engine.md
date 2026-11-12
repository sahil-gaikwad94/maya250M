# 02 — Vexx 3.5 Six-Tier Neuro-Symbolic Engine

`Vexx35Controller` (`maya250m/vexx.py`) runs inside both `generate_maya` and the post-training RLVR / GRPO loop:

1. **Tier 0 — CRANE Dual Phase + `<experiment>` Protocol + SWYB Budget**:
   - Inside `<think>...</think>`, generation is unconstrained (`UNCONSTRAINED_THINK`) to preserve reasoning accuracy ([CRANE, ICML 2025](https://arxiv.org/html/2502.09061v3)).
   - Inside `<experiment>...</experiment>`, `EXPERIMENT_DESIGN` enforces valid Python syntax for miniature hypothesis tests.
   - Outside `<think>`, `STRICT_VERIFIED` enforces strict target grammar and SWYB delimiter-closure lookahead.
2. **Tier 1 — Live AST Symbol Table + f-String Lexer + Parallel MTP-3 Tree Verifier**:
   - Tracks single/double/triple/raw strings, f-string `{...}` expression holes, comments, callee receiver spoofing, and function arity.
   - `verify_mtp_draft_tree()` verifies 3-token speculative drafts proposed by `MayaModel.propose_mtp_drafts()` in a single pass.
3. **Tier 2 — Restricted-AST SymPy CAS Splice (`safe_sympy`)**:
   - Evaluates `<<expr=...>>` symbolically using exact `sp.Rational` arithmetic while blocking `sympify` code injection, plus rank-verified `solve_linear_system()`, `integrate_definite()`, and `modular_pow()`.
4. **Tier 3 — Single-Row Rational Bound Contractor + Hybrid Cache Jumpback**:
   - Propagates exact `Fraction` bounds along linear inequalities, widens outward by 1 ULP (`math.nextafter`), and rewinds both Full-Attention KV caches and GDN FP32 states to the culprit decision checkpoint.
5. **Tier 4 — Statistical & Epistemic Calibration Guard**:
   - Flags confidence-interval / p-value disagreements (`CONFLICT`), repeating decimals (`0.999... == 1`), voluntary response bias, and internal contradictions.
6. **Tier 5 — Isolated Subprocess Sandbox + Live `<experiment>` Splice + RLVR Oracle**:
   - Executes candidate functions and live `<experiment>` blocks inside an isolated `python -I -S` subprocess (`RLIMIT_CPU=2s`, `RLIMIT_AS=512 MiB`).
   - `compute_rlvr_group_rewards()` computes group-normalized GRPO advantages directly from unit-test execution for reward-hacking-proof RLVR ([DeepSeek-R1](https://arxiv.org/abs/2501.12948); [RLVR](https://arxiv.org/abs/2506.14245)).
