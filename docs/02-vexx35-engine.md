# 02 — Vexx 3.5 Six-Tier Neuro-Symbolic Engine

`Vexx35Controller` (`maya250m/vexx.py`) runs inside both `generate_maya` and the post-training RLVR / GRPO loop:

1. **Tier 0 — CRANE Dual Phase + `<experiment>` Protocol + SWYB Budget**:
   - Inside `<think>...</think>`, generation is unconstrained (`UNCONSTRAINED_THINK`) to preserve reasoning accuracy ([CRANE, ICML 2025](https://arxiv.org/html/2502.09061v3)).
   - Inside `<experiment>...</experiment>`, `EXPERIMENT_DESIGN` enforces valid Python syntax for miniature hypothesis tests.
   - Outside `<think>`, `STRICT_VERIFIED` enforces strict target grammar and SWYB delimiter-closure lookahead.
2. **Tier 1 — Live AST Symbol Table + Incremental f-String Lexer + MTP-3 Grammar Filter**:
   - Tracks single/double/triple/raw strings, f-string `{...}` expression holes, comments, callee receiver spoofing, and function arity.
   - `advance_lexer_state(state, chunk)` is a resumable `LexerState` transition: a decode step re-lexes only each candidate's own characters instead of the whole prefix, and vocabulary categories are classified once per tokenizer. Incomplete trailing quote runs are carried into the next chunk so token boundaries cannot change how a string is interpreted.
   - `SAFE_STANDARD_METHODS` allows standard list/dict/set/deque/str methods on locals, subscripts and attribute chains (`stack.append(x)`, `adj[u].append(v)`, `d.get(k, 0)`) while dunder attributes and forbidden identifiers remain masked.
   - `filter_draft_tokens_by_grammar()` prunes 3-token speculative drafts proposed by `MayaModel.propose_mtp_drafts()`; distributional verification belongs to `generate_maya`.
3. **Tier 2 — Restricted-AST SymPy CAS Splice (`safe_sympy`)**:
   - Evaluates `<<expr=...>>` symbolically using exact `sp.Rational` arithmetic while blocking `sympify` code injection, plus rank-verified `solve_linear_system()`, `integrate_definite()`, and `modular_pow()`.
   - The function whitelist covers number theory and combinatorics (`gcd`, `lcm`, `binomial`/`comb`, `factorial`, `isprime`, `totient`, `Mod`) and rounding/algebra (`floor`, `ceil`/`ceiling`, `Rational`, `sign`, `Min`, `Max`); `binomial` and `factorial` carry hard argument caps against CPU denial of service.
4. **Tier 3 — Single-Row Rational Bound Contractor + Hybrid Cache Jumpback**:
   - Propagates exact `Fraction` bounds along linear inequalities, widens outward by 1 ULP (`math.nextafter`), and rewinds both Full-Attention KV caches and GDN FP32 states to the culprit decision checkpoint.
   - A GDN recurrent state has no sequence dimension, so `save_checkpoint` requires the cache position to equal the checkpoint token, and `truncate_cache_at_token` refuses to slice a bundle that still carries future recurrent state. Jumpback therefore restores an exact snapshot (or replays via `maya250m.cache.rewind_cache`) instead of corrupting causal parity.
5. **Tier 4 — Statistical & Epistemic Calibration Guard**:
   - Flags confidence-interval / p-value disagreements (`CONFLICT`), repeating decimals (`0.999... == 1`), voluntary response bias, and internal contradictions.
   - `verify_statistical_report()` runs first as an executable audit: GRIM-impossible percentages/means at a stated `N`, test-statistic/p-value pairs no `df` can produce (`z`, `t`, `chi²`, exact gamma-recursion tails), and unit-dimension vectors that disagree across `=`. It returns `None` when nothing is provably wrong, so the heuristic checks above keep their behaviour.
6. **Tier 5 — Isolated Subprocess Sandbox + Live `<experiment>` Splice + RLVR Oracle**:
   - Executes candidate functions and live `<experiment>` blocks inside an isolated `python -I -S` subprocess (`RLIMIT_CPU=2s`, `RLIMIT_AS=512 MiB`).
   - `ast.Nonlocal` is allowed and imports are restricted to pure algorithmic stdlib modules (`math`, `heapq`, `collections`, `bisect`, `itertools`, `functools`, `typing`, `fractions`, `decimal`, `operator`, `string`) at both the AST policy layer and the runtime `__import__` layer.
   - Succeeding `<experiment>` cells contribute their function/class/import definitions (never `RESULT`) to the episode preamble, so multi-step research traces work without `NameError`.
   - `compute_rlvr_group_rewards()` computes group-normalized GRPO advantages directly from unit-test execution for reward-hacking-proof RLVR ([DeepSeek-R1](https://arxiv.org/abs/2501.12948); [RLVR](https://arxiv.org/abs/2506.14245)). The reward ladder is shaped (`-0.5` syntax/policy, `-0.2` contract or input mutation, `0.0` clean run with no passing test, `+0.3·pass_rate`, `+1.0` full pass) to avoid zero-variance GRPO groups.
   - `fuzz_against_oracle()` differentially checks a candidate against a self-written naive reference on boundary probes (empty, singleton, negatives, duplicates, sorted, reversed); a crashing or non-literal oracle fails closed rather than reporting agreement.
   - `select_by_mbr()` performs minimum-Bayes-risk consensus selection: Tier-1 AST/contract filtering, sandboxed execution on self-generated probes, majority output-tuple vote, ties broken by shortest AST then least code, input-mutating candidates excluded, and `consensus` only for a strict majority of at least two survivors.
