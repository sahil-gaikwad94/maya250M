# 04 — Google TPU Research Cloud (TRC) Application & 50B–100B Scaling Roadmap

## 1. Positioning Maya-250M for Google TRC Approval

The [Google TPU Research Cloud (TRC)](https://sites.research.google/trc/faq/) grants free access to Cloud TPU v4, v5e, and v6e pods to researchers who share open-source code, reproducible benchmarks, and architectural feedback on XLA/TPU workloads ([Google Open Source Blog](https://opensource.googleblog.com/2025/12/)).

Maya-250M is specifically co-designed for Cloud TPU MXU hardware:
- **Exact 128-Tile Alignment**: Every matrix dimension (`d = 768 = 6 x 128`, `d_ff = 1792 = 14 x 128`, `head_dim_gdn = 128`, `vocab = 32,000 = 250 x 128`, `chunk_size = 64`) aligns with Cloud TPU `$128 \times 128$` systolic arrays with zero padding waste.
- **Matrix-Multiply-Native Gated DeltaNet**: Our WY Chunkwise Parallel solver (`_chunk_parallel_scan`) expresses recurrent state updates as dense batched `matmul` and `solve_triangular` ops that compile cleanly under `torch_xla` and JAX/Pallas.

## 2. Scaling from Maya-250M to Maya-50B-A4B / Maya-100B-A6B MoE

Under [Maximal Update Parametrization ($\mu\text{P}$, Yang et al.)](https://arxiv.org/abs/2203.03466) and the [Muon optimizer](https://kellerjordan.github.io/posts/muon/), learning rates and initialization scales tuned on `MAYA_250M` (`d = 768`) transfer zero-shot as width scales to `MAYA_1B_SCALE` (`d = 1536`) and **Maya-50B-A4B** (`d = 2048, L = 48`, 128 fine-grained SwiGLU experts with top-8 routing, 3.8B active parameters per token). The entire Vexx 3.5 six-tier neuro-symbolic engine, 32,000-vocab protocol tokenizer, and RLVR pipeline remain identical across scales.
