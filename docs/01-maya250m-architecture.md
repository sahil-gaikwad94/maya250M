# 01 — Maya-250M Architecture Specification & Falsifiers

## 1. Overview

`MAYA_250M` is a **250,007,328-parameter** (**245.29M active per token** during standard inference) hybrid recurrent/attention language model designed for high-density coding, mathematical reasoning, and scientific experimentation from zero.

It organizes **32 physical layers** (`d = 768`, `d_ff = 1792`, `vocab = 32,000`) into 8 repeating four-layer macro-blocks (`[gdn, gdn, gdn, full]`):

- **Prelude (Layers 0–7, 2 macro-blocks)**: Encodes local syntax, token order, and initial key-value bindings.
- **Recurrent Reasoning Core (Layers 8–23, 4 macro-blocks)**: Iterates `R_loop` times (`default = 2`, adaptive up to `4`) in continuous latent space via a zero-initialized residual input-injection gate (`core_loop_gate`), expanding the 32 physical layers into **48 to 80 effective layers** of computation ([Geiping et al., NeurIPS 2025](https://arxiv.org/abs/2502.05171); [Coconut, Hao et al., 2024](https://arxiv.org/abs/2412.06769)).
- **Coda + MTP-3 Draft Trunk (Layers 24–31, 2 macro-blocks + `MTPDraftHead`)**: Projects the refined latent state into vocabulary logits and proposes up to 3 speculative future tokens (`t+2, t+3, t+4`) verified in parallel by Vexx 3.5.

## 2. Five Major Upgrades in Maya-250M

### 2.1 WY-Representation Chunkwise Parallel Gated DeltaNet (`chunk_size = 64`)

A token-by-token Python loop over sequence length `n` launches `O(n)` tiny kernels per layer, starving GPUs and stalling XLA compilation on TPUs. `GatedDeltaNetLayer._chunk_parallel_scan` partitions the sequence into chunks of `C = 64` tokens and resolves the intra-chunk Householder delta rule via a unit lower-triangular linear system:

```
(I + M) U = R,   where M_{i,j} = beta_i * Gamma_{i,j} * (k_i . k_j) for i > j
```

solved in a single batched `torch.linalg.solve_triangular(..., unitriangular=True)` call. By masking `Gamma_{i,j}` with `doc_ids[i] == doc_ids[j]`, the chunkwise solver matches the sequential recurrence to `< 1e-5` even when multiple packed documents start and end inside the same 64-token chunk.

### 2.2 Expanded Per-Head State Capacity (`head_dim = 128`) & Negative Eigenvalues (`beta_scale = 2.0`)

- **4x State Matrix Area per Head**: Setting `gdn_heads = 6` at `d = 768` gives `head_dim = 128`, expanding each head's FP32 state matrix from `64 x 64` to `128 x 128` (`16,384` floats per head) without adding projection parameters, doubling the associative recall cliff `m*` ([Gated Delta Networks, ICLR 2025](https://arxiv.org/abs/2412.06464)).
- **Negative Eigenvalues**: Scaling the write gate to `beta_t = 2 * sigmoid(b_t) in (0, 2)` allows state-transition eigenvalues in `[-1, 1]`, unlocking parity, permutation composition (`S_5`), and variable-mutation tracking in code ([Grazzi et al., ICLR 2025](https://arxiv.org/abs/2411.12537); [DeltaProduct, 2025](https://arxiv.org/abs/2502.10297)).

### 2.3 Differential Gated GQA + Fractional Slow-RoPE on Global Layers

Every 4th layer (`full`) uses [Differential Attention (Ye et al., ICLR 2025)](https://arxiv.org/abs/2410.05258), subtracting two softmax maps (`A_1 - lambda * A_2`) to suppress irrelevant context noise, while alternating `rope_theta = 500,000` and `slow_rope_theta = 10,000,000` across global layers so half of the global retrieval heads perform position-agnostic symbol lookup.
