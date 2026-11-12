"""Compute the exact parameter, KV-cache, and GDN state ledger for all Maya presets.

Computes exact parameter counts from tensor shapes analytically and verifies
against live ``MayaModel`` instances whenever PyTorch is installed.
"""

from __future__ import annotations

import json
from typing import Tuple
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m import CONFIGS, TORCH_AVAILABLE, get_config
from maya250m.memory.projection import memory_projection_parameters

#: Hard ceiling for the flagship preset. Auxiliary heads must fit inside it.
PARAMETER_BUDGET = 251_000_000


def compute_analytical_params(cfg: dict) -> Tuple[int, int]:
    """Compute exact parameter count and MTP parameter count from tensor dimensions."""
    d = int(cfg["d"])
    vocab = int(cfg["vocab"])
    layers = int(cfg["layers"])
    d_ff = int(cfg["d_ff"])
    n_heads = int(cfg["n_heads"])
    n_kv = int(cfg["n_kv_heads"])
    gdn_heads = int(cfg["gdn_heads"])
    conv_k = int(cfg.get("conv_kernel", 4))
    head_dim_attn = d // n_heads
    head_dim_gdn = d // gdn_heads

    schedule = cfg.get("hybrid_schedule") or ["full"] * layers
    n_gdn = sum(1 for s in schedule if s == "gdn")
    n_full = sum(1 for s in schedule if s == "full")

    # Embedding (+ untied lm_head if tie_embeddings is False)
    embed_p = vocab * d * (1 if cfg.get("tie_embeddings", True) else 2)

    # GatedDeltaNetLayer + pre-norm n1:
    # in_proj_qkvz (d * 4d) + in_proj_ba (d * 2*gdn_heads) + conv_weight (3d * conv_k)
    # + A_log (gdn_heads) + dt_bias (gdn_heads) + out_norm_w (head_dim_gdn) + out_proj (d * d) + n1 (d)
    gdn_layer_p = (
        (4 * d * d)
        + (2 * d * gdn_heads)
        + (3 * d * conv_k)
        + (2 * gdn_heads)
        + head_dim_gdn
        + (d * d)
        + d
    )

    # DifferentialGatedCausalAttention + pre-norm n1:
    # w_q (d*d) + w_k (d * n_kv * head_dim_attn) + w_v (d * n_kv * head_dim_attn) + w_o (d*d)
    # + q_norm_w/k_norm_w (2 * head_dim_attn) + w_gate (d*d)
    # + diff_attn (4 * n_heads * (head_dim_attn // 2) + head_dim_attn) + n1 (d)
    full_layer_p = (
        (d * d)
        + 2 * (d * n_kv * head_dim_attn)
        + (d * d)
        + (2 * head_dim_attn if cfg.get("qk_norm", True) else 0)
        + (d * d if cfg.get("attn_output_gate", True) else 0)
        + (
            (4 * n_heads * (head_dim_attn // 2) + head_dim_attn)
            if cfg.get("diff_attn", True)
            else 0
        )
        + d
    )

    # SwiGLU + pre-norm n2: 3 * d * d_ff + d
    mlp_layer_p = (3 * d * d_ff) + d

    # Core loop gate + core loop norm + final norm: 3 * d
    trunk_norms_p = 3 * d

    # MTPDraftHead: proj (2d * d) + norm (d) + SwiGLU (3 * d * 2d) = 8*d*d + d
    mtp_p = (8 * d * d + d) if int(cfg.get("mtp_depth", 0)) > 0 else 0

    total = (
        embed_p
        + (n_gdn * gdn_layer_p)
        + (n_full * full_layer_p)
        + (layers * mlp_layer_p)
        + trunk_norms_p
        + mtp_p
    )
    return total, mtp_p


def compute_preset_ledger(name: str) -> dict:
    cfg = get_config(name)
    total_params, mtp_params = compute_analytical_params(cfg)

    if TORCH_AVAILABLE:
        from maya250m import MayaModel
        model = MayaModel(cfg, tokenizer_hash="0" * 64)
        live_total = sum(p.numel() for p in model.parameters())
        assert live_total == total_params, f"Analytical {total_params} != live {live_total}"

    active_inference_params = total_params - mtp_params
    schedule = cfg.get("hybrid_schedule") or ["full"] * int(cfg["layers"])
    n_full = sum(1 for s in schedule if s == "full")
    n_gdn = sum(1 for s in schedule if s == "gdn")

    head_dim_attn = int(cfg["d"]) // int(cfg["n_heads"])
    head_dim_gdn = int(cfg["d"]) // int(cfg["gdn_heads"])
    max_ctx = int(cfg["max_ctx"])

    kv_bytes_per_full_layer = 2 * int(cfg["n_kv_heads"]) * head_dim_attn * max_ctx * 2
    hybrid_kv_bytes = n_full * kv_bytes_per_full_layer
    dense_counterfactual_kv_bytes = int(cfg["layers"]) * kv_bytes_per_full_layer
    gdn_state_bytes = n_gdn * int(cfg["gdn_heads"]) * (head_dim_gdn ** 2) * 4

    eff_2x = int(cfg["prelude_layers"]) + 2 * int(cfg["core_layers"]) + int(cfg["coda_layers"])
    eff_4x = int(cfg["prelude_layers"]) + 4 * int(cfg["core_layers"]) + int(cfg["coda_layers"])

    # Auxiliary (non-checkpointed) heads: the shared-trunk memory projection W_mem.
    # It lives outside MayaModel.state_dict() to keep Schema-9 checkpoints compatible,
    # so it must be accounted for explicitly here.
    w_mem_params = memory_projection_parameters(int(cfg["d"]))
    trainable_total = total_params + w_mem_params
    if name == "MAYA_250M":
        assert trainable_total <= PARAMETER_BUDGET, (
            f"MAYA_250M exceeds the {PARAMETER_BUDGET:,} parameter budget: {trainable_total:,}"
        )

    return {
        "preset": name,
        "total_parameters": total_params,
        "active_inference_parameters": active_inference_params,
        "mtp_head_parameters": mtp_params,
        "auxiliary_memory_head_parameters": w_mem_params,
        "trainable_total_with_auxiliary": trainable_total,
        "parameter_budget": PARAMETER_BUDGET,
        "within_budget": trainable_total <= PARAMETER_BUDGET,
        "physical_layers": int(cfg["layers"]),
        "effective_layers_2x_loop": eff_2x,
        "effective_layers_4x_loop": eff_4x,
        "gdn_layers": n_gdn,
        "full_diff_attn_layers": n_full,
        "gdn_head_dim": head_dim_gdn,
        "attn_head_dim": head_dim_attn,
        "max_ctx": max_ctx,
        "kv_cache_bytes_bf16": hybrid_kv_bytes,
        "dense_counterfactual_kv_bytes_bf16": dense_counterfactual_kv_bytes,
        "kv_reduction_pct": round(
            100.0 * (1.0 - hybrid_kv_bytes / max(1, dense_counterfactual_kv_bytes)), 2
        ),
        "gdn_state_bytes_fp32": gdn_state_bytes,
    }


def main() -> int:
    ledger = {}
    for preset in ("MAYA_TINY_TEST", "MAYA_50M_PILOT", "MAYA_250M", "MAYA_1B_SCALE"):
        if preset in CONFIGS:
            ledger[preset] = compute_preset_ledger(preset)

    out_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "metrics_maya250m.json")
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    print(json.dumps(ledger, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
