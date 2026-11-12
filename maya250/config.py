"""Configuration schema and strict validation for Maya-250M.

Maya-250M is a standalone 3:1 Gated DeltaNet / Differential-GQA hybrid
architecture with a Prelude-Recurrent-Core-Coda latent reasoning loop and
Vexx 3.5 neuro-symbolic verification.

Invariants enforced at config load and model construction:
1. All '_'-prefixed metadata keys in config.json are stripped.
2. Unknown config keys raise ValueError immediately (fail-closed).
3. Prelude + Core + Coda layer counts must sum exactly to ``layers``.
4. Differential Attention requires an even number of query and KV heads
   (or pairs each head across two softmax maps).
5. Gated DeltaNet head dimension ``d // gdn_heads`` must be even, and
   ``beta_scale`` must be in (0, 2.0] to support negative-eigenvalue state
   transitions (Grazzi et al., ICLR 2025; DeltaProduct, 2025).
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _RAW_CONFIGS: Dict[str, object] = json.load(_f)

CONFIGS: Dict[str, dict] = {
    k: v
    for k, v in _RAW_CONFIGS.items()
    if not k.startswith("_") and isinstance(v, dict)
}

VALID_CONFIG_KEYS = {
    "vocab",
    "d",
    "n_heads",
    "n_kv_heads",
    "gdn_heads",
    "d_ff",
    "layers",
    "max_ctx",
    "rope_theta",
    "slow_rope_theta",
    "norm_eps",
    "tie_embeddings",
    "attn_impl",
    "hybrid_schedule",
    "sliding_window",
    "qk_norm",
    "attn_output_gate",
    "diff_attn",
    "zero_centered_norm",
    "beta_scale",
    "gdn_chunk_size",
    "prelude_layers",
    "core_layers",
    "coda_layers",
    "recurrent_loops",
    "mtp_depth",
    "conv_kernel",
}

_DEFAULTS = {
    "attn_impl": "hybrid_gdn",
    "hybrid_schedule": None,
    "sliding_window": None,
    "norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "slow_rope_theta": 10000000.0,
    "tie_embeddings": True,
    "qk_norm": True,
    "attn_output_gate": True,
    "diff_attn": True,
    "zero_centered_norm": True,
    "beta_scale": 2.0,
    "gdn_chunk_size": 64,
    "recurrent_loops": 1,
    "mtp_depth": 3,
    "conv_kernel": 4,
}


def validate_config(c: dict, name: str = "config") -> dict:
    """Validate a Maya-250M config dict, filling optional defaults."""
    unknown = set(c) - VALID_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config fields in '{name}': {sorted(unknown)}")
    for key, value in _DEFAULTS.items():
        c.setdefault(key, value)

    layers = int(c["layers"])
    c.setdefault("gdn_heads", int(c["n_heads"]))
    c.setdefault("prelude_layers", layers // 4)
    c.setdefault("core_layers", layers - 2 * (layers // 4))
    c.setdefault("coda_layers", layers // 4)

    if (
        layers < 1
        or int(c["d"]) < 1
        or int(c["vocab"]) < 1
        or int(c["n_heads"]) < 1
        or int(c["n_kv_heads"]) < 1
        or int(c["gdn_heads"]) < 1
        or int(c["d_ff"]) < 1
        or int(c["max_ctx"]) < 1
        or int(c["conv_kernel"]) < 1
        or int(c["gdn_chunk_size"]) < 1
        or int(c["recurrent_loops"]) < 1
        or float(c["rope_theta"]) <= 0
        or float(c["slow_rope_theta"]) <= 0
        or float(c["norm_eps"]) <= 0
        or not (0.0 < float(c["beta_scale"]) <= 2.0)
    ):
        raise ValueError(
            f"'{name}': dimensions, context, kernel, chunk_size, loops, eps must be positive "
            "and beta_scale must be in (0, 2.0]"
        )

    if int(c["d"]) % int(c["n_heads"]) != 0:
        raise ValueError(f"'{name}': d ({c['d']}) must be divisible by n_heads ({c['n_heads']})")
    if int(c["d"]) % int(c["gdn_heads"]) != 0:
        raise ValueError(f"'{name}': d ({c['d']}) must be divisible by gdn_heads ({c['gdn_heads']})")

    head_dim = int(c["d"]) // int(c["n_heads"])
    if head_dim % 2 != 0:
        raise ValueError(f"'{name}': attention head_dim must be even for RoPE, got {head_dim}")
    if bool(c["diff_attn"]) and head_dim % 4 != 0:
        raise ValueError(
            f"'{name}': Differential Attention requires head_dim divisible by 4, got {head_dim}"
        )
    if int(c["n_heads"]) % int(c["n_kv_heads"]) != 0:
        raise ValueError(f"'{name}': n_heads must be a multiple of n_kv_heads")
    if int(c["mtp_depth"]) not in (0, 1, 2, 3):
        raise ValueError(f"'{name}': mtp_depth must be one of 0, 1, 2, 3, got {c['mtp_depth']}")

    p_l = int(c["prelude_layers"])
    core_l = int(c["core_layers"])
    coda_l = int(c["coda_layers"])
    if p_l < 0 or core_l < 1 or coda_l < 0 or (p_l + core_l + coda_l) != layers:
        raise ValueError(
            f"'{name}': prelude_layers ({p_l}) + core_layers ({core_l}) + coda_layers ({coda_l}) "
            f"must equal layers ({layers})"
        )

    schedule = c.get("hybrid_schedule")
    if schedule is not None:
        if len(schedule) != layers:
            raise ValueError(
                f"'{name}': hybrid_schedule length {len(schedule)} != layers {layers}"
            )
        bad = set(schedule) - {"gdn", "full"}
        if bad:
            raise ValueError(f"'{name}': unknown mixer types {sorted(bad)}")
    if c["attn_impl"] == "hybrid_gdn" and schedule is None:
        raise ValueError(f"'{name}': attn_impl='hybrid_gdn' requires a hybrid_schedule")
    if c["attn_impl"] == "dense" and schedule is not None:
        raise ValueError(f"'{name}': attn_impl='dense' cannot declare a hybrid_schedule")
    if c["attn_impl"] not in {"dense", "hybrid_gdn"}:
        raise ValueError(f"'{name}': unsupported attn_impl {c['attn_impl']!r}")

    window = c.get("sliding_window")
    if window is not None and not (0 < int(window) <= int(c["max_ctx"])):
        raise ValueError(f"'{name}': sliding_window must be in (0, max_ctx], got {window}")
    return c


def get_config(name: str) -> dict:
    """Return a validated deep copy of a named Maya preset."""
    if name not in CONFIGS:
        raise KeyError(f"Unknown Maya config preset '{name}'. Valid presets: {sorted(CONFIGS)}")
    c = copy.deepcopy({k: v for k, v in CONFIGS[name].items() if not k.startswith("_")})
    return validate_config(c, name)
