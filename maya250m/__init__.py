"""Maya-250M + Vexx 3.5 Standalone Package.

Exports config, tokenizer, data-packing, and Vexx 3.5 neuro-symbolic modules
unconditionally, and exports all PyTorch model/training/checkpoint modules
when PyTorch is installed in the environment.
"""

from __future__ import annotations

from .config import CONFIGS, VALID_CONFIG_KEYS, get_config, validate_config
from .data import (
    MemmapShardDataset,
    StreamPackedDataset,
    collate_stream_packed,
    deduplicate_and_split,
    write_memmap_shard,
)
from .tokenizer import (
    BOS,
    EOS,
    EXP_END,
    EXP_START,
    LEGAL_STOP,
    NEVER_EMIT,
    PAD,
    PROTOCOL_MARKERS,
    SPECIAL_TOKENS,
    THINK_END,
    THINK_START,
    UNK,
    BPETokenizer,
    token_category,
)
from .vexx import (
    Vexx35Controller,
    Vexx3Controller,
    clone_cache_generic,
    safe_sympy,
    truncate_cache_at_token,
)

try:
    from .checkpoint import (
        VERSION as CHECKPOINT_SCHEMA_VERSION,
        capture_rng_state,
        load_checkpoint_maya,
        restore_rng_state,
        save_checkpoint_maya,
    )
    from .generate import (
        JSON,
        PLAIN,
        PYTHON,
        THINK,
        GenerationResult,
        generate_maya,
        sample_next,
        top_p_filter,
    )
    from .cache import (
        CACHE_SCHEMA_VERSION,
        CacheIdentity,
        CacheIdentityError,
        CacheRewindError,
        CacheSnapshotStore,
        cache_memory_ledger,
        cache_memory_table,
        rewind_cache,
        validate_bundle,
    )
    from .model import (
        DifferentialGatedCausalAttention,
        GatedDeltaNetLayer,
        MTPDraftHead,
        MayaBlock,
        MayaModel,
        SwiGLU,
        build_maya_model,
        clone_hybrid_cache,
        create_empty_hybrid_cache_v2,
        create_empty_hybrid_cache_v3,
        hash_config,
        rms_norm_fp32,
        truncate_hybrid_cache_at_token,
    )
    from .adaptive import (
        SolveConfig,
        SolveResult,
        generate_adaptive,
        select_loops,
        solve_task,
    )
    from .train import (
        ForwardMTPCurriculum,
        LinearWarmupCosineDecay,
        MuonAdamWHybridOptimizer,
        build_adamw_parameter_groups,
        build_maya_parameter_groups,
        compute_on_policy_reverse_kl_loss,
        evaluate_maya,
        train_maya,
        zeropower_via_newtonschulz5,
    )
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
