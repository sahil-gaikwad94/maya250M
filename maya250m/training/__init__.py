"""Staged post-training: SFT, on-policy distillation, DPO and GRPO.

BUG-02 lives here: every trainer applies a real optimizer update and verifies it,
raising :class:`TrainingDidNotUpdate` otherwise.
"""

from __future__ import annotations

from .data import (
    PreferenceDataset,
    PreferencePair,
    RolloutGroup,
    TokenizedExample,
    build_rollout_groups,
    examples_from_trajectories,
    pad_examples,
    pairs_from_vexx_rows,
    tokenize_prompt_completion,
    tokenize_sft_text,
)
from .losses import (
    distillation_loss,
    dpo_loss,
    group_advantages,
    grpo_objective,
    kl_penalty_k3,
    masked_cross_entropy,
    sequence_logprobs,
    token_logprobs,
)
from .stages import (
    StageConfig,
    StagePlan,
    StageResult,
    TrainingDidNotUpdate,
    count_changed_parameters,
    frozen_reference,
    parameter_fingerprint,
    run_distillation_stage,
    run_dpo_stage,
    run_grpo_stage,
    run_sft_stage,
    run_stage,
    run_stage_plan,
    snapshot_parameters,
)

__all__ = [
    "PreferenceDataset",
    "PreferencePair",
    "RolloutGroup",
    "TokenizedExample",
    "build_rollout_groups",
    "examples_from_trajectories",
    "pad_examples",
    "pairs_from_vexx_rows",
    "tokenize_prompt_completion",
    "tokenize_sft_text",
    "distillation_loss",
    "dpo_loss",
    "group_advantages",
    "grpo_objective",
    "kl_penalty_k3",
    "masked_cross_entropy",
    "sequence_logprobs",
    "token_logprobs",
    "StageConfig",
    "StagePlan",
    "StageResult",
    "TrainingDidNotUpdate",
    "count_changed_parameters",
    "frozen_reference",
    "parameter_fingerprint",
    "run_distillation_stage",
    "run_dpo_stage",
    "run_grpo_stage",
    "run_sft_stage",
    "run_stage",
    "run_stage_plan",
    "snapshot_parameters",
]
