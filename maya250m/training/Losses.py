"""Losses for the staged post-training pipeline (SFT / distill / DPO / GRPO).

BUG-02 in the audit was that the "preference" stages computed a scalar and
reported it as a training loss while no gradient ever reached the parameters.
Every function here is written so the policy term is differentiable *and* so the
reference term is provably frozen:

* reference log-probabilities are always computed under ``torch.no_grad()`` by the
  caller and arrive here as plain tensors;
* nothing inside this module calls ``.detach()`` on a policy tensor;
* zero-variance GRPO groups are masked out instead of dividing by ~0.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

__all__ = [
    "masked_cross_entropy",
    "token_logprobs",
    "sequence_logprobs",
    "dpo_loss",
    "kl_penalty_k3",
    "grpo_objective",
    "distillation_loss",
    "group_advantages",
]

IGNORE_INDEX = -100


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Next-token CE with label masking (``labels`` aligned to ``logits``)."""
    if logits.dim() != 3:
        raise ValueError("logits must be [B, T, V]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must be [B, T] aligned with logits")
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probabilities of ``labels`` under ``logits`` (shifted by one).

    Returns ``(token_lp [B, T-1], valid_mask [B, T-1])`` where invalid positions are
    exactly ``0.0`` so they never contribute to a sum.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    targets = labels[:, 1:]
    valid = targets.ne(ignore_index)
    safe_targets = targets.clamp(min=0)
    lp = log_probs[:, :-1].gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return lp * valid.to(lp.dtype), valid


def sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    doc_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Summed log-probability of the supervised tokens in each sequence."""
    logits, _ = model(input_ids, labels=None, doc_ids=doc_ids)
    token_lp, valid = token_logprobs(logits, labels)
    return token_lp.sum(dim=-1), valid.sum(dim=-1)


def dpo_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    chosen_counts: Optional[torch.Tensor] = None,
    rejected_counts: Optional[torch.Tensor] = None,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Length-normalised DPO loss (Rafailov et al., 2023).

    ``policy_*`` tensors must still be attached to the graph; ``ref_*`` must be
    detached (the caller computes them under ``no_grad``).
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    if policy_chosen_lp.shape != policy_rejected_lp.shape:
        raise ValueError("chosen/rejected policy log-probs must have equal shape")

    if chosen_counts is not None and rejected_counts is not None:
        pc = policy_chosen_lp / chosen_counts.clamp(min=1).to(policy_chosen_lp.dtype)
        pr = policy_rejected_lp / rejected_counts.clamp(min=1).to(policy_rejected_lp.dtype)
        rc = ref_chosen_lp / chosen_counts.clamp(min=1).to(ref_chosen_lp.dtype)
        rr = ref_rejected_lp / rejected_counts.clamp(min=1).to(ref_rejected_lp.dtype)
    else:
        pc, pr, rc, rr = policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp

    policy_margin = pc - pr
    ref_margin = rc - rr
    logits = beta * (policy_margin - ref_margin)
    loss = -F.logsigmoid(logits).mean()
    with torch.no_grad():
        metrics = {
            "dpo_loss": float(loss.detach().item()),
            "reward_accuracy": float((logits > 0).float().mean().item()),
            "policy_margin": float(policy_margin.mean().item()),
            "ref_margin": float(ref_margin.mean().item()),
            "implicit_reward_gap": float(logits.mean().item()),
        }
    return loss, metrics


def kl_penalty_k3(
    policy_token_lp: torch.Tensor,
    ref_token_lp: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Unbiased (k3) per-sequence KL estimate ``KL(pi || ref)``, token-masked."""
    log_ratio = ref_token_lp - policy_token_lp
    per_token = log_ratio.exp() - log_ratio - 1.0
    per_token = per_token * valid_mask.to(per_token.dtype)
    counts = valid_mask.to(per_token.dtype).sum(dim=-1).clamp(min=1.0)
    return per_token.sum(dim=-1) / counts


def grpo_objective(
    policy_token_lp: torch.Tensor,
    old_token_lp: torch.Tensor,
    ref_token_lp: torch.Tensor,
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    group_valid: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    beta: float = 0.0,
    max_log_ratio: float = 20.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Clipped GRPO surrogate plus optional k3 KL to a frozen reference.

    ``advantages`` is one value per completion (group-normalised). Groups whose
    rewards had zero variance must arrive with ``group_valid=False`` (or all-zero
    advantages); they contribute no gradient, which is what prevents the classic
    ``std + 1e-6`` blow-up.
    """
    counts = valid_mask.to(policy_token_lp.dtype).sum(dim=-1).clamp(min=1.0)
    seq_lp = (policy_token_lp * valid_mask).sum(dim=-1) / counts
    seq_old = (old_token_lp * valid_mask).sum(dim=-1) / counts
    log_ratio = torch.clamp(seq_lp - seq_old, -max_log_ratio, max_log_ratio)
    ratio = log_ratio.exp()

    adv = advantages.to(seq_lp.dtype).detach()
    surrogate = torch.minimum(ratio * adv, torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv)

    if group_valid is not None:
        mask = group_valid.to(surrogate.dtype)
    else:
        mask = (adv.abs() > 0).to(surrogate.dtype)
    denom = mask.sum().clamp(min=1.0)
    policy_loss = -(surrogate * mask).sum() / denom

    kl = kl_penalty_k3(policy_token_lp, ref_token_lp, valid_mask)
    kl_loss = (kl * mask).sum() / denom if beta > 0 else kl.sum() * 0.0
    loss = policy_loss + float(beta) * kl_loss

    with torch.no_grad():
        clipped = (torch.abs(ratio - 1.0) > clip_eps).float()
        metrics = {
            "grpo_loss": float(loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "kl": float(kl.mean().item()) if kl.numel() else 0.0,
            "clip_fraction": float((clipped * mask).sum().item() / max(1.0, float(mask.sum().item()))),
            "valid_groups_frac": float(mask.mean().item()) if mask.numel() else 0.0,
            "mean_ratio": float(ratio.mean().item()) if ratio.numel() else 1.0,
        }
    return loss, metrics


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    ce_weight: float = 0.5,
    reverse_kl: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """On-policy distillation: reverse KL on the supervised positions (+ optional CE)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logits must have identical shape")

    student_scaled = student_logits.float() / temperature
    teacher_scaled = teacher_logits.detach().float() / temperature
    log_p_s = F.log_softmax(student_scaled, dim=-1)
    log_p_t = F.log_softmax(teacher_scaled, dim=-1)
    p_s = log_p_s.exp()

    if reverse_kl:
        per_token = (p_s * (log_p_s - log_p_t)).sum(dim=-1)
    else:
        per_token = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)

    shift_mask = labels[:, 1:].ne(IGNORE_INDEX)
    per_token = per_token[:, :-1]
    denom = shift_mask.to(per_token.dtype).sum().clamp(min=1.0)
    kl = (per_token * shift_mask.to(per_token.dtype)).sum() / denom
    loss = (temperature ** 2) * kl

    ce_value = 0.0
    if ce_weight > 0:
        ce = masked_cross_entropy(student_logits[:, :-1], labels[:, 1:])
        loss = loss + float(ce_weight) * ce
        ce_value = float(ce.detach().item())

    with torch.no_grad():
        metrics = {
            "distill_kl": float(kl.detach().item()),
            "distill_ce": ce_value,
            "distill_loss": float(loss.detach().item()),
            "teacher_student_agreement": float(
                (
                    student_scaled[:, :-1].argmax(-1)[shift_mask]
                    == teacher_scaled[:, :-1].argmax(-1)[shift_mask]
                ).float().mean().item()
            )
            if shift_mask.any()
            else 0.0,
        }
    return loss, metrics


def group_advantages(rewards, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-normalised advantages plus a validity mask for zero-variance groups."""
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(list(rewards), dtype=torch.float32)
    rewards = rewards.float().flatten()
    if rewards.numel() == 0:
        return rewards, torch.zeros(0, dtype=torch.bool)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    valid = std > eps
    if not bool(valid):
        return torch.zeros_like(rewards), torch.zeros_like(rewards, dtype=torch.bool)
    return (rewards - mean) / (std + eps), torch.ones_like(rewards, dtype=torch.bool)