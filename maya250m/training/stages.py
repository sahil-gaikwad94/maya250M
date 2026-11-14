"""Staged post-training trainers (SFT -> distillation -> DPO -> GRPO).

BUG-02 fix. The audit found that the "DPO" and "GRPO" paths produced a loss number
while the policy log-probabilities were detached, so ``optimizer.step()`` moved
nothing and the reported loss was decoration. Every stage here:

* builds the policy term from graph-attached tensors;
* computes reference/teacher terms under ``torch.no_grad()`` in a frozen copy;
* fails loudly on a non-finite loss instead of taking a silent NaN step;
* verifies *after* the stage that the parameters actually changed, and raises
  :class:`TrainingDidNotUpdate` if they did not.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

try:
    import torch
    import torch.nn.functional as F
    from ..train import MuonAdamWHybridOptimizer, build_maya_parameter_groups
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    MuonAdamWHybridOptimizer = None  # type: ignore[assignment,misc]
    build_maya_parameter_groups = None  # type: ignore[assignment]
from .data import (
    PreferencePair,
    RolloutGroup,
    TokenizedExample,
    tokenize_prompt_completion,
)
from .losses import (
    distillation_loss,
    dpo_loss,
    grpo_objective,
    masked_cross_entropy,
    sequence_logprobs,
    token_logprobs,
)

__all__ = [
    "TrainingDidNotUpdate",
    "StageConfig",
    "StageResult",
    "StagePlan",
    "frozen_reference",
    "parameter_fingerprint",
    "snapshot_parameters",
    "count_changed_parameters",
    "run_sft_stage",
    "run_distillation_stage",
    "run_dpo_stage",
    "run_grpo_stage",
    "run_stage",
    "run_stage_plan",
]


class TrainingDidNotUpdate(RuntimeError):
    """Raised when a stage reports steps but no parameter actually changed."""


# --------------------------------------------------------------------- config


@dataclass
class StageConfig:
    """Bounded, explicit configuration for one post-training stage."""

    name: str = "sft"
    lr: float = 1e-5
    muon_lr: float = 5e-4
    steps: int = 10
    batch_size: int = 1
    grad_accum: int = 1
    grad_clip: float = 1.0
    beta: float = 0.1
    clip_eps: float = 0.2
    temperature: float = 1.0
    ce_weight: float = 0.5
    max_len: int = 512
    seed: int = 0
    log_every: int = 1
    track_changes: bool = False
    freeze_reference: bool = True

    def validate(self) -> "StageConfig":
        if self.steps < 0:
            raise ValueError("steps must be >= 0")
        if self.batch_size < 1 or self.grad_accum < 1:
            raise ValueError("batch_size and grad_accum must be >= 1")
        if self.lr <= 0 or self.muon_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.max_len < 16:
            raise ValueError("max_len too small for supervised completions")
        return self


@dataclass
class StageResult:
    """What a stage actually did, with the evidence that it did it."""

    stage: str
    steps: int = 0
    losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    fingerprint_before: str = ""
    fingerprint_after: str = ""
    changed_parameters: Optional[int] = None
    examples_seen: int = 0
    notes: str = ""

    @property
    def parameters_changed(self) -> bool:
        return bool(self.fingerprint_after) and self.fingerprint_after != self.fingerprint_before

    @property
    def mean_loss(self) -> float:
        return float(sum(self.losses) / len(self.losses)) if self.losses else float("nan")

    def verify_updated(self) -> "StageResult":
        """Fail closed if the stage claimed work but changed no weights (BUG-02)."""
        if self.steps > 0 and not self.parameters_changed:
            raise TrainingDidNotUpdate(
                f"stage {self.stage!r} ran {self.steps} step(s) but no parameter changed; "
                "the policy term was probably detached"
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "steps": self.steps,
            "mean_loss": self.mean_loss,
            "first_loss": self.losses[0] if self.losses else None,
            "last_loss": self.losses[-1] if self.losses else None,
            "mean_grad_norm": (
                float(sum(self.grad_norms) / len(self.grad_norms)) if self.grad_norms else None
            ),
            "metrics": dict(self.metrics),
            "parameters_changed": self.parameters_changed,
            "changed_parameters": self.changed_parameters,
            "examples_seen": self.examples_seen,
            "notes": self.notes,
        }


# ------------------------------------------------------------------- utilities


def frozen_reference(model: torch.nn.Module) -> torch.nn.Module:
    """Deep-copied, eval-mode, no-grad reference policy for KL / DPO terms."""
    reference = copy.deepcopy(model)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    return reference


def parameter_fingerprint(model: torch.nn.Module, sample_budget: int = 4096) -> str:
    """Cheap deterministic digest of the current weights."""
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        digest.update(name.encode("utf-8"))
        flat = param.detach().float().flatten()
        if flat.numel() <= sample_budget:
            digest.update(flat.numpy().tobytes())
        else:
            stride = max(1, flat.numel() // sample_budget)
            digest.update(flat[::stride][:sample_budget].numpy().tobytes())
    return digest.hexdigest()


def snapshot_parameters(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Exact parameter snapshot for change counting (small/medium models only)."""
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def count_changed_parameters(
    model: torch.nn.Module, snapshot: Dict[str, torch.Tensor], atol: float = 0.0
) -> int:
    changed = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            before = snapshot.get(name)
            if before is None:
                continue
            if atol <= 0:
                changed += int((param.detach() != before).sum().item())
            else:
                changed += int((param.detach().sub(before).abs() > atol).sum().item())
    return changed


def build_stage_optimizer(model: torch.nn.Module, cfg: StageConfig) -> MuonAdamWHybridOptimizer:
    groups = build_maya_parameter_groups(
        model, muon_lr=cfg.muon_lr, adamw_lr=cfg.lr
    )
    return MuonAdamWHybridOptimizer(groups)


def _cycles(items: Sequence[Any], seed: int) -> Iterator[Any]:
    rng = random.Random(int(seed))
    order = list(items)
    while True:
        rng.shuffle(order)
        for item in order:
            yield item


def _summarise(metrics_accum: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        key: float(sum(values) / len(values))
        for key, values in metrics_accum.items()
        if values
    }


def _check_finite(loss: torch.Tensor, stage: str, step: int) -> None:
    if not torch.isfinite(loss.detach()).all():
        raise FloatingPointError(f"{stage} produced a non-finite loss at step {step}")


def _run_loop(
    model: torch.nn.Module,
    cfg: StageConfig,
    optimizer: torch.optim.Optimizer,
    step_fn: Callable[[int], tuple],
) -> tuple:
    """Shared optimisation loop; returns (losses, grad_norms, metrics, examples_seen)."""
    losses: List[float] = []
    grad_norms: List[float] = []
    metrics_accum: Dict[str, List[float]] = {}
    examples_seen = 0
    model.train()
    for step in range(int(cfg.steps)):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _micro in range(int(cfg.grad_accum)):
            loss, metrics, n_examples = step_fn(step)
            _check_finite(loss, cfg.name, step)
            (loss / float(cfg.grad_accum)).backward()
            step_loss += float(loss.detach().item()) / float(cfg.grad_accum)
            examples_seen += int(n_examples)
            for key, value in (metrics or {}).items():
                metrics_accum.setdefault(key, []).append(float(value))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        optimizer.step()
        losses.append(step_loss)
        grad_norms.append(float(grad_norm))
    return losses, grad_norms, _summarise(metrics_accum), examples_seen


def _clamp_len(cfg: StageConfig, model: torch.nn.Module) -> int:
    max_ctx = int(getattr(model, "max_ctx", cfg.max_len))
    return int(min(cfg.max_len, max_ctx))


def _finalize(
    stage: str,
    cfg: StageConfig,
    model: torch.nn.Module,
    fingerprint_before: str,
    losses: List[float],
    grad_norms: List[float],
    metrics: Dict[str, float],
    examples_seen: int,
    snapshot: Optional[Dict[str, torch.Tensor]],
    notes: str = "",
) -> StageResult:
    result = StageResult(
        stage=stage,
        steps=int(cfg.steps),
        losses=losses,
        grad_norms=grad_norms,
        metrics=metrics,
        fingerprint_before=fingerprint_before,
        fingerprint_after=parameter_fingerprint(model),
        examples_seen=examples_seen,
        notes=notes,
    )
    if snapshot is not None:
        result.changed_parameters = count_changed_parameters(model, snapshot)
    return result.verify_updated()


def _prepare(
    model: torch.nn.Module, cfg: StageConfig
) -> tuple:
    cfg.validate()
    model.train()
    optimizer = build_stage_optimizer(model, cfg)
    before = parameter_fingerprint(model)
    snapshot = snapshot_parameters(model) if cfg.track_changes else None
    return optimizer, before, snapshot


# ------------------------------------------------------------------- SFT stage


def run_sft_stage(
    model: torch.nn.Module,
    examples: Sequence[TokenizedExample],
    cfg: Optional[StageConfig] = None,
) -> StageResult:
    """Supervised fine-tuning on pre-tokenised examples (observations already masked)."""
    cfg = cfg or StageConfig(name="sft")
    examples = [e for e in examples if e.supervised_tokens > 0]
    if not examples:
        raise ValueError("SFT stage received no examples with supervised tokens")

    optimizer, before, snapshot = _prepare(model, cfg)
    stream = _cycles(examples, cfg.seed)
    device = next(model.parameters()).device

    def step_fn(step: int):
        batch = [next(stream) for _ in range(int(cfg.batch_size))]
        total = None
        metrics: Dict[str, float] = {}
        for example in batch:
            ids, labels = example.to_tensors(device)
            logits, _ = model(ids, labels=None)
            loss = masked_cross_entropy(logits, labels)
            total = loss if total is None else total + loss
            metrics["sft_loss"] = float(loss.detach().item())
        return total / float(len(batch)), metrics, len(batch)

    losses, grad_norms, metrics, seen = _run_loop(model, cfg, optimizer, step_fn)
    return _finalize("sft", cfg, model, before, losses, grad_norms, metrics, seen, snapshot)


# ---------------------------------------------------------- distillation stage


def run_distillation_stage(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    examples: Sequence[TokenizedExample],
    cfg: Optional[StageConfig] = None,
) -> StageResult:
    """On-policy distillation against a frozen teacher (reverse KL + optional CE)."""
    cfg = cfg or StageConfig(name="distill")
    examples = [e for e in examples if e.supervised_tokens > 0]
    if not examples:
        raise ValueError("distillation stage received no supervised examples")
    if teacher is model:
        raise ValueError("teacher must be a distinct frozen model")

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    optimizer, before, snapshot = _prepare(model, cfg)
    stream = _cycles(examples, cfg.seed)
    device = next(model.parameters()).device

    def step_fn(step: int):
        batch = [next(stream) for _ in range(int(cfg.batch_size))]
        total = None
        metrics: Dict[str, float] = {}
        for example in batch:
            ids, labels = example.to_tensors(device)
            student_logits, _ = model(ids, labels=None)
            with torch.no_grad():
                teacher_logits, _ = teacher(ids, labels=None)
            loss, m = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=cfg.temperature,
                ce_weight=cfg.ce_weight,
                reverse_kl=True,
            )
            total = loss if total is None else total + loss
            metrics = m
        return total / float(len(batch)), metrics, len(batch)

    losses, grad_norms, metrics, seen = _run_loop(model, cfg, optimizer, step_fn)
    return _finalize("distill", cfg, model, before, losses, grad_norms, metrics, seen, snapshot)


# ------------------------------------------------------------------ DPO stage


def _pair_tensors(
    tokenizer, pair: PreferencePair, max_len: int, device: torch.device
):
    chosen = tokenize_prompt_completion(tokenizer, pair.prompt, pair.chosen, max_len=max_len)
    rejected = tokenize_prompt_completion(tokenizer, pair.prompt, pair.rejected, max_len=max_len)
    return chosen.to_tensors(device), rejected.to_tensors(device)


def run_dpo_stage(
    model: torch.nn.Module,
    reference: torch.nn.Module,
    pairs: Sequence[PreferencePair],
    tokenizer,
    cfg: Optional[StageConfig] = None,
) -> StageResult:
    """DPO with a frozen reference policy and length-normalised margins."""
    cfg = cfg or StageConfig(name="dpo", beta=0.1)
    pairs = [p for p in pairs if p.chosen.strip() and p.rejected.strip() and p.chosen != p.rejected]
    if not pairs:
        raise ValueError("DPO stage received no usable preference pairs")
    if reference is model:
        raise ValueError("reference must be a frozen copy, not the training model")

    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    optimizer, before, snapshot = _prepare(model, cfg)
    stream = _cycles(pairs, cfg.seed)
    device = next(model.parameters()).device
    max_len = _clamp_len(cfg, model)
    model.eval()  # deterministic preference scoring

    def step_fn(step: int):
        batch = [next(stream) for _ in range(int(cfg.batch_size))]
        total = None
        metrics: Dict[str, float] = {}
        for pair in batch:
            (chosen_ids, chosen_labels), (rejected_ids, rejected_labels) = _pair_tensors(
                tokenizer, pair, max_len, device
            )
            policy_c, policy_c_n = sequence_logprobs(model, chosen_ids, chosen_labels)
            policy_r, policy_r_n = sequence_logprobs(model, rejected_ids, rejected_labels)
            with torch.no_grad():
                ref_c, _ = sequence_logprobs(reference, chosen_ids, chosen_labels)
                ref_r, _ = sequence_logprobs(reference, rejected_ids, rejected_labels)

            loss, m = dpo_loss(
                policy_c,
                policy_r,
                ref_c,
                ref_r,
                chosen_counts=policy_c_n,
                rejected_counts=policy_r_n,
                beta=cfg.beta,
            )
            total = loss if total is None else total + loss
            metrics = m
        return total / float(len(batch)), metrics, len(batch)

    losses, grad_norms, metrics, seen = _run_loop(model, cfg, optimizer, step_fn)
    return _finalize("dpo", cfg, model, before, losses, grad_norms, metrics, seen, snapshot)


# ----------------------------------------------------------------- GRPO stage


def run_grpo_stage(
    model: torch.nn.Module,
    reference: torch.nn.Module,
    groups: Sequence[RolloutGroup],
    tokenizer,
    cfg: Optional[StageConfig] = None,
) -> StageResult:
    """Clipped GRPO on verifiable rewards, with zero-variance groups masked out."""
    cfg = cfg or StageConfig(name="grpo", beta=0.01)
    groups = [g for g in groups if len(g.completions) >= 2]
    if not groups:
        raise ValueError("GRPO stage received no groups with at least two completions")
    if reference is model:
        raise ValueError("reference must be a frozen copy, not the training model")

    # Zero-variance groups carry no advantage signal: drop them instead of
    # training on noise (the historical `std + 1e-6` blow-up).
    learnable = [g for g in groups if bool(g.advantages()[1].any())]
    degenerate = len(groups) - len(learnable)
    if not learnable:
        return StageResult(
            stage="grpo",
            steps=0,
            metrics={"skipped_zero_variance_groups": float(degenerate)},
            notes="all groups had uniform rewards; no policy-gradient signal",
        )
    groups = learnable

    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    optimizer, before, snapshot = _prepare(model, cfg)
    stream = _cycles(groups, cfg.seed)
    device = next(model.parameters()).device
    max_len = _clamp_len(cfg, model)

    def step_fn(step: int):
        batch = [next(stream) for _ in range(int(cfg.batch_size))]
        total = None
        metrics: Dict[str, float] = {}
        n_examples = 0
        for group in batch:
            advantages, group_valid = group.advantages()
            if not bool(group_valid.any()):
                # Uniform-reward group: nothing to learn, skip it entirely.
                zero = torch.zeros((), device=device, requires_grad=True)
                total = zero if total is None else total + zero
                metrics = {
                    "grpo_loss": 0.0,
                    "skipped_zero_variance_groups": 1.0,
                    "valid_groups_frac": 0.0,
                }
                n_examples += len(group.completions)
                continue

            policy_lp, old_lp, ref_lp, masks = [], [], [], []
            for completion in group.completions:
                example = tokenize_prompt_completion(
                    tokenizer, group.prompt, completion, max_len=max_len
                )
                ids, labels = example.to_tensors(device)
                logits, _ = model(ids, labels=None)
                lp, valid = token_logprobs(logits, labels)
                policy_lp.append(lp)
                old_lp.append(lp.detach())
                with torch.no_grad():
                    ref_logits, _ = reference(ids, labels=None)
                    ref_token_lp, _ = token_logprobs(ref_logits, labels)
                ref_lp.append(ref_token_lp)
                masks.append(valid)

            max_t = max(t.shape[1] for t in policy_lp)
            pad = lambda t: F.pad(t, (0, max_t - t.shape[1]))  # noqa: E731
            valid_mask = torch.cat([pad(m) for m in masks], dim=0).bool()
            loss, m = grpo_objective(
                torch.cat([pad(t) for t in policy_lp], dim=0),
                torch.cat([pad(t) for t in old_lp], dim=0),
                torch.cat([pad(t) for t in ref_lp], dim=0),
                advantages.to(device),
                valid_mask,
                group_valid.to(device),
                clip_eps=cfg.clip_eps,
                beta=cfg.beta,
            )
            total = loss if total is None else total + loss
            metrics = m
            n_examples += len(group.completions)

        return total / float(len(batch)), metrics, n_examples

    losses, grad_norms, metrics, seen = _run_loop(model, cfg, optimizer, step_fn)
    if degenerate:
        metrics["skipped_zero_variance_groups"] = float(degenerate)
    return _finalize("grpo", cfg, model, before, losses, grad_norms, metrics, seen, snapshot)


# ------------------------------------------------------------------- pipeline


@dataclass
class StagePlan:
    """Ordered list of (stage name, kwargs) executed by :func:`run_stage_plan`."""

    stages: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, stage: str, **kwargs: Any) -> "StagePlan":
        self.stages.append({"stage": stage, **kwargs})
        return self

    def __len__(self) -> int:
        return len(self.stages)


_STAGE_FUNCS = {
    "sft": run_sft_stage,
    "distill": run_distillation_stage,
    "dpo": run_dpo_stage,
    "grpo": run_grpo_stage,
}


def run_stage(
    model: torch.nn.Module,
    stage: str,
    cfg: Optional[StageConfig] = None,
    **kwargs: Any,
) -> StageResult:
    """Dispatch one named stage; the stage name and payload are explicit by design."""
    if stage not in _STAGE_FUNCS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(_STAGE_FUNCS)}")
    stage_cfg = cfg or StageConfig(name=stage)
    stage_cfg.name = stage
    return _STAGE_FUNCS[stage](model, cfg=stage_cfg, **kwargs)


def run_stage_plan(
    model: torch.nn.Module, plan: StagePlan, record_path: Optional[str] = None
) -> List[StageResult]:
    """Execute a curriculum of stages, stopping at the first failure."""
    results: List[StageResult] = []
    for entry in plan.stages:
        payload = dict(entry)
        stage = payload.pop("stage")
        cfg = payload.pop("cfg", None)
        results.append(run_stage(model, stage, cfg=cfg, **payload))
    if record_path:
        import json
        import os

        os.makedirs(os.path.dirname(os.path.abspath(record_path)), exist_ok=True)
        with open(record_path, "w", encoding="utf-8") as handle:
            json.dump([r.to_dict() for r in results], handle, indent=2)
    return results
