"""Maya-250M training engine: Muon + AdamW hybrid optimizer, Forward MTP Curriculum,
exact mid-epoch resume, and Kaggle Free-Quota wall-clock watchdog.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .checkpoint import load_checkpoint_maya, save_checkpoint_maya
from .model import MayaModel


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """5-step quintic Newton-Schulz iteration to orthogonalize 2D gradient/momentum matrix G.

    Following Keller Jordan's Muon specification:
        a, b, c = 3.4445, -4.7750, 2.0315
    Computes an approximate semi-orthogonal polar factor U V^T in FP32/BF16.
    """
    if G.ndim != 2:
        raise ValueError(f"zeropower_via_newtonschulz5 requires a 2D tensor, got {G.ndim}D")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.transpose(0, 1)
    norm = X.norm() + eps
    X = X / norm
    for _ in range(steps):
        A = X @ X.transpose(0, 1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(0, 1)
    return X.to(G.dtype)


def build_maya_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.1,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    use_muon: bool = True,
) -> List[dict]:
    """Partition MayaModel parameters into non-overlapping Muon and AdamW groups.

    Rules:
    - ``A_log``, ``dt_bias``, Differential Attention ``lambda_*``, ``core_loop_gate``,
      biases, and 1D ``RMSNorm`` vectors receive ``weight_decay = 0.0`` under AdamW.
    - Token embeddings (``embed.weight``), ``lm_head.weight``, and 2D ``conv_weight``
      receive ``weight_decay`` under AdamW.
    - All internal 2D linear projection matrices inside ``blocks.*`` and ``mtp_head.*``
      use Muon (when ``use_muon=True``) or decayed AdamW (when ``use_muon=False``).
    """
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

    muon_params = []
    adamw_decay = []
    adamw_no_decay = []
    seen = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        ident = id(param)
        if ident in seen:
            raise RuntimeError(f"Duplicate trainable parameter encountered: {name}")
        seen.add(ident)

        is_no_decay_special = (
            param.ndim <= 1
            or name.endswith(".bias")
            or name.endswith(".A_log")
            or name.endswith(".dt_bias")
            or ".lambda_" in name
            or "core_loop_" in name
        )
        is_embed_or_conv = (
            name == "embed.weight"
            or name == "lm_head.weight"
            or name.endswith(".conv_weight")
        )

        if is_no_decay_special:
            adamw_no_decay.append(param)
        elif use_muon and param.ndim == 2 and not is_embed_or_conv:
            muon_params.append(param)
        else:
            adamw_decay.append(param)

    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = {id(p) for p in muon_params + adamw_decay + adamw_no_decay}
    if expected != actual:
        raise RuntimeError("Parameter group partitioning mismatch")

    groups = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "lr": float(muon_lr),
                "weight_decay": float(weight_decay),
                "group_name": "muon_2d",
                "opt_type": "muon",
                "momentum": 0.95,
                "ns_steps": 5,
            }
        )
    if adamw_decay:
        groups.append(
            {
                "params": adamw_decay,
                "lr": float(adamw_lr),
                "weight_decay": float(weight_decay),
                "group_name": "decay",
                "opt_type": "adamw",
                "betas": (0.9, 0.95),
                "eps": 1e-8,
            }
        )
    if adamw_no_decay:
        groups.append(
            {
                "params": adamw_no_decay,
                "lr": float(adamw_lr),
                "weight_decay": 0.0,
                "group_name": "no_decay",
                "opt_type": "adamw",
                "betas": (0.9, 0.95),
                "eps": 1e-8,
            }
        )
    return groups


def build_adamw_parameter_groups(model: torch.nn.Module, weight_decay: float = 0.1) -> List[dict]:
    """Build pure AdamW parameter groups (decay vs no_decay)."""
    return build_maya_parameter_groups(
        model, weight_decay=weight_decay, adamw_lr=3e-4, use_muon=False
    )


class MuonAdamWHybridOptimizer(torch.optim.Optimizer):
    """Unified PyTorch Optimizer executing Muon on 2D trunk matrices and AdamW on 1D/embeddings.

    All optimizer states are standard ``torch.Tensor`` and Python ``int`` objects,
    so checkpoints saved with ``optimizer.state_dict()`` load cleanly under
    ``torch.load(..., weights_only=True)``.
    """

    def __init__(self, param_groups: List[dict]):
        defaults = dict(
            lr=3e-4,
            weight_decay=0.1,
            momentum=0.95,
            ns_steps=5,
            betas=(0.9, 0.95),
            eps=1e-8,
            opt_type="adamw",
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group.get("weight_decay", 0.0))
            opt_type = group.get("opt_type", "adamw")

            if opt_type == "muon":
                mom = float(group.get("momentum", 0.95))
                ns_steps = int(group.get("ns_steps", 5))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["step"] += 1
                    buf = state["momentum_buffer"]
                    buf.mul_(mom).add_(g)
                    # Nesterov lookahead before Newton-Schulz orthogonalization
                    g_nest = g.add(buf, alpha=mom)
                    ortho = zeropower_via_newtonschulz5(g_nest, steps=ns_steps)
                    scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
                    if wd > 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(ortho, alpha=-lr * scale * 0.2)
            else:
                beta1, beta2 = group.get("betas", (0.9, 0.95))
                eps = float(group.get("eps", 1e-8))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    if wd > 0.0:
                        p.mul_(1.0 - lr * wd)
                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    bias_c1 = 1.0 - (beta1 ** t)
                    bias_c2 = 1.0 - (beta2 ** t)
                    step_size = lr / bias_c1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class LinearWarmupCosineDecay:
    """Linear warmup followed by cosine decay, initialized at update 1."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_multiplier: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr_multiplier = float(min_lr_multiplier)
        if not 0.0 <= self.min_lr_multiplier <= 1.0:
            raise ValueError("min_lr_multiplier must be in [0, 1]")
        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.step_num = 0
        self._apply_for_update(1)

    def _multiplier(self, update_number: int) -> float:
        update_number = max(1, int(update_number))
        if self.warmup_steps > 0 and update_number <= self.warmup_steps:
            return update_number / float(self.warmup_steps)
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = (update_number - self.warmup_steps) / float(decay_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_multiplier + (1.0 - self.min_lr_multiplier) * cosine

    def _apply_for_update(self, update_number: int) -> float:
        mult = self._multiplier(update_number)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * mult
        return mult

    def step(self) -> float:
        self.step_num += 1
        return self._apply_for_update(self.step_num + 1)

    def state_dict(self) -> dict:
        return {
            "step_num": self.step_num,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_multiplier": self.min_lr_multiplier,
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(self, state: dict) -> None:
        self.step_num = int(state.get("step_num", 0))
        if state.get("base_lrs") is not None:
            self.base_lrs = [float(x) for x in state["base_lrs"]]
        self._apply_for_update(self.step_num + 1)


class ForwardMTPCurriculum:
    """Forward Multi-Token Prediction curriculum for small language models (ACL 2025).

    Ramps active MTP depth ``1 -> 2 -> 3`` as training progresses so early updates
    focus on next-token representations before optimizing multi-horizon planning.
    """

    def __init__(self, max_depth: int = 3, total_steps: int = 100):
        self.max_depth = max(0, min(3, int(max_depth)))
        self.total_steps = max(1, int(total_steps))

    def active_depth(self, step: int) -> int:
        if self.max_depth <= 1:
            return self.max_depth
        frac = float(step) / float(self.total_steps)
        if frac < 0.25:
            return 1
        if frac < 0.60:
            return min(2, self.max_depth)
        return self.max_depth


def compute_on_policy_reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mode-seeking Reverse KL ``T^2 * KL(pi_student || pi_teacher)`` in native PyTorch."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    s_scaled = student_logits.float() / float(temperature)
    t_scaled = teacher_logits.detach().float() / float(temperature)
    log_p_s = F.log_softmax(s_scaled, dim=-1)
    log_p_t = F.log_softmax(t_scaled, dim=-1)
    p_s = log_p_s.exp()
    token_kl = (p_s * (log_p_s - log_p_t)).sum(dim=-1)
    if mask is not None:
        valid = mask.to(token_kl.dtype)
        denom = valid.sum().clamp(min=1.0)
        mean_kl = (token_kl * valid).sum() / denom
    else:
        mean_kl = token_kl.mean()
    return (float(temperature) ** 2) * mean_kl


def _dataset_fingerprint(dataset) -> str:
    manifest = getattr(dataset, "dataset_manifest", None)
    if isinstance(manifest, dict):
        blob = json.dumps(manifest, sort_keys=True) if "json" in globals() else str(sorted(manifest.items()))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    if hasattr(dataset, "tensors"):
        digest = hashlib.sha256()
        for t in dataset.tensors:
            digest.update(t.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    raise ValueError(
        "Dataset must expose a content-addressed dataset_manifest or tensors for exact resume"
    )


def evaluate_maya(
    model: MayaModel,
    val_loader: DataLoader,
    mtp_weight: float = 0.2,
    intermediate_supervision_weight: float = 0.0,
    device: Optional[torch.device] = None,
) -> dict:
    """Evaluate next-token CE, perplexity, MTP loss, and intermediate loss on held-out split."""
    was_training = model.training
    model.eval()
    dev = device or next(model.parameters()).device

    total_lm_loss_sum = 0.0
    total_lm_targets = 0
    horizon_sums = [0.0, 0.0, 0.0]
    horizon_counts = [0, 0, 0]
    inter_sum = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels, doc_ids = [t.to(dev) for t in batch]
            _logits, _loss, comp = model(
                input_ids,
                labels=labels,
                doc_ids=doc_ids,
                mtp_weight=mtp_weight,
                intermediate_supervision_weight=intermediate_supervision_weight,
                return_components=True,
            )
            n_lm = int(comp["lm_valid_targets"])
            if n_lm > 0:
                total_lm_loss_sum += float(comp["lm_loss"].item()) * n_lm
                total_lm_targets += n_lm
                inter_sum += float(comp["intermediate_loss"].item()) * n_lm

            # BUG-07 fix: accumulate each horizon's own CE, never the pooled mtp_loss.
            h_counts = comp.get("mtp_horizon_counts", [])
            h_losses = comp.get("mtp_horizon_losses", [])
            for idx, cnt in enumerate(h_counts[:3]):
                if cnt > 0 and idx < len(h_losses) and h_losses[idx] is not None:
                    horizon_sums[idx] += float(h_losses[idx].item()) * cnt
                    horizon_counts[idx] += cnt

    if was_training:
        model.train()

    mean_lm_ce = total_lm_loss_sum / max(1, total_lm_targets)
    active_h_means = [
        horizon_sums[i] / horizon_counts[i] for i in range(3) if horizon_counts[i] > 0
    ]
    mean_mtp = sum(active_h_means) / len(active_h_means) if active_h_means else 0.0
    mean_inter = inter_sum / max(1, total_lm_targets)
    total_obj = mean_lm_ce + (mtp_weight * mean_mtp) + (intermediate_supervision_weight * mean_inter)
    return {
        "val_loss": mean_lm_ce,
        "val_total_objective": total_obj,
        "val_ppl": math.exp(min(20.0, mean_lm_ce)),
        "val_mtp_loss": mean_mtp,
        "val_intermediate_loss": mean_inter,
        "val_tokens": total_lm_targets,
    }


def train_maya(
    model: MayaModel,
    train_dataset,
    val_dataset=None,
    optimizer=None,
    scheduler=None,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    max_steps: int = 20,
    mtp_weight: float = 0.2,
    intermediate_supervision_weight: float = 0.05,
    use_mtp_curriculum: bool = False,
    max_grad_norm: float = 1.0,
    checkpoint_path: Optional[str] = None,
    best_checkpoint_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    tokenizer=None,
    seed: int = 42,
    max_wall_hours: Optional[float] = None,
    device: Optional[torch.device] = None,
    checkpoint_role: str = "unspecified",
) -> dict:
    """Deterministic, bitwise-resumable Maya-250M training loop with Kaggle wall-clock guard."""
    if len(train_dataset) == 0:
        raise ValueError("train_dataset cannot be empty")
    dev = device or next(model.parameters()).device
    model.to(dev)

    if optimizer is None:
        groups = build_maya_parameter_groups(model, weight_decay=0.1, use_muon=True)
        optimizer = MuonAdamWHybridOptimizer(groups)
    if scheduler is None:
        scheduler = LinearWarmupCosineDecay(
            optimizer, warmup_steps=max(1, max_steps // 10), total_steps=max_steps
        )

    dataset_fp = _dataset_fingerprint(train_dataset)
    collate_fn = getattr(train_dataset, "collate_fn", None)
    if collate_fn is None:
        from .data import collate_stream_packed
        collate_fn = collate_stream_packed

    epoch = 0
    cursor = 0
    step = 0
    best_val_loss = float("inf")
    history: List[dict] = []
    permutation: Optional[List[int]] = None

    if resume_from is not None and os.path.exists(resume_from):
        saved_state = load_checkpoint_maya(
            resume_from,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,
        )
        if saved_state.get("dataset_fingerprint") != dataset_fp:
            raise ValueError("Dataset fingerprint mismatch on resume (fail-closed)")
        if int(saved_state.get("batch_size", batch_size)) != int(batch_size):
            raise ValueError("batch_size mismatch on resume")
        epoch = int(saved_state.get("epoch", 0))
        cursor = int(saved_state.get("next_batch_cursor", 0))
        step = int(saved_state.get("step", 0))
        best_val_loss = float(saved_state.get("best_val_loss", float("inf")))
        history = list(saved_state.get("history", []))
        permutation = list(saved_state["permutation"]) if saved_state.get("permutation") else None

    mtp_curriculum = ForwardMTPCurriculum(max_depth=model.mtp_depth, total_steps=max_steps)
    start_wall = time.monotonic()
    stopped_by_wall_clock = False

    model.train()
    while step < max_steps:
        n_samples = len(train_dataset)
        if permutation is None or cursor >= n_samples:
            if permutation is not None and cursor >= n_samples:
                epoch += 1
                cursor = 0
            rng_ep = random.Random(seed + epoch)
            permutation = list(range(n_samples))
            rng_ep.shuffle(permutation)

        rem_indices = permutation[cursor:]
        sub_ds = Subset(train_dataset, rem_indices)
        loader = DataLoader(
            sub_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
        )
        # Stream one gradient-accumulation group at a time. Materialising
        # ``list(loader)`` held every remaining batch of the epoch in host RAM, which
        # made peak memory O(epoch) instead of O(grad_accum_steps).
        loader_iter = iter(loader)

        while step < max_steps:
            group_batches = list(itertools.islice(loader_iter, int(grad_accum_steps)))
            if not group_batches:
                break

            active_mtp_d = (
                mtp_curriculum.active_depth(step + 1)
                if use_mtp_curriculum
                else model.mtp_depth
            )

            # Pre-count exact valid targets across the entire accumulation group
            group_lm_targets = 0
            group_mtp_counts = [0, 0, 0]
            for g_ids, g_lbls, g_docs in group_batches:
                n_seq = g_ids.shape[1]
                group_lm_targets += int((g_lbls[:, 1:] != -100).sum().item())
                if active_mtp_d > 0 and model.mtp_head is not None and n_seq >= 3:
                    for k_hor in range(1, active_mtp_d + 1):
                        if n_seq <= k_hor + 1:
                            continue
                        t_k = g_lbls[:, k_hor + 1 : n_seq].clone()
                        if g_docs is not None:
                            base_d = g_docs[:, : n_seq - k_hor - 1]
                            same_d = base_d >= 0
                            for off in range(1, k_hor + 2):
                                same_d = same_d & (
                                    g_docs[:, off : n_seq - k_hor - 1 + off] == base_d
                                )
                            t_k = t_k.masked_fill(~same_d, -100)
                        group_mtp_counts[k_hor - 1] += int((t_k != -100).sum().item())

            active_horizons = sum(1 for c in group_mtp_counts[:active_mtp_d] if c > 0)
            optimizer.zero_grad(set_to_none=True)

            step_lm_loss_val = 0.0
            step_mtp_loss_val = 0.0
            step_inter_loss_val = 0.0

            for m_ids, m_lbls, m_docs in group_batches:
                m_ids = m_ids.to(dev)
                m_lbls = m_lbls.to(dev)
                m_docs = m_docs.to(dev)

                _logits, _raw_loss, comp = model(
                    m_ids,
                    labels=m_lbls,
                    doc_ids=m_docs,
                    mtp_weight=mtp_weight,
                    mtp_active_depth=active_mtp_d,
                    intermediate_supervision_weight=intermediate_supervision_weight,
                    return_components=True,
                )

                mb_lm_valid = int(comp["lm_valid_targets"])
                mb_obj = _logits.sum() * 0.0

                if mb_lm_valid > 0 and group_lm_targets > 0:
                    lm_scale = float(mb_lm_valid) / float(group_lm_targets)
                    mb_obj = mb_obj + comp["lm_loss"] * lm_scale
                    step_lm_loss_val += float(comp["lm_loss"].item()) * lm_scale

                    if intermediate_supervision_weight > 0.0:
                        mb_obj = (
                            mb_obj
                            + float(intermediate_supervision_weight)
                            * comp["intermediate_loss"]
                            * lm_scale
                        )
                        step_inter_loss_val += float(comp["intermediate_loss"].item()) * lm_scale

                mb_mtp_counts = comp.get("mtp_horizon_counts", [])
                mb_mtp_losses = comp.get("mtp_horizon_losses", [])
                if active_horizons > 0 and mtp_weight > 0.0 and sum(mb_mtp_counts) > 0:
                    # BUG-07 fix: normalize each horizon k by its own group-wide valid-token
                    # count, then average equally across the active horizons. This makes
                    # microbatch gradient accumulation exactly equal (to < 1e-6) to a single
                    # concatenated batch even when horizons have unequal valid counts.
                    per_hzn = float(mtp_weight) / float(active_horizons)
                    for k_idx in range(
                        min(active_mtp_d, len(mb_mtp_losses), len(group_mtp_counts))
                    ):
                        cnt_k = int(mb_mtp_counts[k_idx]) if k_idx < len(mb_mtp_counts) else 0
                        if cnt_k <= 0 or group_mtp_counts[k_idx] <= 0:
                            continue
                        loss_k = mb_mtp_losses[k_idx]
                        if loss_k is None:
                            continue
                        frac_k = float(cnt_k) / float(group_mtp_counts[k_idx])
                        mb_obj = mb_obj + per_hzn * loss_k * frac_k
                        step_mtp_loss_val += float(loss_k.item()) * frac_k * per_hzn

                mb_obj.backward()

            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            # Advance the resume cursor only after the whole group is consumed, so a
            # mid-epoch checkpoint resumes at the exact next unseen sample.
            cursor += sum(int(batch[0].shape[0]) for batch in group_batches)

            step += 1
            model.optimizer_updates += 1
            model.tokens_seen += int(group_lm_targets)
            model.weights_origin = "trained_from_scratch"

            history.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "lm_loss": step_lm_loss_val,
                    "mtp_loss": step_mtp_loss_val,
                    "intermediate_loss": step_inter_loss_val,
                    "tokens_seen": model.tokens_seen,
                }
            )

            if max_wall_hours is not None:
                elapsed_hours = (time.monotonic() - start_wall) / 3600.0
                if elapsed_hours >= float(max_wall_hours):
                    stopped_by_wall_clock = True
                    break

        if stopped_by_wall_clock:
            break

    val_metrics = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
        )
        val_metrics = evaluate_maya(
            model,
            val_loader,
            mtp_weight=mtp_weight,
            intermediate_supervision_weight=intermediate_supervision_weight,
            device=dev,
        )
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            if best_checkpoint_path is not None:
                save_checkpoint_maya(
                    model,
                    optimizer,
                    scheduler,
                    trainer_state={
                        "step": step,
                        "epoch": epoch,
                        "next_batch_cursor": cursor,
                        "permutation": permutation,
                        "batch_size": batch_size,
                        "dataset_fingerprint": dataset_fp,
                        "best_val_loss": best_val_loss,
                        "history": history,
                        "stopped_by_wall_clock": stopped_by_wall_clock,
                        "checkpoint_role": str(checkpoint_role),
                    },
                    path=best_checkpoint_path,
                    extra={"val_metrics": val_metrics},
                )

    trainer_state = {
        "step": step,
        "epoch": epoch,
        "next_batch_cursor": cursor,
        "permutation": permutation,
        "batch_size": batch_size,
        "dataset_fingerprint": dataset_fp,
        "best_val_loss": best_val_loss,
        "history": history,
        "stopped_by_wall_clock": stopped_by_wall_clock,
        "val_metrics": val_metrics,
        "checkpoint_role": str(checkpoint_role),
    }
    if checkpoint_path is not None:
        save_checkpoint_maya(
            model,
            optimizer,
            scheduler,
            trainer_state=trainer_state,
            path=checkpoint_path,
            extra={"val_metrics": val_metrics},
        )
    return trainer_state
