"""The A–P ablation matrix.

Each ablation removes exactly one claim-bearing component so its contribution is
measured, not asserted. States are explicit: an ablation that was not executed is
``NOT_RUN`` with a reason, never a zero.

Config overrides only touch keys that exist in ``maya250m.config``; behavioural
ablations (tools, memory, firewall, verifier gate ...) are recorded as disabled
capabilities and plumbed through the runner.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import get_config, validate_config
from .identity import Measurement, MeasurementIdentity, MeasurementStatus

__all__ = [
    "AblationSpec",
    "AblationResult",
    "ABLATIONS",
    "apply_ablation",
    "run_ablation",
    "run_ablation_matrix",
    "ablation_table",
]


@dataclass(frozen=True)
class AblationSpec:
    """One row of the ablation matrix."""

    key: str
    name: str
    description: str
    disables: Tuple[str, ...] = ()
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "disables": list(self.disables),
            "config_overrides": dict(self.config_overrides),
        }


_ALL_FULL_SCHEDULE = "full_layers_only"

ABLATIONS: Dict[str, AblationSpec] = {
    "A": AblationSpec("A", "full_model", "Reference: every component enabled."),
    "B": AblationSpec(
        "B", "no_memory", "Remove the shared-trunk memory projection and retrieval.", ("memory",)
    ),
    "C": AblationSpec(
        "C", "no_tools", "Remove the agent tool suite (repo map, search, tests).", ("tools",)
    ),
    "D": AblationSpec(
        "D", "no_recurrent_loops", "Single dense pass: R = 1 instead of adaptive depth.",
        ("adaptive_depth",), {"recurrent_loops": 1},
    ),
    "E": AblationSpec(
        "E", "no_mtp", "Remove the multi-token-prediction draft head.", ("mtp",), {"mtp_depth": 0}
    ),
    "F": AblationSpec(
        "F", "no_gdn", "Replace every Gated DeltaNet layer with full differential attention.",
        ("linear_attention",), {"hybrid_schedule": _ALL_FULL_SCHEDULE},
    ),
    "G": AblationSpec(
        "G", "no_differential_attention", "Disable differential noise cancellation.",
        ("differential_attention",), {"diff_attn": False},
    ),
    "H": AblationSpec(
        "H", "no_distillation", "Skip on-policy teacher distillation.", ("distillation",)
    ),
    "I": AblationSpec("I", "no_dpo", "Skip preference optimisation.", ("dpo",)),
    "J": AblationSpec("J", "no_grpo", "Skip verifiable-reward RL.", ("grpo",)),
    "K": AblationSpec(
        "K", "no_repo_map", "Disable PPR repo-map localisation.", ("repo_map",)
    ),
    "L": AblationSpec(
        "L", "no_cross_file_verification",
        "Disable cross-file arity/impact verification before commit.",
        ("cross_file_verification",),
    ),
    "M": AblationSpec(
        "M", "no_injection_firewall",
        "Feed raw tool output without marker neutralisation or fencing.",
        ("firewall",),
    ),
    "N": AblationSpec(
        "N", "no_self_correction",
        "Disable failure-driven correction and abstention.",
        ("self_correction",),
    ),
    "O": AblationSpec(
        "O", "no_decomposition",
        "Disable task decomposition; solve monolithically.",
        ("decomposition",),
    ),
    "P": AblationSpec(
        "P", "no_verification_gate",
        "Accept the first patch without AST/policy/test verification.",
        ("verification_gate",),
    ),
}


@dataclass
class AblationResult:
    """Outcome of one ablation run."""

    key: str
    name: str
    status: MeasurementStatus = MeasurementStatus.NOT_RUN
    metrics: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    def to_measurement(self, metric_name: str = "score") -> Measurement:
        value = self.metrics.get(metric_name)
        if self.status is MeasurementStatus.MEASURED and value is None:
            self.status = MeasurementStatus.FAILED
            self.reason = self.reason or f"measured run produced no {metric_name!r}"
        return Measurement(
            identity=MeasurementIdentity.ABLATION,
            name=f"ablation_{self.key}",
            value=value if self.status is MeasurementStatus.MEASURED else None,
            status=self.status,
            n=int(self.metadata.get("n", 0)),
            unit=metric_name,
            metadata={"reason": self.reason, **self.metadata},
        )


def apply_ablation(base_config: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return a validated copy of ``base_config`` with the ablation applied."""
    if key not in ABLATIONS:
        raise KeyError(f"unknown ablation {key!r}; expected one of {sorted(ABLATIONS)}")
    spec = ABLATIONS[key]
    cfg = copy.deepcopy(base_config)

    for name, value in spec.config_overrides.items():
        if value == _ALL_FULL_SCHEDULE:
            cfg["hybrid_schedule"] = ["full"] * int(cfg["layers"])
        else:
            cfg[name] = value

    validate_config(cfg, name=f"ablation_{key}")
    return cfg


def run_ablation(
    key: str,
    base_config: Dict[str, Any],
    train_fn: Optional[Callable[[Dict[str, Any], AblationSpec], Dict[str, float]]] = None,
    metric_name: str = "score",
    reason: str = "no training/evaluation compute was provided in this environment",
) -> AblationResult:
    """Run one ablation if a trainer is supplied; otherwise report NOT_RUN."""
    if key not in ABLATIONS:
        raise KeyError(f"unknown ablation {key!r}")
    spec = ABLATIONS[key]

    if train_fn is None:
        return AblationResult(
            key=key,
            name=spec.name,
            status=MeasurementStatus.NOT_RUN,
            reason=reason,
            metadata={"disables": list(spec.disables)},
        )

    config = apply_ablation(base_config, key)
    try:
        metrics = train_fn(config, spec) or {}
    except Exception as exc:  # noqa: BLE001 - a failed ablation is data, not a crash
        return AblationResult(
            key=key,
            name=spec.name,
            status=MeasurementStatus.FAILED,
            reason=f"{type(exc).__name__}: {exc}",
            metadata={"disables": list(spec.disables)},
        )

    value = metrics.get(metric_name)
    status = MeasurementStatus.MEASURED if value is not None else MeasurementStatus.FAILED
    return AblationResult(
        key=key,
        name=spec.name,
        status=status,
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        reason="" if status is MeasurementStatus.MEASURED else f"no {metric_name!r} returned",
        metadata={"disables": list(spec.disables)},
    )


def run_ablation_matrix(
    base_config: Dict[str, Any],
    keys: Optional[Sequence[str]] = None,
    train_fn: Optional[Callable[[Dict[str, Any], AblationSpec], Dict[str, float]]] = None,
    metric_name: str = "score",
    **kwargs: Any,
) -> List[AblationResult]:
    selected = list(keys) if keys is not None else list(ABLATIONS)
    return [
        run_ablation(key, base_config, train_fn=train_fn, metric_name=metric_name, **kwargs)
        for key in selected
    ]


def ablation_table(preset: str = "MAYA_250M") -> List[Dict[str, Any]]:
    """Static description of the matrix; useful for docs and experiment planning."""
    base = get_config(preset)
    rows: List[Dict[str, Any]] = []
    for key, spec in ABLATIONS.items():
        try:
            cfg = apply_ablation(base, key)
            deltas = {
                name: cfg.get(name) for name in spec.config_overrides if name in cfg
            }
            valid = True
        except Exception as exc:  # noqa: BLE001
            deltas = {"error": str(exc)}
            valid = False
        rows.append({**spec.to_dict(), "config_deltas": deltas, "valid_config": valid})
    return rows
