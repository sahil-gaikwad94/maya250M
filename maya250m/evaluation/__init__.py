"""Evaluation: five separate identities, executable verifiers, A-P ablation matrix."""

from __future__ import annotations

from .ablations import (
    ABLATIONS,
    AblationResult,
    AblationSpec,
    ablation_table,
    apply_ablation,
    run_ablation,
    run_ablation_matrix,
)
from .harness import (
    EvalTask,
    TaskOutcome,
    composed_verifier,
    constraint_verifier,
    contains_all_verifier,
    exact_match_verifier,
    pass_at_k,
    run_benchmark,
    run_tasks,
    unit_test_verifier,
)
from .identity import (
    EvalReport,
    Measurement,
    MeasurementError,
    MeasurementIdentity,
    MeasurementStatus,
)

__all__ = [
    "ABLATIONS",
    "AblationResult",
    "AblationSpec",
    "ablation_table",
    "apply_ablation",
    "run_ablation",
    "run_ablation_matrix",
    "EvalTask",
    "TaskOutcome",
    "composed_verifier",
    "constraint_verifier",
    "contains_all_verifier",
    "exact_match_verifier",
    "pass_at_k",
    "run_benchmark",
    "run_tasks",
    "unit_test_verifier",
    "EvalReport",
    "Measurement",
    "MeasurementError",
    "MeasurementIdentity",
    "MeasurementStatus",
]
