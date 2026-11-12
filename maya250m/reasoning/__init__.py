"""Reasoning subsystem: work state, adaptive depth, decomposition, self-correction."""

from __future__ import annotations

from .controller import DEFAULT_DIFFICULTY_WEIGHTS, AdaptiveLoopController
from .correction import CorrectionPlan, CorrectionPolicy, diagnose_failure
from .decompose import (
    STRATEGIES,
    Decomposition,
    DecompositionError,
    StrategySelector,
    Subproblem,
    decompose_task,
)
from .state import (
    PHASE_DONE,
    PHASE_FAILED,
    PHASE_HYPOTHESIZE,
    PHASE_REPRODUCE,
    PHASE_REVIEW,
    PHASE_TRANSITIONS,
    PHASE_UNDERSTAND,
    PHASE_VERIFY,
    PHASES,
    Hypothesis,
    ReasoningState,
    ReasoningStateError,
)

__all__ = [
    "AdaptiveLoopController",
    "DEFAULT_DIFFICULTY_WEIGHTS",
    "CorrectionPlan",
    "CorrectionPolicy",
    "diagnose_failure",
    "STRATEGIES",
    "Decomposition",
    "DecompositionError",
    "StrategySelector",
    "Subproblem",
    "decompose_task",
    "PHASES",
    "PHASE_TRANSITIONS",
    "PHASE_UNDERSTAND",
    "PHASE_REPRODUCE",
    "PHASE_HYPOTHESIZE",
    "PHASE_VERIFY",
    "PHASE_REVIEW",
    "PHASE_DONE",
    "PHASE_FAILED",
    "Hypothesis",
    "ReasoningState",
    "ReasoningStateError",
]
