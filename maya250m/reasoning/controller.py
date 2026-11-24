"""Request-level adaptive loop-depth control for the Maya recurrent core.

``AdaptiveLoopController`` chooses ``R in {1, 2, 3, 4}`` — the number of times the
16-layer Recurrent Core is executed — from *observable* task difficulty signals.
It is a deterministic, auditable policy: no extra parameters, no hidden state, so
an evaluation report can always reproduce the chosen depth from the task spec.

Fixed-depth (``R = config.recurrent_loops``) remains the default for training and
for any caller that does not opt in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..understanding import TaskSpec
from .state import ReasoningState

__all__ = ["AdaptiveLoopController", "DEFAULT_DIFFICULTY_WEIGHTS"]

#: Weight per difficulty signal. Positive weights increase the chosen loop count.
DEFAULT_DIFFICULTY_WEIGHTS: Dict[str, float] = {
    "constraints": 0.6,
    "requirements": 0.5,
    "files": 0.9,
    "symbols": 0.4,
    "ambiguity": 0.8,
    "prior_failures": 1.0,
    "task_type": 0.7,
    "long_prompt": 0.4,
}

#: Task types that empirically benefit from extra latent depth.
_LOOP_HUNGRY_TASKS = ("bug_fix", "math", "research", "refactor", "code_edit")


@dataclass
class AdaptiveLoopController:
    """Map task-difficulty signals onto a loop count in ``[min_loops, max_loops]``."""

    min_loops: int = 1
    max_loops: int = 4
    thresholds: tuple = (1.0, 2.5, 4.5)  # score bands for R=2, R=3, R=4
    weights: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not 1 <= int(self.min_loops) <= int(self.max_loops) <= 8:
            raise ValueError("require 1 <= min_loops <= max_loops <= 8")
        if len(self.thresholds) != int(self.max_loops) - int(self.min_loops):
            raise ValueError(
                "len(thresholds) must equal max_loops - min_loops "
                f"({self.max_loops - self.min_loops})"
            )
        self.weights = dict(DEFAULT_DIFFICULTY_WEIGHTS if self.weights is None else self.weights)

    def difficulty_score(
        self,
        spec: Optional[TaskSpec] = None,
        state: Optional[ReasoningState] = None,
        prompt_tokens: Optional[int] = None,
    ) -> float:
        """Compute a non-negative difficulty score from observable signals."""
        score = 0.0
        if spec is not None:
            score += self.weights["constraints"] * len(spec.constraints)
            score += self.weights["requirements"] * len(spec.requirements)
            score += self.weights["files"] * max(0, len(spec.files_mentioned) - 1)
            score += self.weights["symbols"] * max(0, len(spec.symbols_mentioned) - 2)
            score += self.weights["ambiguity"] * len(spec.ambiguity_flags)
            if spec.task_type in _LOOP_HUNGRY_TASKS:
                score += self.weights["task_type"]
        if state is not None:
            score += self.weights["prior_failures"] * len(state.rejected_approaches)
        if prompt_tokens is not None and prompt_tokens > 1024:
            score += self.weights["long_prompt"]
        return float(max(0.0, score))

    def select_loops(
        self,
        spec: Optional[TaskSpec] = None,
        state: Optional[ReasoningState] = None,
        prompt_tokens: Optional[int] = None,
        fixed: Optional[int] = None,
    ) -> int:
        """Return the loop count for this request (``fixed`` overrides the policy)."""
        if fixed is not None:
            return max(1, min(int(self.max_loops), int(fixed)))
        score = self.difficulty_score(spec=spec, state=state, prompt_tokens=prompt_tokens)
        loops = int(self.min_loops)
        for threshold in self.thresholds:
            if score >= float(threshold):
                loops += 1
        return max(int(self.min_loops), min(int(self.max_loops), loops))

    def explain(
        self,
        spec: Optional[TaskSpec] = None,
        state: Optional[ReasoningState] = None,
        prompt_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return the decision plus the signal breakdown for evaluation logs."""
        score = self.difficulty_score(spec=spec, state=state, prompt_tokens=prompt_tokens)
        return {
            "difficulty_score": score,
            "loops": self.select_loops(spec=spec, state=state, prompt_tokens=prompt_tokens),
            "thresholds": list(self.thresholds),
            "signals": {
                "constraints": len(spec.constraints) if spec else 0,
                "requirements": len(spec.requirements) if spec else 0,
                "files": len(spec.files_mentioned) if spec else 0,
                "ambiguity": len(spec.ambiguity_flags) if spec else 0,
                "prior_failures": len(state.rejected_approaches) if state else 0,
            },
        }

    def marginal_compute_value(
        self,
        loop_idx: int,
        current_entropy: float,
        prev_entropy: Optional[float] = None,
        cost_per_loop: float = 0.08,
    ) -> float:
        """Estimated net value of executing another recurrent core pass.

        Positive when the relative reduction in predictive uncertainty exceeds
        ``cost_per_loop``; negative when representation refinement has saturated.
        """
        if loop_idx >= int(self.max_loops):
            return -float(cost_per_loop)
        cur = max(0.0, float(current_entropy))
        if prev_entropy is None:
            return float(cur - cost_per_loop)
        prev = max(1e-6, float(prev_entropy))
        relative_drop = (prev - cur) / prev
        return float(relative_drop - float(cost_per_loop))

    def should_halt_early(
        self,
        loop_idx: int,
        current_entropy: float,
        prev_entropy: Optional[float] = None,
        *,
        cost_per_loop: float = 0.08,
        entropy_floor: float = 0.05,
        top_margin: Optional[float] = None,
        margin_threshold: float = 4.0,
    ) -> Dict[str, Any]:
        """Decide whether recurrent core refinement can halt at ``loop_idx`` (1-indexed)."""
        step = int(loop_idx)
        if step < int(self.min_loops):
            return {"halt": False, "reason": "below_min_loops", "loop": step}
        if step >= int(self.max_loops):
            return {"halt": True, "reason": "reached_max_loops", "loop": step}
        if float(current_entropy) <= float(entropy_floor):
            return {"halt": True, "reason": "low_entropy_converged", "loop": step}
        if top_margin is not None and float(top_margin) >= float(margin_threshold):
            return {"halt": True, "reason": "high_logit_margin", "loop": step}
        net_val = self.marginal_compute_value(
            step, current_entropy, prev_entropy=prev_entropy, cost_per_loop=cost_per_loop
        )
        if prev_entropy is not None and net_val <= 0.0:
            return {
                "halt": True,
                "reason": "diminishing_marginal_gain",
                "loop": step,
                "marginal_value": net_val,
            }
        return {
            "halt": False,
            "reason": "continue_refinement",
            "loop": step,
            "marginal_value": net_val,
        }
