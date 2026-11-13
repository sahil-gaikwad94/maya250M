"""Bounded CodeAct runtime: 5-phase state machine, loop breaker, budget discipline.

The runtime is the only place allowed to execute a tool. It enforces:

* the reasoning phase graph (see :mod:`maya250m.reasoning.state`);
* a hard ``max_steps`` budget plus working-memory eviction;
* a **duplicate-action loop breaker**: an identical read/search action repeated
  without an intervening edit is intercepted and answered with a guard message
  instead of being executed again;
* the tool-output firewall, so no observation reaches model context unfenced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..reasoning.state import (
    PHASE_HYPOTHESIZE,
    PHASE_REPRODUCE,
    PHASE_VERIFY,
    PHASE_TRANSITIONS,
    ReasoningState,
    ReasoningStateError,
)
from ..verification import VerificationResult, failure
from .firewall import ToolOutputFirewall
from .tools import RepoTools, ToolCallError
from .trajectories import (
    AgentTrajectory,
    ProcessReward,
    TrajectoryStep,
    score_trajectory,
)

__all__ = ["ToolObservation", "AgentRuntime", "LOOP_GUARD_TEMPLATE"]

LOOP_GUARD_TEMPLATE = (
    "[LOOP_GUARD: You already executed this exact query at step {step}; "
    "advance to hypothesis or inspect a different span]"
)

_READ_TOOLS = {
    "get_repo_map",
    "get_repo_context",
    "search_repo",
    "view_file_span",
    "inspect_symbol_graph",
}
_EDIT_TOOLS = {"apply_patch", "git_revert_file"}

#: Next phase to try after a successful call; applied only when the edge is legal.
_PHASE_AFTER_TOOL = {
    "get_repo_map": PHASE_REPRODUCE,
    "get_repo_context": PHASE_REPRODUCE,
    "search_repo": PHASE_REPRODUCE,
    "view_file_span": PHASE_REPRODUCE,
    "inspect_symbol_graph": PHASE_REPRODUCE,
    "apply_patch": PHASE_VERIFY,
    "git_revert_file": PHASE_HYPOTHESIZE,
}


@dataclass
class ToolObservation:
    """One executed (or intercepted) tool call."""

    step: int
    tool_name: str
    tool_args: Dict[str, Any]
    fenced_text: str
    result: Any = None
    verification: Optional[VerificationResult] = None
    loop_guarded: bool = False
    firewall_audit: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.verification.passed) if self.verification is not None else not self.loop_guarded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "tool_name": self.tool_name,
            "tool_args": dict(self.tool_args),
            "fenced_text": self.fenced_text,
            "loop_guarded": self.loop_guarded,
            "verification": self.verification.to_dict() if self.verification else None,
            "firewall_audit": dict(self.firewall_audit),
        }


class AgentRuntime:
    """Stateful, bounded tool-execution machine for one engineering episode."""

    def __init__(
        self,
        tools: RepoTools,
        state: Optional[ReasoningState] = None,
        max_steps: int = 12,
        firewall: Optional[ToolOutputFirewall] = None,
        task_id: str = "",
        ground_truth_files: Sequence[str] = (),
        ground_truth_symbols: Sequence[str] = (),
    ):
        self.tools = tools
        self.state = state or ReasoningState(objective="", max_steps=max_steps)
        if self.state.max_steps != int(max_steps):
            self.state.max_steps = int(max_steps)
        self.firewall = firewall or ToolOutputFirewall()
        self.trajectory = AgentTrajectory(task_id=task_id)
        self.ground_truth_files = list(ground_truth_files)
        self.ground_truth_symbols = list(ground_truth_symbols)
        self._seen_actions: Dict[str, int] = {}
        self._edit_generation = 0
        self._current_step = 0
        self.last_patch_text: Optional[str] = None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _action_key(tool_name: str, kwargs: Dict[str, Any]) -> str:
        return f"{tool_name}::{json.dumps(kwargs, sort_keys=True, default=str)}"

    def _maybe_advance(self, tool_name: str) -> None:
        target = _PHASE_AFTER_TOOL.get(tool_name)
        if target is None:
            return
        if target in PHASE_TRANSITIONS.get(self.state.phase, ()):
            try:
                self.state.transition(target)
            except ReasoningStateError:  # pragma: no cover - defensive
                pass

    def _record_step(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        fenced_text: str,
        status: str,
        verification: Optional[VerificationResult],
        thought: str = "",
    ) -> TrajectoryStep:
        step = TrajectoryStep(
            step=self._current_step,
            phase=self.state.phase,
            thought=thought,
            tool_name=tool_name,
            tool_args=dict(kwargs),
            observation=fenced_text,
            status=status,
            verification=verification.to_dict() if verification is not None else None,
        )
        self.trajectory.add_step(step)
        if verification is not None and not verification.passed:
            self.state.last_failure_category = verification.failure_category
        return step

    # ---------------------------------------------------------------- execution

    def call_tool(
        self,
        tool_name: str,
        thought: str = "",
        **kwargs: Any,
    ) -> ToolObservation:
        """Execute one tool call under budget, loop-guard and firewall control."""
        if self.state.finished:
            raise ReasoningStateError("episode already finished")
        if self.state.budget_exhausted:
            observation = ToolObservation(
                step=self._current_step,
                tool_name=tool_name,
                tool_args=dict(kwargs),
                fenced_text=self.firewall.wrap(
                    "budget", "step budget exhausted; stop and report verified state"
                )[0],
                verification=failure("budget_exhausted", "step budget exhausted"),
                loop_guarded=True,
            )
            self._record_step(
                tool_name, kwargs, observation.fenced_text, "budget", observation.verification, thought
            )
            return observation

        self._current_step += 1
        self.state.consume_step(f"{tool_name} {json.dumps(kwargs, sort_keys=True, default=str)[:120]}")

        key = self._action_key(tool_name, kwargs)
        if tool_name in _READ_TOOLS:
            previous = self._seen_actions.get(key)
            if previous is not None and self._seen_actions.get(f"__gen__{key}") == self._edit_generation:
                guard_text, audit = self.firewall.wrap(
                    "loop_guard", LOOP_GUARD_TEMPLATE.format(step=previous)
                )
                observation = ToolObservation(
                    step=self._current_step,
                    tool_name=tool_name,
                    tool_args=dict(kwargs),
                    fenced_text=guard_text,
                    verification=failure("loop_detected", "duplicate read/search action"),
                    loop_guarded=True,
                    firewall_audit=audit,
                )
                self._record_step(
                    tool_name, kwargs, guard_text, "loop_guard", observation.verification, thought
                )
                return observation
            self._seen_actions[key] = self._current_step
            self._seen_actions[f"__gen__{key}"] = self._edit_generation

        verification: Optional[VerificationResult] = None
        status = "ok"
        try:
            result = self.tools.call(tool_name, **kwargs)
        except ToolCallError as exc:
            verification = failure("invalid_tool_call", str(exc))
            result = None
            status = "failed"
        except Exception as exc:  # noqa: BLE001 - tool failures must never crash the loop
            verification = failure("sandbox_error", f"{type(exc).__name__}: {exc}")
            result = None
            status = "failed"

        if isinstance(result, VerificationResult):
            verification = result
            status = "ok" if result.passed else "failed"

        if verification is not None and verification.passed and tool_name == "apply_patch":
            self._edit_generation += 1
            self.last_patch_text = kwargs.get("replace_block")
            self.state.record_hypothesis(
                f"patch {kwargs.get('file_path')}", patch_targets=[kwargs.get("file_path", "")]
            )
            try:
                self.state.record_verification(verification)
            except ReasoningStateError:  # pragma: no cover - defensive
                pass
        elif verification is not None and not verification.passed and tool_name == "apply_patch":
            self._edit_generation += 1
            if not self.state.hypotheses:
                self.state.record_hypothesis(f"patch {kwargs.get('file_path')}")
            try:
                self.state.record_verification(verification)
            except ReasoningStateError:  # pragma: no cover - defensive
                pass

        payload = result if result is not None else (verification.to_dict() if verification else "error")
        fenced_text, audit = self.firewall.wrap(tool_name, self._serialize(payload))
        observation = ToolObservation(
            step=self._current_step,
            tool_name=tool_name,
            tool_args=dict(kwargs),
            fenced_text=fenced_text,
            result=result,
            verification=verification,
            firewall_audit=audit,
        )
        self._record_step(tool_name, kwargs, fenced_text, status, verification, thought)

        if verification is None or verification.passed:
            self._maybe_advance(tool_name)

        if tool_name == "run_tests" and verification is not None and verification.passed:
            self.state.record_constraint_coverage(
                {k: True for k in self.state.constraint_status}
            )
        return observation

    @staticmethod
    def _serialize(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, indent=2, sort_keys=True, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return repr(payload)

    # ---------------------------------------------------------------- conveniences

    def localize(self, query: str, max_tokens: int = 1200) -> ToolObservation:
        return self.call_tool("get_repo_map", query=query, max_tokens=max_tokens)

    def repo_context(self, query: str, max_tokens: int = 1200) -> ToolObservation:
        return self.call_tool("get_repo_context", query=query, max_tokens=max_tokens)

    def search(self, pattern: str, max_hits: int = 20) -> ToolObservation:
        return self.call_tool("search_repo", pattern=pattern, max_hits=max_hits)

    def view(self, file_path: str, start_line: int, end_line: int) -> ToolObservation:
        return self.call_tool(
            "view_file_span",
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
        )

    def inspect(self, symbol_name: str) -> ToolObservation:
        return self.call_tool("inspect_symbol_graph", symbol_name=symbol_name)

    def patch(self, file_path: str, search_block: str, replace_block: str) -> ToolObservation:
        return self.call_tool(
            "apply_patch",
            file_path=file_path,
            search_block=search_block,
            replace_block=replace_block,
        )

    def run_tests(self, test_target: str = "", timeout_s: float = 10.0) -> ToolObservation:
        return self.call_tool(
            "run_tests", test_target=test_target, timeout_s=timeout_s
        )

    def revert(self, file_path: str) -> ToolObservation:
        return self.call_tool("git_revert_file", file_path=file_path)

    def finish(self, passed: bool, note: str = "") -> AgentTrajectory:
        """Close the episode, freezing the trajectory and outcome."""
        self.trajectory.outcome = 1 if passed else 0
        self.trajectory.metadata["note"] = note
        self.trajectory.metadata["steps_used"] = self.state.steps_used
        self.trajectory.metadata["final_phase"] = self.state.phase
        self.trajectory.metadata["rejected_approaches"] = list(self.state.rejected_approaches)
        self.trajectory.final_patch = self.last_patch_text
        try:
            self.state.finalize(bool(passed))
        except ReasoningStateError:
            self.state.phase = "DONE" if passed else "FAILED"
            self.state.finished = True
        return self.trajectory

    def reward(self) -> ProcessReward:
        return score_trajectory(
            self.trajectory,
            ground_truth_files=self.ground_truth_files,
            ground_truth_symbols=self.ground_truth_symbols,
        )

    def pinned_context(self) -> str:
        """Working context for the next model call: state + latest failure evidence."""
        return self.state.pinned_context()
