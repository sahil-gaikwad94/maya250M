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

import hashlib
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

__all__ = [
    "ToolObservation",
    "AgentRuntime",
    "ConstraintEvidence",
    "CompletionEvidence",
    "LOOP_GUARD_TEMPLATE",
]

REQUIREMENT_STATUSES = ("unknown", "addressed", "verified", "failed", "unverified")
EVIDENCE_KINDS = ("test", "typecheck", "contract", "diff", "manual_review")

_CODE_TASK_TYPES = frozenset(
    {"bug_fix", "code_edit", "code_generation", "refactor", "test_writing"}
)

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
class ConstraintEvidence:
    """What is actually known about one requirement, and the evidence proving it.

    ``status`` is one of :data:`REQUIREMENT_STATUSES`. Only ``verified`` may satisfy
    a mandatory requirement; lexical matches are at best ``addressed``.
    """

    constraint_id: str
    status: str = "unknown"
    evidence_kind: str = "manual_review"
    evidence_ids: List[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in REQUIREMENT_STATUSES:
            raise ValueError(f"unknown requirement status {self.status!r}")
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence kind {self.evidence_kind!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "status": self.status,
            "evidence_kind": self.evidence_kind,
            "evidence_ids": list(self.evidence_ids),
            "note": self.note,
        }


@dataclass
class CompletionEvidence:
    """Structured completion proof supplied to :meth:`AgentRuntime.finalize`.

    A caller-supplied boolean cannot satisfy this contract: the patch candidate, the
    final diff hash, the post-patch test runs and the per-requirement evidence are
    all checked against the runtime's own ledger before an episode can resolve.
    """

    patch_applied: bool
    targeted_tests_passed: bool
    regression_tests_passed: bool
    static_checks_passed: Optional[bool]
    required_constraints_verified: List[str] = field(default_factory=list)
    unresolved_constraints: List[str] = field(default_factory=list)
    final_diff_sha256: Optional[str] = None
    test_run_ids: List[str] = field(default_factory=list)
    #: Requirements the user explicitly accepted as out of scope. Recorded, never
    #: silently inferred; a waiver cannot be manufactured by the agent itself.
    waived_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_applied": bool(self.patch_applied),
            "targeted_tests_passed": bool(self.targeted_tests_passed),
            "regression_tests_passed": bool(self.regression_tests_passed),
            "static_checks_passed": self.static_checks_passed,
            "required_constraints_verified": list(self.required_constraints_verified),
            "unresolved_constraints": list(self.unresolved_constraints),
            "final_diff_sha256": self.final_diff_sha256,
            "test_run_ids": list(self.test_run_ids),
            "waived_constraints": list(self.waived_constraints),
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
        self.requirement_evidence: Dict[str, ConstraintEvidence] = {}
        self.requirement_test_map: Dict[str, List[str]] = {}
        self.mandatory_requirements: Dict[str, bool] = {}
        self.requirement_verifiers: Dict[str, List[str]] = {}
        self._test_runs: List[Dict[str, Any]] = []

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
        tool_kwargs = dict(kwargs)
        if tool_name == "run_tests":
            # ``requirement_ids`` is runtime bookkeeping, not a RepoTools argument.
            tool_kwargs.pop("requirement_ids", None)
        try:
            result = self.tools.call(tool_name, **tool_kwargs)
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

        if tool_name == "run_tests":
            self.record_test_run(kwargs, verification)
        return observation

    # ------------------------------------------------------ requirement ledger

    def register_requirement(
        self,
        text: str,
        requirement_id: Optional[str] = None,
        mandatory: bool = True,
        test_ids: Sequence[str] = (),
        verifier_ids: Sequence[str] = (),
    ) -> str:
        """Register one requirement with a stable id and its verifier declarations."""
        if requirement_id is None:
            requirement_id = f"constraint_{len(self.state.constraints)}"
            self.state.constraints.append(text)
        elif text:
            self.state.pin_fact(f"requirement {requirement_id}: {text}")
        self.state.constraint_status.setdefault(requirement_id, False)
        self.mandatory_requirements[requirement_id] = bool(mandatory)
        self.requirement_evidence.setdefault(
            requirement_id, ConstraintEvidence(constraint_id=requirement_id)
        )
        if test_ids:
            self.bind_requirement_tests(requirement_id, test_ids)
        if verifier_ids:
            self.requirement_verifiers[requirement_id] = [str(v) for v in verifier_ids]
        elif text:
            from ..requirement_evidence import requirement_verifiers_for_text

            self.requirement_verifiers[requirement_id] = requirement_verifiers_for_text(text)
        return requirement_id

    def bind_requirement_tests(
        self, requirement_id: str, test_ids: Sequence[str]
    ) -> None:
        """Declare exactly which test targets verify ``requirement_id``."""
        mapping = self.requirement_test_map.setdefault(requirement_id, [])
        for test_id in test_ids:
            cleaned = str(test_id).strip()
            if cleaned and cleaned not in mapping:
                mapping.append(cleaned)

    def record_requirement_evidence(
        self,
        constraint_id: str,
        status: str,
        evidence_kind: str = "manual_review",
        evidence_ids: Sequence[str] = (),
        note: str = "",
    ) -> ConstraintEvidence:
        """Record what a verifier proved about one requirement.

        Nothing here is inferred from an unrelated passing command; callers pass the
        exact requirement ids a verifier covers.
        """
        evidence = ConstraintEvidence(
            constraint_id=constraint_id,
            status=status,
            evidence_kind=evidence_kind,
            evidence_ids=list(evidence_ids),
            note=note,
        )
        self.requirement_evidence[constraint_id] = evidence
        self.state.constraint_status[constraint_id] = status == "verified"
        return evidence

    def _requirements_for_target(self, test_target: str) -> List[str]:
        """Requirements whose declared test ids match ``test_target``.

        An empty target runs the whole suite, which does not *prove* that any
        particular requirement's test executed, so it maps to nothing.
        """
        target = str(test_target or "").strip().replace("\\", "/")
        if not target:
            return []
        matched: List[str] = []
        for requirement_id, test_ids in self.requirement_test_map.items():
            for test_id in test_ids:
                declared = str(test_id).replace("\\", "/")
                if target == declared or target.endswith(declared) or declared.endswith(target):
                    matched.append(requirement_id)
                    break
        return matched

    def record_test_run(
        self, kwargs: Dict[str, Any], verification: Optional[VerificationResult]
    ) -> str:
        """Log one test command and update only the requirements it covers."""
        target = str(kwargs.get("test_target") or "")
        passed = bool(verification is not None and verification.passed)
        run_id = f"test_run_{len(self._test_runs) + 1}"
        requirement_ids = list(kwargs.get("requirement_ids") or ())
        if not requirement_ids:
            requirement_ids = self._requirements_for_target(target)
        self._test_runs.append(
            {
                "id": run_id,
                "target": target,
                "passed": passed,
                "step": int(self._current_step),
                "edit_generation": int(self._edit_generation),
                "requirement_ids": requirement_ids,
                "failure_category": (
                    verification.failure_category
                    if verification is not None and not passed
                    else None
                ),
            }
        )
        self.state.add_evidence(run_id)
        for requirement_id in requirement_ids:
            self.record_requirement_evidence(
                requirement_id,
                "verified" if passed else "failed",
                "test",
                [run_id],
                note=f"run_tests target={target or '.'}",
            )
        return run_id

    def check_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run typed mechanical verifiers over registered requirements.

        ``context`` supplies the observed facts (signatures, paths, test outcomes,
        dependencies, output format, measured complexity). A requirement with no
        mechanical verifier is recorded ``unverified``, never promoted by wording.
        """
        from ..requirement_evidence import (
            apply_requirement_evidence,
            verify_requirements as _verify_requirements,
        )

        requirements = [
            {
                "requirement_id": requirement_id,
                "verifier_ids": self.requirement_verifiers.get(requirement_id, []),
            }
            for requirement_id in self.requirement_evidence
        ]
        report = _verify_requirements(requirements, context)
        apply_requirement_evidence(self, report)
        return {
            requirement_id: evidence.to_dict()
            for requirement_id, evidence in report.items()
        }

    def requirement_report(self) -> Dict[str, Any]:
        """Per-requirement status with the exact evidence ids that back it."""
        return {
            requirement_id: evidence.to_dict()
            for requirement_id, evidence in sorted(self.requirement_evidence.items())
        }

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

    def run_tests(
        self,
        test_target: str = "",
        timeout_s: float = 10.0,
        requirement_ids: Sequence[str] = (),
    ) -> ToolObservation:
        return self.call_tool(
            "run_tests",
            test_target=test_target,
            timeout_s=timeout_s,
            requirement_ids=list(requirement_ids),
        )

    def revert(self, file_path: str) -> ToolObservation:
        return self.call_tool("git_revert_file", file_path=file_path)

    @staticmethod
    def diff_sha256(patch_text: Optional[str]) -> Optional[str]:
        """Stable hash of the retained patch candidate (``None`` when absent)."""
        if patch_text is None:
            return None
        return hashlib.sha256(str(patch_text).encode("utf-8")).hexdigest()

    def _completion_problems(self, evidence: CompletionEvidence) -> List[str]:
        """Every reason the reported evidence does not prove a resolved episode."""
        problems: List[str] = []
        retained = self.last_patch_text
        if bool(evidence.patch_applied) != (retained is not None):
            problems.append("patch_applied disagrees with the retained patch candidate")
        if retained is not None:
            expected_hash = self.diff_sha256(retained)
            if not evidence.final_diff_sha256:
                problems.append("missing final_diff_sha256 for the retained patch")
            elif evidence.final_diff_sha256 != expected_hash:
                problems.append("final_diff_sha256 does not match the retained patch")
        elif evidence.final_diff_sha256:
            problems.append("final_diff_sha256 supplied with no retained patch")

        if not evidence.targeted_tests_passed:
            problems.append("targeted tests did not pass")
        if not evidence.regression_tests_passed:
            problems.append("regression tests did not pass")
        if evidence.static_checks_passed is False:
            problems.append("static checks failed")
        if not evidence.test_run_ids:
            problems.append("no test run ids reported")
        for run_id in evidence.test_run_ids:
            run = next((item for item in self._test_runs if item["id"] == run_id), None)
            if run is None:
                problems.append(f"unknown test run {run_id}")
                continue
            if not run["passed"]:
                problems.append(f"test run {run_id} did not pass")
            if int(run["edit_generation"]) != int(self._edit_generation):
                problems.append(f"test run {run_id} predates the final patch")

        verified_ids = set(evidence.required_constraints_verified)
        unresolved = set(evidence.unresolved_constraints)
        if unresolved:
            problems.append(
                "unresolved mandatory constraints: " + ", ".join(sorted(unresolved))
            )
        waived = set(evidence.waived_constraints)
        mandatory = {
            requirement_id
            for requirement_id, required in self.mandatory_requirements.items()
            if required
        }
        unknown_waivers = waived - mandatory
        if unknown_waivers:
            problems.append(
                "waived constraints that are not registered mandatory requirements: "
                + ", ".join(sorted(unknown_waivers))
            )
        missing = mandatory - waived - verified_ids - unresolved
        if missing:
            problems.append(
                "mandatory requirements missing from evidence: " + ", ".join(sorted(missing))
            )
        for requirement_id in sorted(verified_ids):
            record = self.requirement_evidence.get(requirement_id)
            if record is None:
                problems.append(f"requirement {requirement_id} has no recorded evidence")
            elif record.status != "verified":
                problems.append(f"requirement {requirement_id} is not verified")
            elif not record.evidence_ids:
                problems.append(f"requirement {requirement_id} has no evidence id")

        if (
            self.state.task_type in _CODE_TASK_TYPES
            and retained is None
            and not evidence.patch_applied
        ):
            problems.append("code-edit episode finished without a retained patch candidate")
        return problems

    def finalize(self, evidence: CompletionEvidence, note: str = "") -> AgentTrajectory:
        """Close the episode from verified evidence, never from a bare boolean."""
        problems = self._completion_problems(evidence)
        verified = not problems
        payload = evidence.to_dict()
        payload["verified"] = verified
        payload["problems"] = list(problems)
        self.trajectory.metadata["completion_evidence"] = payload
        self.trajectory.metadata["note"] = note
        self.trajectory.metadata["steps_used"] = self.state.steps_used
        self.trajectory.metadata["test_runs"] = [dict(run) for run in self._test_runs]
        self.trajectory.metadata["requirements"] = self.requirement_report()
        self.trajectory.metadata["final_phase"] = self.state.phase
        self.trajectory.metadata["rejected_approaches"] = list(self.state.rejected_approaches)
        self.trajectory.final_patch = self.last_patch_text
        self.trajectory.outcome = 1 if verified else 0
        try:
            self.state.finalize(verified)
        except ReasoningStateError:
            self.state.phase = "DONE" if verified else "FAILED"
            self.state.finished = True
        return self.trajectory

    def finish(self, passed: bool = False, note: str = "") -> AgentTrajectory:
        """Deprecated boolean finalizer; it can never produce a resolved trajectory.

        Use :meth:`finalize` with :class:`CompletionEvidence`. Any caller that still
        passes ``True`` gets an unresolved trajectory whose metadata says why, so the
        label can never leak into training as a success.
        """
        self.trajectory.outcome = 0
        self.trajectory.metadata["completion_evidence"] = {
            "verified": False,
            "claimed_passed": bool(passed),
            "problems": ["caller-supplied boolean is not completion evidence"],
        }
        self.trajectory.metadata["note"] = note
        self.trajectory.metadata["steps_used"] = self.state.steps_used
        self.trajectory.metadata["requirements"] = self.requirement_report()
        self.trajectory.metadata["final_phase"] = self.state.phase
        self.trajectory.metadata["rejected_approaches"] = list(self.state.rejected_approaches)
        self.trajectory.final_patch = self.last_patch_text
        try:
            self.state.finalize(False)
        except ReasoningStateError:
            self.state.phase = "FAILED"
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
