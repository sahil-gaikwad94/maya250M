"""Verifier-driven error diagnosis and targeted self-correction.

Instead of regenerating blindly after a failure, ``diagnose_failure`` maps the
structured ``VerificationResult.failure_category`` onto one specific corrective
action, and ``apply_correction`` performs it through the deterministic agent
tools. Every correction is bounded: after ``max_attempts`` refutations the
episode must abstain instead of spinning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..verification import VerificationResult

__all__ = ["CorrectionPlan", "diagnose_failure", "CorrectionPolicy"]

#: category -> (action, human-readable instruction)
_CORRECTION_TABLE: Dict[str, tuple] = {
    "patch_not_found": (
        "widen_context",
        "The SEARCH block did not match. Re-read the enclosing function and include 2-3 "
        "surrounding context lines verbatim (indentation included).",
    ),
    "patch_ambiguous": (
        "narrow_context",
        "The SEARCH block matched multiple locations. Include the enclosing function or "
        "class header so the anchor is unique.",
    ),
    "syntax_error": (
        "fix_syntax",
        "The patch does not parse. Fix indentation/brackets and re-verify with ast.parse "
        "before applying.",
    ),
    "ast_policy_violation": (
        "remove_forbidden_construct",
        "The patch uses a forbidden construct (imports, dunder access, attribute on a "
        "literal). Rewrite using only allowed syntax.",
    ),
    "tests_failed": (
        "inspect_failure",
        "Tests failed. Read the failing test's traceback, localize the exact assertion, "
        "and change the smallest responsible span.",
    ),
    "cross_file_arity_breakage": (
        "update_callers",
        "A caller now passes the wrong number of arguments. Update the caller or restore "
        "the original signature in the same patch set.",
    ),
    "cross_file_missing_symbol": (
        "restore_symbol",
        "A referenced symbol no longer exists. Restore it, or update every reference.",
    ),
    "symbol_not_found": (
        "search_symbol",
        "The symbol was not found in the index. Search the repository for its current "
        "name or definition site.",
    ),
    "localization_failure": (
        "expand_search",
        "Localization was too narrow. Expand the regex search over the whole repository "
        "and inspect the call graph.",
    ),
    "dependency_cycle": (
        "break_cycle",
        "The dependency graph has a cycle. Introduce an interface/lazy import to break it.",
    ),
    "test_timeout": (
        "reduce_scope",
        "The test run timed out. Narrow the test target to the module under change.",
    ),
    "loop_detected": (
        "change_action",
        "The same action was repeated. Choose a different span, hypothesis, or tool.",
    ),
    "budget_exhausted": (
        "abstain",
        "The step budget is exhausted. Stop and report the current state with evidence.",
    ),
    "sandbox_error": (
        "fix_runtime_error",
        "The sandbox raised a runtime error. Inspect the exception type and repair the "
        "offending expression.",
    ),
    "schema_violation": (
        "repair_schema",
        "The output violates the task schema. Emit every required field explicitly.",
    ),
}


@dataclass
class CorrectionPlan:
    """One diagnosed failure and the concrete next action it implies."""

    action: str
    instruction: str
    failure_category: str
    failed_requirements: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "instruction": self.instruction,
            "failure_category": self.failure_category,
            "failed_requirements": list(self.failed_requirements),
            "evidence": dict(self.evidence),
        }


def diagnose_failure(result: VerificationResult) -> CorrectionPlan:
    """Map a failed verification onto exactly one corrective action."""
    if result.passed:
        raise ValueError("diagnose_failure requires a failed VerificationResult")
    category = result.failure_category or "internal_error"
    action, instruction = _CORRECTION_TABLE.get(
        category,
        (
            "generalize",
            "Unexpected failure. Re-read the objective and constraints, then form a new "
            "hypothesis instead of repeating the previous patch.",
        ),
    )
    return CorrectionPlan(
        action=action,
        instruction=instruction,
        failure_category=category,
        failed_requirements=list(result.failed_requirements),
        evidence=dict(result.details),
    )


class CorrectionPolicy:
    """Bounded self-correction: refuses to retry forever and escalates to abstention."""

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = int(max_attempts)
        self.history: List[CorrectionPlan] = []

    def next_plan(self, result: VerificationResult) -> CorrectionPlan:
        if result.passed:
            raise ValueError("CorrectionPolicy.next_plan requires a failing result")
        plan = diagnose_failure(result)
        # If the same category has already been refuted ``max_attempts`` times,
        # escalate to abstention rather than repeating the same corrective action.
        repeats = sum(1 for p in self.history if p.failure_category == plan.failure_category)
        if repeats >= self.max_attempts:
            plan = CorrectionPlan(
                action="abstain",
                instruction=(
                    f"{plan.failure_category} has persisted {repeats} times. Stop, report "
                    "the verified evidence, and state that the issue is unresolved."
                ),
                failure_category=plan.failure_category,
                failed_requirements=list(plan.failed_requirements),
                evidence=dict(plan.evidence),
            )
        self.history.append(plan)
        return plan

    def should_abstain(self) -> bool:
        return bool(self.history) and self.history[-1].action == "abstain"

    def reset(self) -> None:
        self.history.clear()
