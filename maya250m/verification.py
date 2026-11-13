"""Structured verification results and the failure taxonomy shared by Vexx 3.5,
the repository tools, the agent runtime and every staged trainer.

A ``VerificationResult`` is the only currency used to decide whether an action
succeeded. Nothing in the system is allowed to infer success from a tool exit
code, a generated string, or a reward model score; every consumer must inspect
this record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "VerificationResult",
    "FAILURE_CATEGORIES",
    "ok",
    "failure",
    "classify_python_error",
]

#: Canonical failure taxonomy. Every ``VerificationResult.failure_category`` must
#: be one of these strings (or ``None`` when ``passed`` is True).
FAILURE_CATEGORIES = (
    "none",
    "syntax_error",
    "ast_policy_violation",
    "patch_error",
    "patch_not_found",
    "patch_ambiguous",
    "dependency_cycle",
    "tests_failed",
    "test_timeout",
    "cross_file_arity_breakage",
    "cross_file_missing_symbol",
    "test_tampering_blocked",
    "transaction_conflict",
    "wrong_answer",
    "missing_constraint",
    "ambiguous_request",
    "generation_error",
    "symbol_not_found",
    "localization_failure",
    "budget_exhausted",
    "invalid_tool_call",
    "loop_detected",
    "sandbox_error",
    "schema_violation",
    "internal_error",
)


@dataclass
class VerificationResult:
    """Outcome of one verifiable action (patch, test run, tool call, ...)."""

    passed: bool
    failure_category: Optional[str] = None
    message: str = ""
    failed_requirements: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.passed and self.failure_category is None:
            self.failure_category = "internal_error"
        if self.passed and self.failure_category not in (None, "none"):
            raise ValueError(
                "A passing VerificationResult cannot carry a failure_category"
            )
        if self.failure_category is not None and self.failure_category not in FAILURE_CATEGORIES:
            raise ValueError(
                f"Unknown failure_category {self.failure_category!r}; "
                f"valid: {sorted(FAILURE_CATEGORIES)}"
            )

    @property
    def is_retryable(self) -> bool:
        """Whether a bounded agent loop should attempt another hypothesis."""
        return bool(
            not self.passed
            and self.failure_category
            in {
                "syntax_error",
                "ast_policy_violation",
                "patch_error",
                "patch_not_found",
                "patch_ambiguous",
                "tests_failed",
                "cross_file_arity_breakage",
                "cross_file_missing_symbol",
                "symbol_not_found",
                "invalid_tool_call",
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "failure_category": self.failure_category,
            "message": self.message,
            "failed_requirements": list(self.failed_requirements),
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VerificationResult":
        return cls(
            passed=bool(payload["passed"]),
            failure_category=payload.get("failure_category"),
            message=str(payload.get("message", "")),
            failed_requirements=list(payload.get("failed_requirements") or []),
            details=dict(payload.get("details") or {}),
        )

    def __bool__(self) -> bool:  # convenience: ``if result:`` means passed
        return bool(self.passed)


def ok(message: str = "", **details: Any) -> VerificationResult:
    """Construct a passing result."""
    return VerificationResult(passed=True, message=message, details=dict(details))


def failure(
    category: str,
    message: str,
    failed_requirements: Optional[List[str]] = None,
    **details: Any,
) -> VerificationResult:
    """Construct a failing result with a canonical category."""
    return VerificationResult(
        passed=False,
        failure_category=category,
        message=message,
        failed_requirements=list(failed_requirements or []),
        details=dict(details),
    )


_PY_ERROR_CATEGORY = {
    "SyntaxError": "syntax_error",
    "IndentationError": "syntax_error",
    "TabError": "syntax_error",
    "TimeoutError": "test_timeout",
    "PermissionError": "ast_policy_violation",
    "ImportError": "cross_file_missing_symbol",
    "ModuleNotFoundError": "cross_file_missing_symbol",
    "NameError": "cross_file_missing_symbol",
    "TypeError": "cross_file_arity_breakage",
}


def classify_python_error(exc: BaseException) -> str:
    """Map a raised Python exception onto a failure-taxonomy category."""
    for base in type(exc).__mro__:
        if base.__name__ in _PY_ERROR_CATEGORY:
            return _PY_ERROR_CATEGORY[base.__name__]
    return "internal_error"
