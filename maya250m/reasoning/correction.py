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

__all__ = [
    "CorrectionPlan",
    "diagnose_failure",
    "CorrectionPolicy",
    "ast_normalized_fingerprint",
    "extract_traceback_diagnosis",
]

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
        self.seen_patch_fingerprints: List[str] = []

    def record_patch_attempt(self, code_or_patch: str) -> Dict[str, Any]:
        """Fingerprint a proposed code/patch attempt and flag AST-equivalent duplicates."""
        fp = ast_normalized_fingerprint(code_or_patch)
        duplicate = fp in self.seen_patch_fingerprints
        self.seen_patch_fingerprints.append(fp)
        return {"fingerprint": fp, "is_duplicate": duplicate, "attempt": len(self.seen_patch_fingerprints)}

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
        self.seen_patch_fingerprints.clear()


import ast
import hashlib
import re

_TB_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in ([A-Za-z0-9_.<>]+)')
_ASSERT_LINE_RE = re.compile(r"(AssertionError.*|E\s+assert\s+.+)")


class _AlphaNormalizer(ast.NodeTransformer):
    """Canonicalize local variable & parameter names inside functions for AST dedup."""

    def _canonicalize_function(self, node: ast.AST) -> ast.AST:
        mapping: Dict[str, str] = {}

        def _bind(name: str) -> str:
            if name not in mapping:
                mapping[name] = f"_v{len(mapping)}"
            return mapping[name]

        # Bind positional/kwonly/vararg/kwarg parameter names
        args = getattr(node, "args", None)
        if args is not None:
            for arg in list(getattr(args, "posonlyargs", [])) + list(args.args) + list(args.kwonlyargs):
                arg.arg = _bind(arg.arg)
            if args.vararg is not None:
                args.vararg.arg = _bind(args.vararg.arg)
            if args.kwarg is not None:
                args.kwarg.arg = _bind(args.kwarg.arg)

        # Collect locally assigned variable names (Store targets)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                _bind(sub.id)

        # Rewrite all Name nodes bound in this function scope
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in mapping:
                sub.id = mapping[sub.id]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._canonicalize_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._canonicalize_function(node)


def ast_normalized_fingerprint(code_or_patch: str, *, alpha_normalize: bool = True) -> str:
    """Compute a whitespace-, comment-, and local-variable-rename-invariant AST fingerprint."""
    text = str(code_or_patch or "").strip()
    try:
        tree = ast.parse(text)
        if alpha_normalize:
            tree = _AlphaNormalizer().visit(tree)
            ast.fix_missing_locations(tree)
        canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    except SyntaxError:
        canonical = " ".join(text.split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_traceback_diagnosis(
    output_text: str, indexer: Any = None
) -> Dict[str, Any]:
    """Extract failing frames, assertions, and 1-hop repository call neighborhood from test output."""
    text = str(output_text or "")
    frames: List[Dict[str, Any]] = []
    implicated_symbols: List[str] = []
    implicated_callers: List[str] = []

    for match in _TB_FRAME_RE.finditer(text):
        file_path, line_str, func_name = match.group(1), match.group(2), match.group(3)
        line_no = int(line_str)
        frames.append({"file": file_path, "line": line_no, "function": func_name})
        if func_name and not func_name.startswith("<") and func_name not in implicated_symbols:
            implicated_symbols.append(func_name)
        if indexer is not None and hasattr(indexer, "callers"):
            for caller_site in indexer.callers(func_name) or []:
                caller_q = getattr(caller_site, "caller_qualname", None)
                if caller_q and caller_q not in implicated_callers:
                    implicated_callers.append(caller_q)

    assertions = [m.group(1).strip() for m in _ASSERT_LINE_RE.finditer(text)]
    return {
        "frames": frames,
        "implicated_symbols": implicated_symbols,
        "implicated_callers": implicated_callers,
        "failing_assertions": assertions[:5],
    }
