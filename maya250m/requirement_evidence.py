"""Typed, evidence-backed requirement verification (P0.4 — hardened against false passes).

A task requirement is only *verified* when a mechanical verifier reports the exact
evidence that proves it. Word overlap between the request and the answer is at most
``ADDRESSED``; anything that cannot be checked mechanically is ``UNVERIFIED``.

Hardening guarantees (v4.1):
1. Explicit boolean pass semantics: in ``_check_test_outcomes``, a test target only
   counts as passed when its recorded outcome is explicitly ``True`` (or a non-lexical
   ``VerificationResult`` with ``passed is True``). ``0``, ``None``, ``"failed"``,
   empty dicts, or lexical-only results are treated as failures, never ``VERIFIED``.
2. Per-requirement target scoping: each requirement may declare its own target
   entities (``expected_paths``, ``required_test_ids``, ``expected_signatures``,
   ``forbidden_dependencies``, ``expected_format``, ``expected_complexity``) or have
   them extracted from its ``text``. When a requirement names a specific path, symbol,
   test, or complexity bound that is absent from ``context``, it returns ``FAILED`` or
   ``UNVERIFIED`` rather than falsely inheriting an unrelated requirement's global
   context entry.
3. Asymptotic complexity order comparison: ``_check_complexity`` parses big-O classes
   (``O(1) < O(log n) < O(sqrt(n)) < O(n) < O(n log n) < O(n^2) < O(n^3) < O(2^n)``)
   so a faster measured complexity (e.g. ``O(n)`` when ``O(n log n)`` is required)
   verifies cleanly while a slower complexity (``O(n^2)``) fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "RequirementStatus",
    "RequirementEvidence",
    "VERIFIER_KINDS",
    "apply_requirement_evidence",
    "classify_requirement_kind",
    "extract_requirement_targets",
    "parse_complexity_rank",
    "requirement_verifiers_for_text",
    "verify_requirements",
    "verify_task_spec_requirements",
]


class RequirementStatus(str, Enum):
    """Lifecycle of one requirement's evidence."""

    UNKNOWN = "unknown"
    ADDRESSED = "addressed"
    VERIFIED = "verified"
    FAILED = "failed"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class RequirementEvidence:
    """One requirement plus the verifier and evidence ids that back its status."""

    requirement_id: str
    status: RequirementStatus
    verifier: str
    evidence_ids: Tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "status": self.status.value,
            "verifier": self.verifier,
            "evidence_ids": list(self.evidence_ids),
            "detail": self.detail,
        }


VERIFIER_KINDS = (
    "api_signature",
    "path_presence",
    "test_outcomes",
    "dependency_restrictions",
    "output_schema",
    "complexity",
)

_NORMALIZE_PATH = re.compile(r"[\\/]+")
_FILE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z0-9_./-]+\.(?:py|json|toml|yaml|yml|md|sh|txt|cfg|ini))\b"
)
_BACKTICK_IDENT_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")
_COMPLEXITY_RE = re.compile(
    r"\bO\(\s*(1|log\s*n|sqrt\(n\)|n\s*log\s*n|n\^2|n\*\*2|n2|n\^3|n\*\*3|n3|2\^n|2\*\*n|n!|n)\s*\)",
    re.IGNORECASE,
)


def _clean_path(value: str) -> str:
    return _NORMALIZE_PATH.sub("/", str(value).strip()).lstrip("./")


def _field(requirement: Any, name: str, default: Any = None) -> Any:
    if isinstance(requirement, Mapping):
        return requirement.get(name, default)
    return getattr(requirement, name, default)


def _is_explicit_pass(outcome: Any) -> bool:
    """True only for an explicit boolean True or a non-lexical passing VerificationResult."""
    if outcome is True:
        return True
    if outcome is False or outcome is None or isinstance(outcome, (int, float, str)):
        return False
    passed = getattr(outcome, "passed", None)
    if passed is True:
        details = getattr(outcome, "details", None) or {}
        if isinstance(details, Mapping) and details.get("lexical_only"):
            return False
        return True
    if isinstance(outcome, Mapping):
        if outcome.get("passed") is True and not outcome.get("lexical_only"):
            return True
    return False


def parse_complexity_rank(expr: str) -> Optional[int]:
    """Map standard big-O expressions onto an ordinal rank (smaller is faster)."""
    raw = str(expr or "").strip().lower()
    if not raw:
        return None
    if "constant time" in raw:
        return 0
    if "logarithmic" in raw:
        return 1
    if "linearithmic" in raw:
        return 4
    if "linear time" in raw:
        return 3
    if "quadratic" in raw:
        return 5
    if "cubic" in raw:
        return 6
    m = _COMPLEXITY_RE.search(raw)
    token = (m.group(1) if m else raw).replace(" ", "").replace("**", "^")
    if token.startswith("o(") and token.endswith(")"):
        token = token[2:-1]
    order = {
        "1": 0,
        "logn": 1,
        "sqrt(n)": 2,
        "n": 3,
        "nlogn": 4,
        "n*logn": 4,
        "n^2": 5,
        "n2": 5,
        "n^3": 6,
        "n3": 6,
        "2^n": 7,
        "n!": 8,
    }
    return order.get(token)


def extract_requirement_targets(text: str) -> Dict[str, Any]:
    """Extract requirement-specific paths, tests, symbols, format, and complexity from text."""
    raw = str(text or "")
    lower = raw.lower()
    files = [_clean_path(m.group(1)) for m in _FILE_TOKEN_RE.finditer(raw)]
    test_files = [
        f
        for f in files
        if f.startswith(("tests/", "test/"))
        or "/tests/" in f
        or os.path.basename(f).startswith("test_")
        or os.path.basename(f).endswith("_test.py")
    ]
    non_test_files = [f for f in files if f not in test_files]
    symbols = [
        m.group(1).split("(")[0]
        for m in _BACKTICK_IDENT_RE.finditer(raw)
        if not m.group(1).endswith((".py", ".json", ".md", ".toml", ".yaml", ".yml"))
    ]
    complexity_match = _COMPLEXITY_RE.search(raw)
    expected_complexity: Optional[str] = None
    if complexity_match:
        expected_complexity = f"O({complexity_match.group(1).strip()})"
    elif "linear time" in lower:
        expected_complexity = "O(n)"
    elif "constant time" in lower:
        expected_complexity = "O(1)"

    expected_format: Optional[str] = None
    for fmt in ("json", "yaml", "markdown", "csv", "python"):
        if re.search(rf"\b{fmt}\b", lower) and any(
            w in lower for w in ("format", "schema", "output", "return")
        ):
            expected_format = fmt
            break

    return {
        "paths": non_test_files,
        "test_ids": test_files,
        "symbols": symbols,
        "expected_complexity": expected_complexity,
        "expected_format": expected_format,
    }


import os


def _check_api_signature(
    requirement_id: str, requirement: Any, context: Mapping[str, Any]
) -> RequirementEvidence:
    req_expected = _field(requirement, "expected_signatures")
    global_expected = context.get("expected_signatures") or {}
    observed = context.get("observed_signatures") or {}

    targets = extract_requirement_targets(_field(requirement, "text", ""))
    forbidden_deps = set(_field(requirement, "forbidden_dependencies") or context.get("forbidden_dependencies") or ())
    req_symbols = [
        s for s in (_field(requirement, "symbols", ()) or targets["symbols"])
        if s not in forbidden_deps
    ]

    if req_expected is not None:
        expected = dict(req_expected)
    elif req_symbols:
        missing_from_expected = [s for s in req_symbols if s not in global_expected]
        if missing_from_expected:
            return RequirementEvidence(
                requirement_id,
                RequirementStatus.UNVERIFIED,
                "api_signature",
                (),
                f"no expected signature supplied for requirement symbol(s): {missing_from_expected}",
            )
        expected = {s: global_expected[s] for s in req_symbols}
    else:
        expected = dict(global_expected)

    if not expected:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.UNVERIFIED,
            "api_signature",
            (),
            "no expected API signatures were supplied",
        )
    if not isinstance(observed, Mapping) or not observed:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.FAILED if observed is not None and isinstance(observed, Mapping) and "observed_signatures" in context else RequirementStatus.UNVERIFIED,
            "api_signature",
            tuple(f"signature:{name}" for name in expected),
            "no observed API signatures were supplied",
        )
    missing = sorted(name for name in expected if name not in observed)
    changed = sorted(
        name
        for name in expected
        if name in observed and str(observed[name]) != str(expected[name])
    )
    evidence = tuple(f"signature:{name}" for name in expected)
    if missing or changed:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.FAILED,
            "api_signature",
            evidence,
            f"missing={missing} changed={changed}",
        )
    return RequirementEvidence(
        requirement_id,
        RequirementStatus.VERIFIED,
        "api_signature",
        evidence,
        "all expected signatures preserved",
    )


def _check_path_presence(
    requirement_id: str, requirement: Any, context: Mapping[str, Any]
) -> RequirementEvidence:
    req_paths = _field(requirement, "expected_paths")
    targets = extract_requirement_targets(_field(requirement, "text", ""))
    if req_paths is not None:
        expected = [_clean_path(p) for p in req_paths if str(p).strip()]
    elif targets["paths"]:
        expected = targets["paths"]
    else:
        expected = [_clean_path(p) for p in (context.get("expected_paths") or []) if str(p).strip()]

    present_raw = context.get("present_paths")
    if not expected:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.UNVERIFIED,
            "path_presence",
            (),
            "no expected paths were supplied",
        )
    if present_raw is None:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.UNVERIFIED,
            "path_presence",
            (),
            "present_paths was not supplied in context",
        )
    present = {_clean_path(p) for p in present_raw if str(p).strip()}
    missing = [path for path in expected if path not in present]
    evidence = tuple(f"path:{path}" for path in expected)
    if missing:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.FAILED,
            "path_presence",
            evidence,
            f"missing paths: {missing}",
        )
    return RequirementEvidence(
        requirement_id,
        RequirementStatus.VERIFIED,
        "path_presence",
        evidence,
        "all required paths exist",
    )


def _check_test_outcomes(
    requirement_id: str, requirement: Any, context: Mapping[str, Any]
) -> RequirementEvidence:
    req_tests = _field(requirement, "required_test_ids") or _field(requirement, "test_ids")
    targets = extract_requirement_targets(_field(requirement, "text", ""))
    if req_tests is not None:
        required = [_clean_path(t) for t in req_tests if str(t).strip()]
    elif targets["test_ids"]:
        required = targets["test_ids"]
    else:
        required = [_clean_path(t) for t in (context.get("required_test_ids") or []) if str(t).strip()]

    raw_results = context.get("test_results")
    if not required:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.UNVERIFIED,
            "test_outcomes",
            (),
            "no test ids were supplied",
        )
    if not isinstance(raw_results, Mapping) or not raw_results:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.UNVERIFIED,
            "test_outcomes",
            (),
            "no test_results were supplied",
        )
    results = {_clean_path(k): v for k, v in raw_results.items()}
    missing = [test_id for test_id in required if test_id not in results]
    # Hardened check: require explicit pass (True or non-lexical VerificationResult(passed=True)).
    failed = [
        test_id
        for test_id in required
        if test_id in results and not _is_explicit_pass(results.get(test_id))
    ]
    evidence = tuple(f"test:{test_id}" for test_id in required)
    if missing or failed:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.FAILED,
            "test_outcomes",
            evidence,
            f"missing={missing} failed={failed}",
        )
    return RequirementEvidence(
        requirement_id,
        RequirementStatus.VERIFIED,
        "test_outcomes",
        evidence,
        "all required test ids passed",
    )


def _check_dependencies(
    requirement_id: str, requirement: Any, context: Mapping[str, Any]
) -> RequirementEvidence:
    observed_raw = context.get("observed_dependencies")
    if observed_raw is None:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.UNVERIFIED,
            "dependency_restrictions",
            (),
            "observed_dependencies was not supplied",
        )
    observed = [_clean_path(name) for name in observed_raw if str(name).strip()]
    req_forbidden = _field(requirement, "forbidden_dependencies")
    forbidden = set(
        req_forbidden if req_forbidden is not None else (context.get("forbidden_dependencies") or [])
    )
    allowed = _field(requirement, "allowed_dependencies", context.get("allowed_dependencies"))

    text_lower = str(_field(requirement, "text", "") or "").lower()
    # If the requirement states "no new dependencies" / "stdlib only" and neither
    # forbidden nor allowed lists are given, enforce that every observed dependency
    # is a stdlib module (prefixed with "stdlib:" or in Python's stdlib allowlist).
    stdlib_only = any(
        phrase in text_lower
        for phrase in (
            "no new dependenc",
            "not add new dependenc",
            "without external",
            "stdlib only",
            "standard library only",
        )
    )
    if not observed and not forbidden and allowed is None and not stdlib_only:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.UNVERIFIED,
            "dependency_restrictions",
            (),
            "no dependency policy or observed dependencies supplied",
        )
    violations = [name for name in observed if name in forbidden]
    if allowed is not None:
        allowed_set = {_clean_path(a) for a in allowed}
        violations += [name for name in observed if name not in allowed_set]
    elif stdlib_only:
        from .vexx import Vexx35Controller

        allowed_stdlib = Vexx35Controller.ALLOWED_STDLIB_MODULES
        for name in observed:
            bare = name.split(":", 1)[-1].split(".")[0]
            if not name.startswith("stdlib:") and bare not in allowed_stdlib:
                violations.append(name)

    evidence = tuple(f"dep:{name}" for name in observed) or ("dep:none_added",)
    if violations:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.FAILED,
            "dependency_restrictions",
            evidence,
            f"forbidden dependencies: {sorted(set(violations))}",
        )
    return RequirementEvidence(
        requirement_id,
        RequirementStatus.VERIFIED,
        "dependency_restrictions",
        evidence,
        "no forbidden dependency was added",
    )


def _check_output_schema(
    requirement_id: str, requirement: Any, context: Mapping[str, Any]
) -> RequirementEvidence:
    targets = extract_requirement_targets(_field(requirement, "text", ""))
    expected = (
        _field(requirement, "expected_format")
        or targets["expected_format"]
        or context.get("expected_format")
    )
    observed = context.get("observed_format")
    if expected is None or observed is None:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.UNVERIFIED,
            "output_schema",
            (),
            "expected and observed output formats are both required",
        )
    evidence = (f"format:{observed}",)
    if str(expected).strip().lower() != str(observed).strip().lower():
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.FAILED,
            "output_schema",
            evidence,
            f"expected {expected!r}, observed {observed!r}",
        )
    return RequirementEvidence(
        requirement_id,
        RequirementStatus.VERIFIED,
        "output_schema",
        evidence,
        f"output format {observed!r} matches",
    )


def _check_complexity(
    requirement_id: str, requirement: Any, context: Mapping[str, Any]
) -> RequirementEvidence:
    targets = extract_requirement_targets(_field(requirement, "text", ""))
    expected = (
        _field(requirement, "expected_complexity")
        or targets["expected_complexity"]
        or context.get("expected_complexity")
    )
    measured = context.get("measured_complexity")
    if expected is None or measured is None:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.UNVERIFIED,
            "complexity",
            (),
            "complexity needs both an expected bound and a measurement",
        )
    evidence = (f"complexity:{measured}",)
    exp_rank = parse_complexity_rank(str(expected))
    meas_rank = parse_complexity_rank(str(measured))
    if exp_rank is not None and meas_rank is not None:
        if meas_rank > exp_rank:
            return RequirementEvidence(
                requirement_id,
                RequirementStatus.FAILED,
                "complexity",
                evidence,
                f"expected at most {expected!r} (rank {exp_rank}), measured {measured!r} (rank {meas_rank})",
            )
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.VERIFIED,
            "complexity",
            evidence,
            f"measured complexity {measured!r} satisfies bound {expected!r}",
        )
    if str(expected).replace(" ", "").lower() != str(measured).replace(" ", "").lower():
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.FAILED,
            "complexity",
            evidence,
            f"expected {expected!r}, measured {measured!r}",
        )
    return RequirementEvidence(
        requirement_id,
        RequirementStatus.VERIFIED,
        "complexity",
        evidence,
        f"measured complexity {measured!r} matches the requirement",
    )


_CHECKERS = {
    "api_signature": _check_api_signature,
    "path_presence": _check_path_presence,
    "test_outcomes": _check_test_outcomes,
    "dependency_restrictions": _check_dependencies,
    "output_schema": _check_output_schema,
    "complexity": _check_complexity,
}


def _verify_one(requirement: Any, context: Mapping[str, Any]) -> RequirementEvidence:
    requirement_id = str(_field(requirement, "requirement_id", "") or "")
    verifier_ids = list(_field(requirement, "verifier_ids", ()) or ())
    if not verifier_ids:
        text = str(_field(requirement, "text", "") or "")
        if text:
            verifier_ids = requirement_verifiers_for_text(text)
    if not verifier_ids:
        return RequirementEvidence(
            requirement_id,
            RequirementStatus.UNVERIFIED,
            "none",
            (),
            "no mechanical verifier is declared for this requirement",
        )
    results: List[RequirementEvidence] = []
    for verifier_id in verifier_ids:
        checker = _CHECKERS.get(verifier_id)
        if checker is None:
            results.append(
                RequirementEvidence(
                    requirement_id,
                    RequirementStatus.UNVERIFIED,
                    str(verifier_id),
                    (),
                    f"unknown verifier {verifier_id!r}",
                )
            )
            continue
        results.append(checker(requirement_id, requirement, context))

    statuses = {result.status for result in results}
    evidence_ids = tuple(
        evidence_id for result in results for evidence_id in result.evidence_ids
    )
    detail = "; ".join(result.detail for result in results if result.detail)
    if RequirementStatus.FAILED in statuses:
        status = RequirementStatus.FAILED
    elif statuses == {RequirementStatus.VERIFIED}:
        status = RequirementStatus.VERIFIED
    else:
        status = RequirementStatus.UNVERIFIED
    return RequirementEvidence(
        requirement_id, status, "+".join(verifier_ids), evidence_ids, detail
    )


def verify_requirements(
    requirements: Sequence[Any], context: Mapping[str, Any]
) -> Dict[str, RequirementEvidence]:
    """Verify every requirement in ``requirements`` against ``context``."""
    report: Dict[str, RequirementEvidence] = {}
    for requirement in requirements:
        result = _verify_one(requirement, context)
        if not result.requirement_id:
            continue
        report[result.requirement_id] = result
    return report


#: Requirement verifiers map onto the agent ledger's evidence taxonomy.
_EVIDENCE_KIND_BY_VERIFIER = {
    "api_signature": "contract",
    "path_presence": "diff",
    "test_outcomes": "test",
    "dependency_restrictions": "contract",
    "output_schema": "contract",
    "complexity": "manual_review",
}


def apply_requirement_evidence(
    runtime: Any,
    evidence_by_requirement: Mapping[str, RequirementEvidence],
) -> None:
    """Push verifier outcomes into an ``AgentRuntime`` requirement ledger."""
    for requirement_id, evidence in evidence_by_requirement.items():
        primary = str(evidence.verifier).split("+")[0]
        kind = _EVIDENCE_KIND_BY_VERIFIER.get(primary, "manual_review")
        runtime.record_requirement_evidence(
            requirement_id,
            evidence.status.value,
            kind,
            list(evidence.evidence_ids),
            note=evidence.detail,
        )


def requirement_verifiers_for_text(text: str) -> List[str]:
    """Declare which mechanical checks a requirement text maps onto."""
    lower = str(text).lower()
    targets = extract_requirement_targets(text)
    verifiers: List[str] = []
    if any(word in lower for word in ("signature", "public api", "arity", "parameter")):
        verifiers.append("api_signature")
    if targets["paths"] or any(
        word in lower for word in ("file", "path", "add a new", "create a")
    ):
        verifiers.append("path_presence")
    if targets["test_ids"] or any(
        word in lower for word in ("test", "pytest", "unittest")
    ):
        verifiers.append("test_outcomes")
    if any(
        word in lower
        for word in ("dependenc", "stdlib", "standard library", "import", "external librar")
    ):
        verifiers.append("dependency_restrictions")
    if targets["expected_format"] or any(
        word in lower for word in ("json", "schema", "output format")
    ):
        verifiers.append("output_schema")
    if targets["expected_complexity"] or any(
        word in lower for word in ("o(", "complexity", "linear time", "constant time", "runtime")
    ):
        verifiers.append("complexity")
    return verifiers


def classify_requirement_kind(text: str) -> str:
    """Classify a requirement as functional / nonfunctional / compatibility / safety / output."""
    lower = str(text).lower()
    if any(word in lower for word in ("must not", "don't", "never", "forbid", "safety", "security")):
        return "safety"
    if any(word in lower for word in ("backwards", "compatible", "preserve", "keep", "unchanged")):
        return "compatibility"
    if any(word in lower for word in ("o(", "complexity", "latency", "memory", "performance")):
        return "nonfunctional"
    if any(word in lower for word in ("output", "format", "json", "schema")):
        return "output"
    return "functional"


def verify_task_spec_requirements(
    spec: Any, context: Mapping[str, Any]
) -> Dict[str, RequirementEvidence]:
    """Verify ``TaskSpec.requirements``/``constraints`` using typed verifiers.

    Requirements without declared verifiers are reported ``UNVERIFIED`` rather than
    being promoted by lexical overlap.
    """
    from .understanding import TaskSpec

    if not isinstance(spec, TaskSpec):
        raise TypeError("spec must be a TaskSpec")

    requirements: List[Dict[str, Any]] = []
    for index, text in enumerate(spec.requirements):
        requirements.append(
            {
                "requirement_id": f"requirement_{index}",
                "text": text,
                "verifier_ids": requirement_verifiers_for_text(text),
            }
        )
    for index, text in enumerate(spec.constraints):
        requirements.append(
            {
                "requirement_id": f"constraint_{index}",
                "text": text,
                "verifier_ids": requirement_verifiers_for_text(text),
            }
        )
    return verify_requirements(requirements, context)
