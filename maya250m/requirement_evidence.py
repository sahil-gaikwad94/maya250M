"""Typed, evidence-backed requirement verification (P0.4).

A task requirement is only *verified* when a mechanical verifier reports the exact
evidence that proves it. Word overlap between the request and the answer is at most
``ADDRESSED``; anything that cannot be checked mechanically is ``UNVERIFIED``.

The verifiers here are deliberately narrow and data-driven: each takes the observed
facts a caller already has (signatures, paths, test outcomes, imports, output
format, measured complexity) and returns a :class:`RequirementEvidence`. Nothing is
inferred from a passing command that did not check the requirement itself.
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


def _clean_path(value: str) -> str:
    return _NORMALIZE_PATH.sub("/", str(value).strip()).lstrip("./")


def _verified(verifier: str, evidence_ids: Iterable[str], detail: str) -> RequirementEvidence:
    return RequirementEvidence("", RequirementStatus.VERIFIED, verifier, tuple(evidence_ids), detail)


def _check_api_signature(requirement_id: str, context: Mapping[str, Any]) -> RequirementEvidence:
    expected = context.get("expected_signatures") or {}
    observed = context.get("observed_signatures") or {}
    if not expected:
        return RequirementEvidence(
            requirement_id, RequirementStatus.UNVERIFIED, "api_signature", (),
            "no expected API signatures were supplied",
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
            requirement_id, RequirementStatus.FAILED, "api_signature", evidence,
            f"missing={missing} changed={changed}",
        )
    return RequirementEvidence(
        requirement_id, RequirementStatus.VERIFIED, "api_signature", evidence,
        "all expected signatures preserved",
    )


def _check_path_presence(requirement_id: str, context: Mapping[str, Any]) -> RequirementEvidence:
    expected = [_clean_path(p) for p in (context.get("expected_paths") or [])]
    present = {_clean_path(p) for p in (context.get("present_paths") or [])}
    if not expected:
        return RequirementEvidence(
            requirement_id, RequirementStatus.UNVERIFIED, "path_presence", (),
            "no expected paths were supplied",
        )
    missing = [path for path in expected if path not in present]
    evidence = tuple(f"path:{path}" for path in expected)
    if missing:
        return RequirementEvidence(
            requirement_id, RequirementStatus.FAILED, "path_presence", evidence,
            f"missing paths: {missing}",
        )
    return RequirementEvidence(
        requirement_id, RequirementStatus.VERIFIED, "path_presence", evidence,
        "all required paths exist",
    )


def _check_test_outcomes(requirement_id: str, context: Mapping[str, Any]) -> RequirementEvidence:
    required = list(context.get("required_test_ids") or [])
    results = dict(context.get("test_results") or {})
    if not required:
        return RequirementEvidence(
            requirement_id, RequirementStatus.UNVERIFIED, "test_outcomes", (),
            "no test ids were supplied",
        )
    missing = [test_id for test_id in required if test_id not in results]
    failed = [test_id for test_id in required if results.get(test_id) is False]
    evidence = tuple(f"test:{test_id}" for test_id in required)
    if missing or failed:
        return RequirementEvidence(
            requirement_id, RequirementStatus.FAILED, "test_outcomes", evidence,
            f"missing={missing} failed={failed}",
        )
    return RequirementEvidence(
        requirement_id, RequirementStatus.VERIFIED, "test_outcomes", evidence,
        "all required test ids passed",
    )


def _check_dependencies(requirement_id: str, context: Mapping[str, Any]) -> RequirementEvidence:
    observed = [_clean_path(name) for name in (context.get("observed_dependencies") or [])]
    forbidden = set(context.get("forbidden_dependencies") or [])
    allowed = context.get("allowed_dependencies")
    if not observed:
        return RequirementEvidence(
            requirement_id, RequirementStatus.UNVERIFIED, "dependency_restrictions", (),
            "no observed dependencies were supplied",
        )
    violations = [name for name in observed if name in forbidden]
    if allowed is not None:
        allowed_set = set(allowed)
        violations += [name for name in observed if name not in allowed_set]
    evidence = tuple(f"dep:{name}" for name in observed)
    if violations:
        return RequirementEvidence(
            requirement_id, RequirementStatus.FAILED, "dependency_restrictions", evidence,
            f"forbidden dependencies: {sorted(set(violations))}",
        )
    return RequirementEvidence(
        requirement_id, RequirementStatus.VERIFIED, "dependency_restrictions", evidence,
        "no forbidden dependency was added",
    )


def _check_output_schema(requirement_id: str, context: Mapping[str, Any]) -> RequirementEvidence:
    expected = context.get("expected_format")
    observed = context.get("observed_format")
    if expected is None or observed is None:
        return RequirementEvidence(
            requirement_id, RequirementStatus.UNVERIFIED, "output_schema", (),
            "expected and observed output formats are both required",
        )
    evidence = (f"format:{observed}",)
    if str(expected).lower() != str(observed).lower():
        return RequirementEvidence(
            requirement_id, RequirementStatus.FAILED, "output_schema", evidence,
            f"expected {expected!r}, observed {observed!r}",
        )
    return RequirementEvidence(
        requirement_id, RequirementStatus.VERIFIED, "output_schema", evidence,
        f"output format {observed!r} matches",
    )


def _check_complexity(requirement_id: str, context: Mapping[str, Any]) -> RequirementEvidence:
    expected = context.get("expected_complexity")
    measured = context.get("measured_complexity")
    if expected is None or measured is None:
        return RequirementEvidence(
            requirement_id, RequirementStatus.UNVERIFIED, "complexity", (),
            "complexity needs both an expected bound and a measurement",
        )
    evidence = (f"complexity:{measured}",)
    if str(expected).replace(" ", "").lower() != str(measured).replace(" ", "").lower():
        return RequirementEvidence(
            requirement_id, RequirementStatus.FAILED, "complexity", evidence,
            f"expected {expected!r}, measured {measured!r}",
        )
    return RequirementEvidence(
        requirement_id, RequirementStatus.VERIFIED, "complexity", evidence,
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


def _field(requirement: Any, name: str, default: Any = None) -> Any:
    if isinstance(requirement, Mapping):
        return requirement.get(name, default)
    return getattr(requirement, name, default)


def _verify_one(requirement: Any, context: Mapping[str, Any]) -> RequirementEvidence:
    requirement_id = str(_field(requirement, "requirement_id", "") or "")
    verifier_ids = list(_field(requirement, "verifier_ids", ()) or ())
    if not verifier_ids:
        return RequirementEvidence(
            requirement_id, RequirementStatus.UNVERIFIED, "none", (),
            "no mechanical verifier is declared for this requirement",
        )
    results: List[RequirementEvidence] = []
    for verifier_id in verifier_ids:
        checker = _CHECKERS.get(verifier_id)
        if checker is None:
            results.append(
                RequirementEvidence(
                    requirement_id, RequirementStatus.UNVERIFIED, str(verifier_id), (),
                    f"unknown verifier {verifier_id!r}",
                )
            )
            continue
        results.append(checker(requirement_id, context))

    statuses = {result.status for result in results}
    evidence_ids = tuple(
        evidence_id for result in results for evidence_id in result.evidence_ids
    )
    detail = "; ".join(result.detail for result in results if result.detail)
    if RequirementStatus.FAILED in statuses:
        status = RequirementStatus.FAILED
    elif statuses == {RequirementStatus.VERIFIED}:
        status = RequirementStatus.VERIFIED
    elif RequirementStatus.VERIFIED in statuses:
        status = RequirementStatus.UNVERIFIED
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
    """Heuristically declare which mechanical checks a requirement is amenable to."""
    lower = str(text).lower()
    verifiers: List[str] = []
    if any(word in lower for word in ("signature", "public api", "arity", "parameter")):
        verifiers.append("api_signature")
    if any(word in lower for word in ("file", "path", "add a new", "create a")):
        verifiers.append("path_presence")
    if any(word in lower for word in ("test", "pytest", "unittest")):
        verifiers.append("test_outcomes")
    if any(word in lower for word in ("dependenc", "stdlib", "standard library", "import")):
        verifiers.append("dependency_restrictions")
    if any(word in lower for word in ("json", "schema", "output format")):
        verifiers.append("output_schema")
    if any(word in lower for word in ("o(", "complexity", "linear time", "runtime")):
        verifiers.append("complexity")
    return verifiers


def classify_requirement_kind(text: str) -> str:
    """Classify a requirement as functional / nonfunctional / compatibility / safety."""
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
