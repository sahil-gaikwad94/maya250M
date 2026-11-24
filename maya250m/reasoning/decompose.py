"""Dependency-DAG problem decomposition and strategy selection.

A decomposition is a small DAG of subproblems with explicit prerequisites, so a
bounded agent cannot silently skip a step (e.g. "patch before reproducing") and an
evaluation can score *which* subproblem failed. Cycle detection is fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..understanding import TaskSpec

__all__ = [
    "Subproblem",
    "Decomposition",
    "DecompositionError",
    "decompose_task",
    "StrategySelector",
    "STRATEGIES",
    "counterexample_first_plan",
    "certify_universal_claim",
    "is_conjecture_or_universal_claim",
    "RESEARCH_STRATEGY",
]

#: CONJECTURE -> FALSIFY -> PROVE strategy for open mathematical/empirical claims.
RESEARCH_STRATEGY = "falsify_then_prove"

STRATEGIES = (
    RESEARCH_STRATEGY,
    "localize_then_patch",
    "reproduce_then_patch",
    "derive_then_verify",
    "retrieve_then_answer",
    "direct_answer",
    "abstain",
)


class DecompositionError(ValueError):
    """Raised for cyclic or malformed decomposition DAGs."""


@dataclass
class Subproblem:
    """One node of the decomposition DAG."""

    subproblem_id: str
    description: str
    kind: str = "generic"  # localize | reproduce | patch | verify | derive | answer
    depends_on: List[str] = field(default_factory=list)
    acceptance: str = ""
    status: str = "pending"  # pending | done | failed | skipped


@dataclass
class Decomposition:
    """A validated DAG of subproblems with a deterministic topological order."""

    subproblems: List[Subproblem]
    strategy: str = "direct_answer"

    def __post_init__(self) -> None:
        ids = [s.subproblem_id for s in self.subproblems]
        if len(set(ids)) != len(ids):
            raise DecompositionError("duplicate subproblem ids")
        known = set(ids)
        for sub in self.subproblems:
            missing = [d for d in sub.depends_on if d not in known]
            if missing:
                raise DecompositionError(
                    f"subproblem {sub.subproblem_id!r} depends on unknown {missing}"
                )
        self.topological_order()  # validates acyclicity

    def by_id(self, subproblem_id: str) -> Subproblem:
        for sub in self.subproblems:
            if sub.subproblem_id == subproblem_id:
                return sub
        raise KeyError(subproblem_id)

    def topological_order(self) -> List[str]:
        """Kahn's algorithm with lexicographic tie-breaking (deterministic)."""
        indegree = {s.subproblem_id: len(s.depends_on) for s in self.subproblems}
        children: Dict[str, List[str]] = {s.subproblem_id: [] for s in self.subproblems}
        for sub in self.subproblems:
            for dep in sub.depends_on:
                children[dep].append(sub.subproblem_id)
        ready = sorted([k for k, v in indegree.items() if v == 0])
        order: List[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for child in sorted(children[current]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(order) != len(self.subproblems):
            cycle = sorted(k for k, v in indegree.items() if v > 0)
            raise DecompositionError(f"dependency cycle among subproblems: {cycle}")
        return order

    def ready_subproblems(self) -> List[Subproblem]:
        done = {s.subproblem_id for s in self.subproblems if s.status in ("done", "skipped")}
        return [
            sub
            for sub in self.subproblems
            if sub.status == "pending" and all(d in done for d in sub.depends_on)
        ]

    def mark(self, subproblem_id: str, status: str) -> None:
        if status not in ("pending", "done", "failed", "skipped"):
            raise DecompositionError(f"invalid subproblem status {status!r}")
        self.by_id(subproblem_id).status = status

    def progress(self) -> Tuple[int, int]:
        done = sum(1 for s in self.subproblems if s.status in ("done", "skipped"))
        return done, len(self.subproblems)

    def describe(self) -> str:
        lines = [f"STRATEGY: {self.strategy}"]
        for sid in self.topological_order():
            sub = self.by_id(sid)
            deps = ",".join(sub.depends_on) if sub.depends_on else "-"
            lines.append(f"[{sub.status}] {sid} ({sub.kind}) <- {deps}: {sub.description}")
        return "\n".join(lines)


_CONJECTURE_MARKERS = (
    "for all ",
    "for every ",
    "is every ",
    "every ",
    "always ",
    "never ",
    "conjecture",
    "counterexample",
    "hypothesis",
    "falsif",
    "disprove",
    "prime",
    "n^2",
    "n**2",
)


def is_conjecture_or_universal_claim(spec: TaskSpec) -> bool:
    """True when ``spec`` states an empirical/mathematical conjecture or universal claim."""
    text = f"{spec.objective} {' '.join(spec.requirements)} {' '.join(spec.constraints)}".lower()
    if any(marker in text for marker in _CONJECTURE_MARKERS):
        return True
    return False


def decompose_task(
    spec: TaskSpec,
    repo_files: Optional[Sequence[str]] = None,
    strategies: Optional[Iterable[str]] = None,
) -> Decomposition:
    """Build a dependency-DAG decomposition from a ``TaskSpec``.

    Code tasks get localization -> reproduction -> patch -> verification.
    Mathematical tasks get derivation -> numeric/symbolic check.
    Research/answer tasks get retrieval -> evidence assembly -> answer.
    """
    allowed = set(strategies or STRATEGIES)
    if spec.task_type in ("bug_fix", "code_edit", "refactor", "test_writing", "code_generation"):
        strategy = (
            "reproduce_then_patch"
            if spec.task_type == "bug_fix" and allowed >= {"reproduce_then_patch"}
            else "localize_then_patch"
        )
        if strategy not in allowed:
            strategy = "localize_then_patch" if "localize_then_patch" in allowed else "direct_answer"
        subproblems = [
            Subproblem(
                "localize",
                "Localize the files and symbols that own the behaviour"
                + (f" (candidates: {', '.join(spec.files_mentioned[:3])})" if spec.files_mentioned else ""),
                kind="localize",
                acceptance="top-k files and symbols identified with evidence",
            ),
            Subproblem(
                "reproduce",
                "Observe the current failure (run the failing test or read the relevant span)",
                kind="reproduce",
                depends_on=["localize"],
                acceptance="a concrete failing observation is recorded",
            ),
            Subproblem(
                "patch",
                "Produce a minimal content-anchored patch that satisfies the objective",
                kind="patch",
                depends_on=["reproduce"] if strategy == "reproduce_then_patch" else ["localize"],
                acceptance="patch applies uniquely and parses",
            ),
            Subproblem(
                "verify",
                "Run target + regression tests and cross-file contract checks; cover every constraint",
                kind="verify",
                depends_on=["patch"],
                acceptance="all checks pass and constraints are covered",
            ),
        ]
    elif (
        spec.task_type in ("research", "math", "question")
        and is_conjecture_or_universal_claim(spec)
        and RESEARCH_STRATEGY in allowed
    ):
        return counterexample_first_plan(spec.objective)
    elif spec.task_type == "math":
        strategy = "derive_then_verify"
        subproblems = [
            Subproblem("parse", "Extract the exact quantities, givens and unknowns", kind="derive"),
            Subproblem(
                "derive",
                "Perform the derivation symbolically, step by step",
                kind="derive",
                depends_on=["parse"],
                acceptance="each step is justified",
            ),
            Subproblem(
                "verify",
                "Verify the result independently (CAS or numeric substitution)",
                kind="verify",
                depends_on=["derive"],
                acceptance="independent check agrees",
            ),
        ]
    elif spec.task_type in ("research", "question", "explanation"):
        strategy = "retrieve_then_answer"
        subproblems = [
            Subproblem("retrieve", "Retrieve the evidence needed for the answer", kind="answer"),
            Subproblem(
                "assemble",
                "Assemble token-budgeted evidence with provenance",
                kind="answer",
                depends_on=["retrieve"],
                acceptance="every claim is attributable",
            ),
            Subproblem(
                "answer",
                "Write the answer and flag anything unsupported",
                kind="answer",
                depends_on=["assemble"],
                acceptance="no unsupported claim presented as fact",
            ),
        ]
    else:
        strategy = "direct_answer" if "direct_answer" in allowed else "abstain"
        subproblems = [
            Subproblem(
                "answer",
                "Answer directly, or abstain if the objective is underspecified",
                kind="answer",
                acceptance="either a supported answer or an explicit abstention",
            )
        ]

    return Decomposition(subproblems=subproblems, strategy=strategy)


def counterexample_first_plan(
    claim: str,
    sweep_n: int = 15,
    falsify_n: int = 500,
) -> Decomposition:
    """Structured CONJECTURE -> FALSIFY -> PROVE decomposition for a claim.

    Stage 1 sweeps small ``n`` and records exact observations, stage 2 fits a
    symbolic pattern, stage 3 runs an adversarial falsification sweep, and stage 4
    refuses universality without an algebraic identity proof. The DAG encodes the
    order so a bounded agent cannot skip falsification and jump to a proof claim.
    """
    if sweep_n < 1 or falsify_n < sweep_n:
        raise DecompositionError("need 1 <= sweep_n <= falsify_n")
    claim_text = str(claim or "").strip() or "the stated claim"
    return Decomposition(
        strategy=RESEARCH_STRATEGY,
        subproblems=[
            Subproblem(
                "sweep",
                f"<experiment>#1: enumerate n in 1..{sweep_n} and emit exact "
                f"[OBS: ...] values for {claim_text}",
                kind="derive",
                acceptance="exact observed values, no rounding",
            ),
            Subproblem(
                "fit",
                "<<...>> / <experiment>#2: fit a polynomial, rational or recurrence "
                "candidate and check it reproduces every observation",
                kind="derive",
                depends_on=["sweep"],
                acceptance="closed form matches all observed values",
            ),
            Subproblem(
                "falsify",
                f"<experiment>#3: adversarial counterexample search up to N={falsify_n}",
                kind="verify",
                depends_on=["fit"],
                acceptance="no counterexample found, or it is reported explicitly",
            ),
            Subproblem(
                "prove",
                "Symbolic identity proof (safe_sympy(lhs - rhs) == 0) before any "
                "\"for all n\" claim",
                kind="verify",
                depends_on=["falsify"],
                acceptance="universality is claimed only with an algebraic identity",
            ),
        ],
    )


def certify_universal_claim(
    finite_checks: int,
    counterexample: Optional[object] = None,
    identity_verified: bool = False,
) -> Dict[str, object]:
    """Decide what a claim's evidence actually licenses.

    A counterexample refutes; an algebraic identity proves; anything else is a
    conjecture supported only by finitely many checks and may never be reported as
    proven for all ``n``.
    """
    checks = int(finite_checks)
    if checks < 0:
        raise ValueError("finite_checks must be non-negative")
    if counterexample is not None:
        return {
            "verdict": "REFUTED",
            "may_claim_for_all_n": False,
            "explanation": f"counterexample found: {counterexample!r}",
        }
    if identity_verified:
        return {
            "verdict": "PROVEN",
            "may_claim_for_all_n": True,
            "explanation": "symbolic identity verified (lhs - rhs == 0)",
        }
    return {
        "verdict": "CONJECTURE_ONLY",
        "may_claim_for_all_n": False,
        "explanation": f"only {checks} finite checks; no algebraic identity proof",
    }


#: Ambiguity flags that are disqualifying on their own: the objective cannot be
#: executed without asking a clarifying question.
CRITICAL_AMBIGUITY_FLAGS = (
    "underspecified_criterion",
    "prompt_too_short",
    "multiple_alternatives",
)


class StrategySelector:
    """Pick a solve strategy from spec signals, with abstention as a first-class option."""

    def __init__(self, ambiguity_threshold: int = 3, confidence_threshold: float = 0.25):
        self.ambiguity_threshold = int(ambiguity_threshold)
        self.confidence_threshold = float(confidence_threshold)

    def select(self, spec: TaskSpec, min_loops: int = 1) -> str:
        if any(flag in CRITICAL_AMBIGUITY_FLAGS for flag in spec.ambiguity_flags):
            return "abstain"
        if len(spec.ambiguity_flags) >= self.ambiguity_threshold:
            return "abstain"
        if spec.confidence < self.confidence_threshold:
            return "abstain"
        if spec.task_type in ("bug_fix", "code_edit", "refactor", "test_writing", "code_generation"):
            return "reproduce_then_patch" if spec.task_type == "bug_fix" else "localize_then_patch"
        if spec.task_type in ("research", "math", "question") and is_conjecture_or_universal_claim(spec):
            return RESEARCH_STRATEGY
        if spec.task_type == "math":
            return "derive_then_verify"
        if spec.task_type in ("research", "question", "explanation"):
            return "retrieve_then_answer"
        if min_loops > 1:
            return "direct_answer"
        return "direct_answer"
