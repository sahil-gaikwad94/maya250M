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
]

STRATEGIES = (
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
        if spec.task_type == "math":
            return "derive_then_verify"
        if spec.task_type in ("research", "question", "explanation"):
            return "retrieve_then_answer"
        if min_loops > 1:
            return "direct_answer"
        return "direct_answer"
