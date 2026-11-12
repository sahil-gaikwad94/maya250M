"""Reasoning and working-memory state machine for grounded problem solving.

``ReasoningState`` is the single source of truth for *where* a bounded solve is,
*what* has been established, and *what* has already failed. The agent runtime,
the verifier and the trainers all read and write this state instead of relying on
raw chat history.

The phase graph is the five-phase engineering protocol:

    UNDERSTAND_AND_LOCALIZE -> REPRODUCE_OR_INSPECT -> HYPOTHESIZE_AND_PATCH
        -> VERIFY_AND_TEST -> FINAL_REVIEW -> DONE

with the only back-edge being ``VERIFY_AND_TEST -> HYPOTHESIZE_AND_PATCH`` after
a rejected hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..verification import VerificationResult

__all__ = [
    "PHASES",
    "PHASE_TRANSITIONS",
    "ReasoningState",
    "ReasoningStateError",
]

PHASE_UNDERSTAND = "UNDERSTAND_AND_LOCALIZE"
PHASE_REPRODUCE = "REPRODUCE_OR_INSPECT"
PHASE_HYPOTHESIZE = "HYPOTHESIZE_AND_PATCH"
PHASE_VERIFY = "VERIFY_AND_TEST"
PHASE_REVIEW = "FINAL_REVIEW"
PHASE_DONE = "DONE"
PHASE_FAILED = "FAILED"

PHASES = (
    PHASE_UNDERSTAND,
    PHASE_REPRODUCE,
    PHASE_HYPOTHESIZE,
    PHASE_VERIFY,
    PHASE_REVIEW,
    PHASE_DONE,
    PHASE_FAILED,
)

PHASE_TRANSITIONS = {
    PHASE_UNDERSTAND: (PHASE_REPRODUCE,),
    PHASE_REPRODUCE: (PHASE_HYPOTHESIZE,),
    PHASE_HYPOTHESIZE: (PHASE_VERIFY,),
    PHASE_VERIFY: (PHASE_HYPOTHESIZE, PHASE_REVIEW),
    PHASE_REVIEW: (PHASE_DONE, PHASE_HYPOTHESIZE),
    PHASE_DONE: (),
    PHASE_FAILED: (),
}

MAX_REJECTED_APPROACHES = 8
MAX_EVIDENCE = 64


class ReasoningStateError(RuntimeError):
    """Raised on an illegal phase transition or budget misuse."""


@dataclass
class Hypothesis:
    """One candidate fix or answer, with the evidence that refuted/accepted it."""

    description: str
    patch_targets: List[str] = field(default_factory=list)
    verification: Optional[Dict[str, Any]] = None
    rejected_reason: Optional[str] = None


@dataclass
class ReasoningState:
    """Mutable, serializable reasoning/work state for one bounded episode."""

    objective: str
    task_type: str = "unknown"
    constraints: List[str] = field(default_factory=list)
    phase: str = PHASE_UNDERSTAND
    max_steps: int = 12
    steps_used: int = 0
    subgoals: List[str] = field(default_factory=list)
    completed_subgoals: List[str] = field(default_factory=list)
    known_facts: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    active_symbol_contracts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rejected_approaches: List[str] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    constraint_status: Dict[str, bool] = field(default_factory=dict)
    last_failure_category: Optional[str] = None
    observation_cursor: int = 0
    finished: bool = False

    # ---------------------------------------------------------------- lifecycle

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ReasoningStateError(f"unknown phase {self.phase!r}")
        if int(self.max_steps) < 1:
            raise ReasoningStateError("max_steps must be >= 1")
        if not self.constraint_status:
            self.constraint_status = {f"constraint_{i}": False for i in range(len(self.constraints))}

    @property
    def budget_remaining(self) -> int:
        return max(0, int(self.max_steps) - int(self.steps_used))

    @property
    def budget_exhausted(self) -> bool:
        return self.budget_remaining <= 0

    def assert_can_act(self) -> None:
        if self.finished:
            raise ReasoningStateError("episode is finished; no further actions allowed")
        if self.budget_exhausted:
            raise ReasoningStateError("step budget exhausted")

    # ------------------------------------------------------------------ updates

    def consume_step(self, label: str = "") -> None:
        """Charge one tool/action step against the episode budget."""
        self.assert_can_act()
        self.steps_used += 1
        if label:
            self.known_facts.append(f"step {self.steps_used}: {label}")
            self._trim_known_facts()

    def _trim_known_facts(self) -> None:
        if len(self.known_facts) > 24:
            # evict the oldest *observation-derived* facts, keep pinned entries
            pinned = [f for f in self.known_facts if f.startswith("pinned:")]
            rolling = [f for f in self.known_facts if not f.startswith("pinned:")]
            self.known_facts = pinned + rolling[-(24 - len(pinned)) :]

    def pin_fact(self, fact: str) -> None:
        entry = f"pinned: {fact}"
        if entry not in self.known_facts:
            self.known_facts.append(entry)

    def add_evidence(self, evidence_id: str) -> None:
        if evidence_id and evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)
            if len(self.evidence_ids) > MAX_EVIDENCE:
                self.evidence_ids = self.evidence_ids[-MAX_EVIDENCE:]

    def set_symbol_contract(self, qualname: str, contract: Dict[str, Any]) -> None:
        self.active_symbol_contracts[qualname] = dict(contract)

    def complete_subgoal(self, subgoal: str) -> None:
        if subgoal in self.subgoals and subgoal not in self.completed_subgoals:
            self.completed_subgoals.append(subgoal)

    def record_constraint_coverage(self, coverage: Dict[str, bool]) -> None:
        for key, value in coverage.items():
            self.constraint_status[key] = bool(value)

    def uncovered_constraints(self) -> List[str]:
        return [k for k, covered in self.constraint_status.items() if not covered]

    # --------------------------------------------------------------- hypotheses

    def record_hypothesis(self, description: str, patch_targets: Optional[Iterable[str]] = None) -> int:
        self.hypotheses.append(
            Hypothesis(description=description, patch_targets=list(patch_targets or []))
        )
        return len(self.hypotheses) - 1

    def record_verification(self, result: VerificationResult) -> None:
        """Attach the latest verification outcome and evolve the phase."""
        if not self.hypotheses:
            raise ReasoningStateError("record_verification called with no hypothesis")
        hypothesis = self.hypotheses[-1]
        hypothesis.verification = result.to_dict()
        if result.passed:
            self.last_failure_category = None
            self.transition(PHASE_REVIEW)
            return

        self.last_failure_category = result.failure_category
        hypothesis.rejected_reason = result.message or str(result.failure_category)
        reason = f"{result.failure_category}: {result.message}".strip()
        if reason and reason not in self.rejected_approaches:
            self.rejected_approaches.append(reason)
            if len(self.rejected_approaches) > MAX_REJECTED_APPROACHES:
                self.rejected_approaches = self.rejected_approaches[-MAX_REJECTED_APPROACHES:]
        if self.phase == PHASE_VERIFY:
            self.transition(PHASE_HYPOTHESIZE)

    # -------------------------------------------------------------- transitions

    def transition(self, target: str) -> str:
        if target not in PHASES:
            raise ReasoningStateError(f"unknown target phase {target!r}")
        if target == self.phase:
            return self.phase
        allowed = PHASE_TRANSITIONS[self.phase]
        if target not in allowed:
            raise ReasoningStateError(
                f"illegal reasoning phase transition {self.phase} -> {target}; "
                f"allowed: {allowed}"
            )
        self.phase = target
        if target in (PHASE_DONE, PHASE_FAILED):
            self.finished = True
        return self.phase

    def advance(self) -> str:
        """Advance along the default (happy-path) edge out of the current phase."""
        allowed = PHASE_TRANSITIONS[self.phase]
        if not allowed:
            return self.phase
        return self.transition(allowed[0])

    def finalize(self, passed: bool) -> str:
        if passed:
            if self.phase not in (PHASE_REVIEW, PHASE_DONE):
                # allow direct finalize from review only; otherwise move through the graph
                if PHASE_REVIEW in PHASE_TRANSITIONS[self.phase]:
                    self.transition(PHASE_REVIEW)
                elif PHASE_DONE not in PHASE_TRANSITIONS.get(self.phase, ()):
                    raise ReasoningStateError(
                        f"cannot finalize success from phase {self.phase}"
                    )
            if self.phase != PHASE_DONE:
                self.transition(PHASE_DONE)
        else:
            self.phase = PHASE_FAILED
            self.finished = True
        return self.phase

    # ------------------------------------------------------------- projections

    def pinned_context(self) -> str:
        """Compact context block for the model: objective, constraints, failures."""
        lines = [
            f"PHASE: {self.phase}",
            f"OBJECTIVE: {self.objective}",
            f"BUDGET: {self.steps_used}/{self.max_steps} steps",
        ]
        if self.subgoals:
            done = set(self.completed_subgoals)
            lines.append(
                "SUBGOALS: "
                + "; ".join(
                    f"[{'x' if sg in done else ' '}] {sg}" for sg in self.subgoals
                )
            )
        if self.constraints:
            uncovered = set(self.uncovered_constraints())
            rendered = []
            for idx, con in enumerate(self.constraints):
                mark = "x" if not self.constraint_status.get(f"constraint_{idx}", False) else "v"
                rendered.append(f"({mark}) {con}")
            lines.append("CONSTRAINTS: " + " | ".join(rendered))
            if uncovered:
                lines.append("UNCOVERED: " + ", ".join(sorted(uncovered)))
        if self.rejected_approaches:
            lines.append("REJECTED: " + " || ".join(self.rejected_approaches[-3:]))
        pinned = [f[8:] for f in self.known_facts if f.startswith("pinned:")]
        if pinned:
            lines.append("KNOWN: " + " | ".join(pinned))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "task_type": self.task_type,
            "constraints": list(self.constraints),
            "phase": self.phase,
            "max_steps": int(self.max_steps),
            "steps_used": int(self.steps_used),
            "subgoals": list(self.subgoals),
            "completed_subgoals": list(self.completed_subgoals),
            "known_facts": list(self.known_facts),
            "evidence_ids": list(self.evidence_ids),
            "active_symbol_contracts": dict(self.active_symbol_contracts),
            "rejected_approaches": list(self.rejected_approaches),
            "hypotheses": [
                {
                    "description": h.description,
                    "patch_targets": list(h.patch_targets),
                    "verification": h.verification,
                    "rejected_reason": h.rejected_reason,
                }
                for h in self.hypotheses
            ],
            "constraint_status": dict(self.constraint_status),
            "last_failure_category": self.last_failure_category,
            "observation_cursor": int(self.observation_cursor),
            "finished": bool(self.finished),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ReasoningState":
        state = cls(
            objective=str(payload["objective"]),
            task_type=str(payload.get("task_type", "unknown")),
            constraints=list(payload.get("constraints") or []),
            phase=str(payload.get("phase", PHASE_UNDERSTAND)),
            max_steps=int(payload.get("max_steps", 12)),
            steps_used=int(payload.get("steps_used", 0)),
            subgoals=list(payload.get("subgoals") or []),
            completed_subgoals=list(payload.get("completed_subgoals") or []),
            known_facts=list(payload.get("known_facts") or []),
            evidence_ids=list(payload.get("evidence_ids") or []),
            active_symbol_contracts=dict(payload.get("active_symbol_contracts") or {}),
            rejected_approaches=list(payload.get("rejected_approaches") or []),
            constraint_status=dict(payload.get("constraint_status") or {}),
            last_failure_category=payload.get("last_failure_category"),
            observation_cursor=int(payload.get("observation_cursor", 0)),
            finished=bool(payload.get("finished", False)),
        )
        state.hypotheses = [
            Hypothesis(
                description=str(h.get("description", "")),
                patch_targets=list(h.get("patch_targets") or []),
                verification=h.get("verification"),
                rejected_reason=h.get("rejected_reason"),
            )
            for h in (payload.get("hypotheses") or [])
        ]
        return state
