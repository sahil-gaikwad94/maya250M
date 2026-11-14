"""Multi-turn agent trajectory schema, SFT loss masking and process rewards.

Two jobs:

* **SFT construction** — a trajectory is serialised into ``input_ids``/``labels``
  where every ``[TOOL_DATA_START] ... [TOOL_DATA_END]`` observation is ``-100``.
  Maya is therefore trained on its own reasoning and actions, never on the
  environment's text (which would teach it to hallucinate file contents).
* **Process + outcome reward** — :func:`score_trajectory` implements the
  deterministic reward from the specification:

      R = R_outcome + R_localization + R_reproduction - P_redundancy

  with ``P_redundancy = 0.05 * duplicate_calls + 0.10 * syntax_error_patches``.
  Binary outcome alone is far too sparse to train a 250M policy.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .firewall import TOOL_DATA_END, TOOL_DATA_START

__all__ = [
    "TrajectoryStep",
    "AgentTrajectory",
    "ProcessReward",
    "score_trajectory",
    "build_sft_example",
    "RESOLVED_THRESHOLD",
]

RESOLVED_THRESHOLD = 1.0

_READ_ONLY_TOOLS = (
    "get_repo_map",
    "get_repo_context",
    "search_repo",
    "view_file_span",
    "inspect_symbol_graph",
)


@dataclass
class TrajectoryStep:
    """One (thought, action, observation) triple."""

    step: int
    phase: str
    thought: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    status: str = "ok"  # ok | failed | loop_guard | budget
    verification: Optional[Dict[str, Any]] = None
    observation_masked: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrajectoryStep":
        return cls(**payload)


@dataclass
class AgentTrajectory:
    """A complete bounded episode."""

    task_id: str = ""
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_patch: Optional[str] = None
    outcome: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = f"traj_{uuid.uuid4().hex[:12]}"

    def add_step(self, step: TrajectoryStep) -> TrajectoryStep:
        self.steps.append(step)
        return step

    @property
    def resolved(self) -> bool:
        """A trajectory resolves only from verified completion evidence.

        ``outcome`` alone is a numeric label; without the structured evidence block
        written by ``AgentRuntime.finalize`` it cannot make a trajectory an RL/success
        target (a caller-supplied ``finish(True)`` used to do exactly that).
        """
        if int(self.outcome) < int(RESOLVED_THRESHOLD):
            return False
        evidence = self.metadata.get("completion_evidence")
        return isinstance(evidence, dict) and bool(evidence.get("verified"))

    def tools_used(self) -> List[str]:
        return [s.tool_name for s in self.steps if s.tool_name]

    def duplicate_calls(self) -> int:
        return sum(1 for s in self.steps if s.status == "loop_guard")

    def syntax_error_patches(self) -> int:
        count = 0
        for step in self.steps:
            verification = step.verification or {}
            if verification.get("failure_category") == "syntax_error":
                count += 1
        return count

    def first_patch_index(self) -> Optional[int]:
        for idx, step in enumerate(self.steps):
            if step.tool_name == "apply_patch":
                return idx
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": [s.to_dict() for s in self.steps],
            "final_patch": self.final_patch,
            "outcome": int(self.outcome),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentTrajectory":
        return cls(
            task_id=str(payload.get("task_id", "")),
            steps=[TrajectoryStep.from_dict(s) for s in payload.get("steps") or []],
            final_patch=payload.get("final_patch"),
            outcome=int(payload.get("outcome", 0)),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_jsonl(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def read_jsonl(cls, path: str) -> List["AgentTrajectory"]:
        out: List[AgentTrajectory] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    out.append(cls.from_dict(json.loads(line)))
        return out


# --------------------------------------------------------------------- SFT


def build_sft_example(
    tokenizer,
    trajectory: AgentTrajectory,
    include_observations: bool = True,
) -> Dict[str, List[int]]:
    """Tokenise a trajectory, masking every environment observation from the loss."""
    input_ids: List[int] = []
    labels: List[int] = []

    def append_supervised(text: str) -> None:
        if not text:
            return
        ids = tokenizer.encode(text, add_bos=not input_ids, add_eos=False)
        input_ids.extend(ids)
        labels.extend(ids)

    def append_masked(text: str) -> None:
        if not text:
            return
        ids = tokenizer.encode(text, add_bos=not input_ids, add_eos=False)
        input_ids.extend(ids)
        labels.extend([-100] * len(ids))

    for step in trajectory.steps:
        header = f"\n[{step.phase}] "
        append_supervised(header)

        thought = f"<{step.thought.strip()}> " if step.thought else ""
        action = json.dumps({"tool": step.tool_name, "args": step.tool_args}, sort_keys=True)
        append_supervised(thought + action + "\n")

        if include_observations and step.observation:
            if step.observation_masked:
                append_masked(step.observation + "\n")
            else:
                append_supervised(step.observation + "\n")

    # Never compute loss on the very first token (no left context).
    if labels:
        labels[0] = -100
    return {"input_ids": input_ids, "labels": labels}


# ------------------------------------------------------------------- rewards


@dataclass
class ProcessReward:
    """Decomposed agentic reward, per specification 21D.4."""

    outcome: float = 0.0
    localization: float = 0.0
    reproduction: float = 0.0
    redundancy_penalty: float = 0.0
    duplicate_calls: int = 0
    syntax_error_patches: int = 0

    @property
    def total(self) -> float:
        return float(
            self.outcome
            + self.localization
            + self.reproduction
            - self.redundancy_penalty
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "localization": self.localization,
            "reproduction": self.reproduction,
            "redundancy_penalty": self.redundancy_penalty,
            "total": self.total,
            "duplicate_calls": self.duplicate_calls,
            "syntax_error_patches": self.syntax_error_patches,
        }


def _step_touched_ground_truth(step: TrajectoryStep, files: Sequence[str], symbols: Sequence[str]) -> bool:
    args = step.tool_args or {}
    candidates: List[str] = []
    for key in ("file_path", "pattern", "symbol_name", "query", "test_target"):
        value = args.get(key)
        if isinstance(value, str):
            candidates.append(value)
    observation = step.observation or ""
    for path in files:
        if any(path in c for c in candidates) or path in observation:
            return True
    for symbol in symbols:
        if any(symbol in c for c in candidates) or symbol in observation:
            return True
    return False


def score_trajectory(
    trajectory: AgentTrajectory,
    ground_truth_files: Sequence[str] = (),
    ground_truth_symbols: Sequence[str] = (),
    outcome_reward: float = 1.0,
) -> ProcessReward:
    """Compute the deterministic process + outcome reward for one trajectory."""
    reward = ProcessReward()
    reward.outcome = float(outcome_reward) if trajectory.resolved else 0.0
    reward.duplicate_calls = trajectory.duplicate_calls()
    reward.syntax_error_patches = trajectory.syntax_error_patches()
    reward.redundancy_penalty = 0.05 * reward.duplicate_calls + 0.10 * reward.syntax_error_patches

    first_patch = trajectory.first_patch_index()
    pre_patch = (
        trajectory.steps[:first_patch]
        if first_patch is not None
        else list(trajectory.steps)
    )

    if ground_truth_files or ground_truth_symbols:
        for step in pre_patch:
            if _step_touched_ground_truth(step, ground_truth_files, ground_truth_symbols):
                reward.localization = 0.20
                break

    ran_before = any(s.tool_name == "run_tests" for s in pre_patch)
    ran_after = (
        any(
            s.tool_name == "run_tests"
            for s in trajectory.steps[(first_patch + 1) :] if first_patch is not None
        )
        if first_patch is not None
        else False
    )
    if ran_before and ran_after:
        reward.reproduction = 0.15
    return reward


def observation_mask_spans(text: str) -> List[Tuple[int, int]]:
    """Character spans of fenced tool data inside ``text`` (for span-level masking)."""
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        open_idx = text.find(TOOL_DATA_START, start)
        if open_idx == -1:
            break
        close_idx = text.find(TOOL_DATA_END, open_idx)
        if close_idx == -1:
            break
        spans.append((open_idx, close_idx + len(TOOL_DATA_END)))
        start = close_idx + len(TOOL_DATA_END)
    return spans
