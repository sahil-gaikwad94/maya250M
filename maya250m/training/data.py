"""Post-training data: tokenised SFT examples, preference pairs, rollout groups.

Design rules that matter for correctness:

* prompt tokens always carry label ``-100``; only the completion is supervised;
* completions are scored with an explicit EOS so a truncated answer cannot look
  cheap;
* length overflow drops *prompt* tokens from the left first, never completion
  tokens, so the thing being scored stays intact;
* :func:`group_advantages_from_rewards` returns a validity mask instead of
  silently producing ``0/1e-6`` advantages for a uniform group.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

__all__ = [
    "TokenizedExample",
    "PreferencePair",
    "PreferenceDataset",
    "RolloutGroup",
    "tokenize_prompt_completion",
    "tokenize_sft_text",
    "examples_from_trajectories",
    "pairs_from_vexx_rows",
    "build_rollout_groups",
    "pad_examples",
]

IGNORE_INDEX = -100
MIN_COMPLETION_TOKENS = 8


def _pad_id(tokenizer) -> int:
    for attr in ("pad_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


@dataclass
class TokenizedExample:
    """One supervised sequence with prompt-masked labels."""

    input_ids: List[int]
    labels: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def supervised_tokens(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)

    def to_tensors(self, device: Optional[torch.device] = None):
        ids = torch.tensor([self.input_ids], dtype=torch.long)
        labels = torch.tensor([self.labels], dtype=torch.long)
        if device is not None:
            ids, labels = ids.to(device), labels.to(device)
        return ids, labels


def tokenize_prompt_completion(
    tokenizer,
    prompt: str,
    completion: str,
    max_len: int = 2048,
    add_eos: bool = True,
) -> TokenizedExample:
    """Tokenise ``prompt + completion`` with the prompt masked out of the loss."""
    if max_len < MIN_COMPLETION_TOKENS + 2:
        raise ValueError(f"max_len too small: {max_len}")

    prompt_ids = list(tokenizer.encode(prompt, add_bos=True, add_eos=False))
    completion_ids = list(
        tokenizer.encode(completion, add_bos=False, add_eos=add_eos)
    )
    if not completion_ids:
        completion_ids = list(tokenizer.encode(" ", add_bos=False, add_eos=add_eos))
        completion_ids = completion_ids[:1] or [0]

    budget = int(max_len) - len(completion_ids)
    if budget < 1:
        completion_ids = completion_ids[: max_len - 1]
        budget = 1
    if len(prompt_ids) > budget:
        keep = max(1, budget)
        prompt_ids = prompt_ids[-keep:]

    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
    return TokenizedExample(
        input_ids=input_ids,
        labels=labels,
        metadata={"prompt_tokens": len(prompt_ids), "completion_tokens": len(completion_ids)},
    )


def tokenize_sft_text(
    tokenizer, text: str, max_len: int = 2048, add_eos: bool = True
) -> TokenizedExample:
    """Full-text SFT example (no prompt masking); used for continued pretraining."""
    ids = list(tokenizer.encode(text, add_bos=True, add_eos=add_eos))
    if len(ids) > max_len:
        ids = ids[:max_len]
    return TokenizedExample(input_ids=ids, labels=list(ids), metadata={"sft_text": True})


def examples_from_trajectories(
    tokenizer, trajectories: Iterable[Any], max_len: int = 2048
) -> List[TokenizedExample]:
    """Convert agent trajectories with tool observations masked out of the loss."""
    from ..agent.trajectories import build_sft_example

    examples: List[TokenizedExample] = []
    for trajectory in trajectories:
        built = build_sft_example(tokenizer, trajectory)
        ids, labels = built["input_ids"], built["labels"]
        if len(ids) > max_len:
            ids, labels = ids[:max_len], labels[:max_len]
        if sum(1 for label in labels if label != IGNORE_INDEX) == 0:
            continue
        examples.append(
            TokenizedExample(
                input_ids=ids, labels=labels, metadata={"task_id": trajectory.task_id}
            )
        )
    return examples


@dataclass
class PreferencePair:
    """A (prompt, chosen, rejected) triple for DPO."""

    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": dict(self.metadata),
        }


@dataclass
class PreferenceDataset:
    """A list of preference pairs loaded from a Vexx DPO export or JSONL."""

    pairs: List[PreferencePair] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    def filter_nonempty(self) -> "PreferenceDataset":
        return PreferenceDataset(
            [p for p in self.pairs if p.chosen.strip() and p.rejected.strip() and p.chosen != p.rejected]
        )

    @classmethod
    def from_jsonl(cls, path: str) -> "PreferenceDataset":
        pairs: List[PreferencePair] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "winner" in row and "losers" in row:
                    pairs.extend(pairs_from_vexx_rows([row]))
                    continue
                pairs.append(
                    PreferencePair(
                        prompt=row["prompt"],
                        chosen=row["chosen"],
                        rejected=row["rejected"],
                        metadata=row.get("metadata") or {},
                    )
                )
        return cls(pairs)


def _vexx_prompt(row: Dict[str, Any]) -> str:
    fn = row.get("fn", "solution")
    tests = row.get("tests") or []
    rendered = json.dumps(tests, ensure_ascii=False, sort_keys=True)
    return (
        f"# Write a Python function `{fn}` that passes these unit tests.\n"
        f"# tests = {rendered}\n"
    )


def pairs_from_vexx_rows(rows: Iterable[Dict[str, Any]]) -> List[PreferencePair]:
    """Expand Vexx ``dpo_buffer`` records into one pair per (winner, loser)."""
    pairs: List[PreferencePair] = []
    for row in rows:
        prompt = _vexx_prompt(row)
        winner = row.get("winner") or ""
        for loser in row.get("losers") or []:
            if not winner.strip() or not loser.strip() or winner == loser:
                continue
            pairs.append(
                PreferencePair(
                    prompt=prompt,
                    chosen=winner,
                    rejected=loser,
                    metadata={
                        "fn": row.get("fn"),
                        "oracle": row.get("oracle", "subprocess unit-test execution"),
                    },
                )
            )
    return pairs


@dataclass
class RolloutGroup:
    """A GRPO group: several completions of one prompt with verifiable rewards."""

    prompt: str
    completions: List[str]
    rewards: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.completions) != len(self.rewards):
            raise ValueError("completions and rewards must have equal length")

    def advantages(self):
        from .losses import group_advantages

        return group_advantages(self.rewards)


def build_rollout_groups(
    prompt: str,
    completions: Sequence[str],
    rewards: Sequence[float],
    metadata: Optional[Dict[str, Any]] = None,
) -> RolloutGroup:
    return RolloutGroup(
        prompt=prompt,
        completions=list(completions),
        rewards=[float(r) for r in rewards],
        metadata=dict(metadata or {}),
    )


def pad_examples(
    examples: Sequence[TokenizedExample], pad_id: int = 0
) -> Dict[str, torch.Tensor]:
    """Right-pad a batch of examples; padded positions are masked in ``labels``."""
    if not examples:
        raise ValueError("cannot pad an empty batch")
    max_len = max(len(e) for e in examples)
    input_ids = torch.full((len(examples), max_len), int(pad_id), dtype=torch.long)
    labels = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long)
    for row, example in enumerate(examples):
        n = len(example)
        input_ids[row, :n] = torch.tensor(example.input_ids, dtype=torch.long)
        labels[row, :n] = torch.tensor(example.labels, dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}
