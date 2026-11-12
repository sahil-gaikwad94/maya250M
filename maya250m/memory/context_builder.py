"""Token-budgeted evidence assembly with provenance.

The context builder is the only place that decides what evidence reaches Maya's
32k window. It is deterministic: given the same candidates, budget and tokenizer
it produces the same block, and it always reports which memory ids were used and
which were dropped, so a "Maya + Persistent Memory" evaluation can print the exact
retrieved chunks alongside its score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .rerank import RerankCandidate

__all__ = ["ContextBundle", "EvidenceContextBuilder"]

DEFAULT_TOKEN_BUDGET = 2048


def _approx_tokens(text: str) -> int:
    """Conservative token estimate when no tokenizer is supplied (4 chars/token)."""
    return max(1, (len(text) + 3) // 4)


@dataclass
class ContextBundle:
    """Assembled evidence block plus its accounting."""

    text: str
    used_ids: List[str] = field(default_factory=list)
    dropped_ids: List[str] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    token_budget: int = DEFAULT_TOKEN_BUDGET
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "used_ids": list(self.used_ids),
            "dropped_ids": list(self.dropped_ids),
            "provenance": list(self.provenance),
            "tokens_used": int(self.tokens_used),
            "token_budget": int(self.token_budget),
            "truncated": bool(self.truncated),
        }


class EvidenceContextBuilder:
    """Assemble ranked evidence into a hard token budget."""

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET, tokenizer=None):
        if int(token_budget) < 32:
            raise ValueError("token_budget must be >= 32")
        self.token_budget = int(token_budget)
        self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_bos=False, add_eos=False))
            except Exception:  # pragma: no cover - tokenizer protocol mismatch
                pass
        return _approx_tokens(text)

    def build(
        self,
        candidates: Sequence[RerankCandidate],
        header: Optional[str] = None,
        pinned: Optional[str] = None,
        max_chars_per_item: int = 2400,
    ) -> ContextBundle:
        """Render ``candidates`` (already ranked) into the budgeted block."""
        parts: List[str] = []
        provenance: List[Dict[str, Any]] = []
        used: List[str] = []
        dropped: List[str] = []

        if header:
            parts.append(header.rstrip())
        if pinned:
            parts.append("--- PINNED STATE ---\n" + pinned.rstrip())

        used_tokens = self.count_tokens("\n\n".join(parts)) if parts else 0

        for rank, candidate in enumerate(candidates):
            body = (candidate.content or "").strip()
            if len(body) > int(max_chars_per_item):
                body = body[: int(max_chars_per_item)].rstrip() + "\n... [truncated]"
            rendered = (
                f"--- EVIDENCE [{candidate.memory_id}] "
                f"(rank={rank + 1}, score={candidate.final_score:.3f})\n{body}"
            )
            cost = self.count_tokens(rendered) + 1
            if used_tokens + cost > self.token_budget:
                dropped.extend(c.memory_id for c in candidates[rank:])
                break
            parts.append(rendered)
            used.append(candidate.memory_id)
            used_tokens += cost
            provenance.append(
                {
                    "memory_id": candidate.memory_id,
                    "rank": rank + 1,
                    "score": round(float(candidate.final_score), 6),
                    "sources": list(candidate.sources),
                }
            )

        text = "\n\n".join(parts)
        return ContextBundle(
            text=text,
            used_ids=used,
            dropped_ids=dropped,
            provenance=provenance,
            tokens_used=self.count_tokens(text),
            token_budget=self.token_budget,
            truncated=bool(dropped),
        )
