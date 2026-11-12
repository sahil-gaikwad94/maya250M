"""Candidate reranking for hybrid memory retrieval.

Two modes, both explicit about what they are:

* ``mode="deterministic"`` (default) — a fixed, auditable blend of lexical score,
  vector cosine, record confidence and freshness. No learned parameters, so an
  evaluation can reproduce every ranking exactly.
* ``mode="neural"`` — a *shared-trunk* relevance score: the caller supplies an
  ``encoder`` (``MemoryProjection`` over pooled trunk hidden states, or a trained
  ``SharedTrunkMemoryReranker`` module). If no encoder is supplied the reranker
  refuses to pretend and falls back to deterministic mode, recording
  ``fallback_reason`` in the audit trail.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["RerankCandidate", "SharedTrunkNeuralReranker"]


@dataclass
class RerankCandidate:
    """One candidate with its per-signal scores and final ranking score."""

    memory_id: str
    content: str
    lexical_score: float = 0.0
    vector_score: float = 0.0
    relational_score: float = 0.0
    confidence: float = 0.5
    created_at: float = 0.0
    last_used_at: float = 0.0
    sources: List[str] = field(default_factory=list)
    neural_score: Optional[float] = None
    final_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "lexical_score": self.lexical_score,
            "vector_score": self.vector_score,
            "relational_score": self.relational_score,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "sources": list(self.sources),
            "neural_score": self.neural_score,
            "final_score": self.final_score,
        }


@dataclass
class SharedTrunkNeuralReranker:
    """Blend lexical, dense and relational evidence, optionally with a learned score."""

    mode: str = "deterministic"
    w_lexical: float = 0.35
    w_vector: float = 0.40
    w_relational: float = 0.10
    w_confidence: float = 0.10
    w_freshness: float = 0.05
    w_neural: float = 0.60
    freshness_half_life_days: float = 30.0
    encoder: Optional[Callable[[Sequence[str]], Sequence[float]]] = None
    fallback_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in ("deterministic", "neural"):
            raise ValueError("mode must be 'deterministic' or 'neural'")
        if self.mode == "neural" and self.encoder is None:
            self.fallback_reason = "no_encoder_supplied"
            self.mode = "deterministic"

    def _freshness(self, timestamp: float, now: float) -> float:
        if timestamp <= 0:
            return 0.0
        age_days = max(0.0, (now - timestamp) / 86400.0)
        return float(math.exp(-age_days / max(1e-6, self.freshness_half_life_days)))

    def score(
        self, candidate: RerankCandidate, now: Optional[float] = None
    ) -> float:
        now = float(time.time() if now is None else now)
        base = (
            self.w_lexical * float(candidate.lexical_score)
            + self.w_vector * float(candidate.vector_score)
            + self.w_relational * float(candidate.relational_score)
            + self.w_confidence * float(candidate.confidence)
            + self.w_freshness
            * max(self._freshness(candidate.created_at, now), self._freshness(candidate.last_used_at, now))
        )
        if self.mode == "neural" and candidate.neural_score is not None:
            base = (1.0 - self.w_neural) * base + self.w_neural * float(candidate.neural_score)
        return float(base)

    def rerank(
        self,
        candidates: List[RerankCandidate],
        now: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[RerankCandidate]:
        if self.mode == "neural" and self.encoder is not None and candidates:
            vectors = list(self.encoder([c.content for c in candidates]))
            for candidate, score in zip(candidates, vectors):
                candidate.neural_score = float(score)
        for candidate in candidates:
            candidate.final_score = self.score(candidate, now=now)
        ordered = sorted(
            candidates,
            key=lambda c: (-c.final_score, -c.confidence, c.memory_id),
        )
        return ordered[: int(top_k)] if top_k is not None else ordered

    def audit(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "fallback_reason": self.fallback_reason,
            "weights": {
                "lexical": self.w_lexical,
                "vector": self.w_vector,
                "relational": self.w_relational,
                "confidence": self.w_confidence,
                "freshness": self.w_freshness,
                "neural": self.w_neural,
            },
        }
