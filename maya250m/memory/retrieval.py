"""Hybrid memory retrieval: lexical (FTS5) + dense (shared-trunk) + relational.

Every candidate keeps its per-source ranks so an evaluation report can state
``recall@k`` per channel and prove which channel contributed the hit. Retrieved
chunk ids are logged verbatim (see :meth:`HybridRetriever.last_trace`) — the
"Maya + Persistent Memory" evaluation identity depends on that audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .index import VectorIndex
from .rerank import RerankCandidate, SharedTrunkNeuralReranker
from .schema import MemoryRecord
from .store import MayaMemoryStore

__all__ = ["RetrievalTrace", "HybridRetriever", "tosco"]


@dataclass
class RetrievalTrace:
    """Audit record of one retrieval call."""

    query: str
    lexical_ids: List[str] = field(default_factory=list)
    vector_ids: List[str] = field(default_factory=list)
    relational_ids: List[str] = field(default_factory=list)
    merged_ids: List[str] = field(default_factory=list)
    reranked_ids: List[str] = field(default_factory=list)
    channels_used: List[str] = field(default_factory=list)
    reranker_audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "lexical_ids": list(self.lexical_ids),
            "vector_ids": list(self.vector_ids),
            "relational_ids": list(self.relational_ids),
            "merged_ids": list(self.merged_ids),
            "reranked_ids": list(self.reranked_ids),
            "channels_used": list(self.channels_used),
            "reranker_audit": dict(self.reranker_audit),
        }


def tosco(value: float, lo: float, hi: float) -> float:
    """Scale a raw score into ``[0, 1]`` deterministically."""
    if hi <= lo:
        return 0.0
    return float(min(1.0, max(0.0, (value - lo) / (hi - lo))))


class HybridRetriever:
    """Merge lexical, dense and relational candidates, then rerank them."""

    def __init__(
        self,
        store: MayaMemoryStore,
        index: Optional[VectorIndex] = None,
        reranker: Optional[SharedTrunkNeuralReranker] = None,
        embed: Optional[Callable[[str], Sequence[float]]] = None,
        relational_weight: float = 0.6,
    ):
        self.store = store
        self.index = index
        self.reranker = reranker or SharedTrunkNeuralReranker()
        self.embed = embed
        self.relational_weight = float(relational_weight)
        self.last_trace: Optional[RetrievalTrace] = None

    # ------------------------------------------------------------------ channels

    def _lexical(self, query: str, limit: int) -> List[tuple]:
        return self.store.lex_search(query, limit=limit)

    def _vector(self, query: str, limit: int) -> List[tuple]:
        if self.index is None or self.embed is None or len(self.index) == 0:
            return []
        vector = np.asarray(self.embed(query), dtype=np.float32)
        return self.index.search(vector, k=limit)

    def _relational(self, query: str, limit: int) -> List[MemoryRecord]:
        """Return records reachable through triples whose terms appear in the query."""
        lowered = query.lower()
        hits: List[MemoryRecord] = []
        seen = set()
        for triple in self.store.query_triples():
            for term in (triple.subject, triple.predicate, triple.obj):
                if term and term.lower() in lowered:
                    for record in self.store.find_by_hash(triple.source_id) or []:
                        if record.memory_id not in seen:
                            seen.add(record.memory_id)
                            hits.append(record)
                    break
            if len(hits) >= limit:
                break
        return hits[:limit]

    # ------------------------------------------------------------------- public

    def retrieve(
        self,
        query: str,
        k: int = 8,
        *,
        kind: Optional[str] = None,
        channel_limit: Optional[int] = None,
        include_superseded: bool = False,
        now: Optional[float] = None,
    ) -> List[RerankCandidate]:
        """Retrieve the top-``k`` memory candidates for ``query``."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("retrieve requires a non-empty query")
        limit = int(channel_limit or max(k * 4, 12))
        trace = RetrievalTrace(query=query)

        candidates: Dict[str, RerankCandidate] = {}

        lex = self._lexical(query, limit)
        if lex:
            trace.channels_used.append("lexical")
            best = max(score for _rec, score in lex) or 1.0
            for rank, (record, score) in enumerate(lex):
                if kind is not None and record.kind != kind:
                    continue
                trace.lexical_ids.append(record.memory_id)
                cand = candidates.setdefault(
                    record.memory_id,
                    RerankCandidate(
                        memory_id=record.memory_id,
                        content=record.content,
                        confidence=record.confidence,
                        created_at=record.created_at,
                        last_used_at=record.last_used_at,
                    ),
                )
                cand.lexical_score = max(cand.lexical_score, tosco(score, 0.0, best))
                cand.sources.append("lexical")

        vec = self._vector(query, limit)
        if vec:
            trace.channels_used.append("vector")
            for memory_id, score in vec:
                record = self.store.get_memory(memory_id)
                if record is None or (kind is not None and record.kind != kind):
                    continue
                trace.vector_ids.append(memory_id)
                cand = candidates.setdefault(
                    memory_id,
                    RerankCandidate(
                        memory_id=memory_id,
                        content=record.content,
                        confidence=record.confidence,
                        created_at=record.created_at,
                        last_used_at=record.last_used_at,
                    ),
                )
                cand.vector_score = max(cand.vector_score, tosco(score, -1.0, 1.0))
                cand.sources.append("vector")

        rel = self._relational(query, limit)
        if rel:
            trace.channels_used.append("relational")
            for record in rel:
                if kind is not None and record.kind != kind:
                    continue
                trace.relational_ids.append(record.memory_id)
                cand = candidates.setdefault(
                    record.memory_id,
                    RerankCandidate(
                        memory_id=record.memory_id,
                        content=record.content,
                        confidence=record.confidence,
                        created_at=record.created_at,
                        last_used_at=record.last_used_at,
                    ),
                )
                cand.relational_score = max(cand.relational_score, self.relational_weight)
                cand.sources.append("relational")

        merged = list(candidates.values())
        if not include_superseded:
            merged = [
                c
                for c in merged
                if (rec := self.store.get_memory(c.memory_id)) is None
                or not rec.is_superseded
            ]
        trace.merged_ids = [c.memory_id for c in merged]

        ranked = self.reranker.rerank(merged, now=now, top_k=k)
        trace.reranked_ids = [c.memory_id for c in ranked]
        trace.reranker_audit = self.reranker.audit()
        self.last_trace = trace

        if ranked:
            self.store.mark_used([c.memory_id for c in ranked], now=now)
        return ranked

    def recall_at_k(self, relevant_ids: Sequence[str], k: int = 5) -> float:
        """Fraction of ``relevant_ids`` found in the last retrieval's top-``k``."""
        if self.last_trace is None:
            raise RuntimeError("recall_at_k requires a prior retrieve() call")
        if not relevant_ids:
            return 1.0
        top = set(self.last_trace.reranked_ids[: int(k)])
        return float(sum(1 for rid in relevant_ids if rid in top) / len(relevant_ids))
