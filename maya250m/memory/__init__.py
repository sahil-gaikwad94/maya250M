"""Memory subsystem: durable store, dense index, hybrid retrieval, consolidation.

The five memory layers are kept strictly distinct:

1. parametric (Maya's weights) — not in this package;
2. fast sequence memory (KV + GDN state) — ``RepoStateRef`` pointers only;
3. working memory — :class:`WorkingMemoryState`;
4. episodic memory — ``MemoryRecord(kind="episodic")``;
5. semantic / codebase relational memory — ``MemoryRecord(kind="semantic"|"code")``
   plus :class:`SemanticTriple` rows.
"""

from __future__ import annotations

from .consolidation import ConsolidationReport, MemoryConsolidator
from .context_builder import ContextBundle, EvidenceContextBuilder
from .index import HashingEmbedder, VectorIndex
from .projection import (
    MEMORY_DIM,
    MemoryProjection,
    build_shared_trunk_memory_projection,
    memory_projection_parameters,
)
from .rerank import RerankCandidate, SharedTrunkNeuralReranker
from .retrieval import HybridRetriever, RetrievalTrace
from .schema import (
    MEMORY_KINDS,
    MemoryRecord,
    RepoStateRef,
    SemanticTriple,
    WorkingMemoryState,
    content_hash,
    new_episode_id,
    stable_memory_id,
)
from .store import SCHEMA_VERSION, MayaMemoryStore

__all__ = [
    "ConsolidationReport",
    "MemoryConsolidator",
    "ContextBundle",
    "EvidenceContextBuilder",
    "HashingEmbedder",
    "VectorIndex",
    "MEMORY_DIM",
    "MemoryProjection",
    "build_shared_trunk_memory_projection",
    "memory_projection_parameters",
    "RerankCandidate",
    "SharedTrunkNeuralReranker",
    "HybridRetriever",
    "RetrievalTrace",
    "MEMORY_KINDS",
    "MemoryRecord",
    "RepoStateRef",
    "SemanticTriple",
    "WorkingMemoryState",
    "content_hash",
    "new_episode_id",
    "stable_memory_id",
    "SCHEMA_VERSION",
    "MayaMemoryStore",
]
