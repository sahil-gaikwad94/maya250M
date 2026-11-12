"""L2-normalised dense vector index for the shared-trunk memory space.

The index is intentionally tiny and dependency-light: a contiguous ``float32``
matrix plus an id list, searched by a single matrix-vector product. At 128
dimensions and 100k records this is ~51 MB and a few milliseconds per query, which
is the right trade-off for a 250M-parameter agent that must stay within a CPU
memory budget. It is not a replacement for an ANN library; it is exact and
auditable, which matters more here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["VectorIndex", "HashingEmbedder"]


@dataclass
class VectorIndex:
    """Exact cosine-similarity index over L2-normalised vectors."""

    dim: int = 128
    ids: List[str] = field(default_factory=list)
    vectors: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if int(self.dim) < 1:
            raise ValueError("dim must be >= 1")
        if self.vectors is None:
            self.vectors = np.zeros((0, int(self.dim)), dtype=np.float32)
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or self.vectors.shape[1] != int(self.dim):
            raise ValueError(
                f"vectors must be [n, {self.dim}], got {self.vectors.shape}"
            )
        if len(self.ids) != self.vectors.shape[0]:
            raise ValueError("ids length must match number of vectors")

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, memory_id: str, vector: Sequence[float]) -> None:
        """Insert or replace one vector (ids are unique)."""
        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (int(self.dim),):
            raise ValueError(f"expected [{self.dim}] vector, got {vec.shape}")
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        if memory_id in self.ids:
            self.vectors[self.ids.index(memory_id)] = vec
            return
        self.ids.append(memory_id)
        self.vectors = np.vstack([self.vectors, vec[None, :]])

    def add_many(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        for memory_id, vector in zip(ids, np.asarray(vectors, dtype=np.float32)):
            self.add(memory_id, vector)

    def remove(self, memory_id: str) -> bool:
        if memory_id not in self.ids:
            return False
        idx = self.ids.index(memory_id)
        self.ids.pop(idx)
        self.vectors = np.delete(self.vectors, idx, axis=0)
        return True

    def search(self, query: Sequence[float], k: int = 10) -> List[Tuple[str, float]]:
        """Return the top-``k`` ``(memory_id, cosine_similarity)`` pairs."""
        if len(self.ids) == 0:
            return []
        query_vec = np.asarray(query, dtype=np.float32)
        if query_vec.shape != (int(self.dim),):
            raise ValueError(f"expected [{self.dim}] query, got {query_vec.shape}")
        norm = float(np.linalg.norm(query_vec))
        if norm > 0:
            query_vec = query_vec / norm
        scores = self.vectors @ query_vec
        k = max(1, min(int(k), len(self.ids)))
        top = np.argpartition(-scores, k - 1)[:k]
        ordered = sorted(top.tolist(), key=lambda i: (-float(scores[i]), self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in ordered]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            ids=np.asarray(self.ids, dtype=object),
            vectors=self.vectors,
            dim=np.asarray([int(self.dim)]),
        )

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        with np.load(path, allow_pickle=True) as payload:
            ids = [str(x) for x in payload["ids"].tolist()]
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
            dim = int(payload["dim"][0])
        return cls(dim=dim, ids=ids, vectors=vectors)

    def stats(self) -> Dict[str, Any]:
        return {
            "dim": int(self.dim),
            "count": len(self.ids),
            "bytes": int(self.vectors.nbytes) if self.vectors is not None else 0,
        }


@dataclass
class HashingEmbedder:
    """Deterministic feature-hashing embedder for tests and cold-start bootstrap.

    This is **not** a learned semantic encoder: it exists so the retrieval plumbing
    can be exercised end-to-end before ``W_mem`` is trained. Production retrieval
    must use trunk hidden states projected through ``MemoryProjection``.
    """

    dim: int = 128
    seed: int = 17

    def encode(self, text: str) -> np.ndarray:
        import hashlib
        import re

        vec = np.zeros(int(self.dim), dtype=np.float32)
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]+", str(text).lower())
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.blake2b(
                f"{self.seed}:{token}".encode("utf-8"), digest_size=8
            ).digest()
            index = int.from_bytes(digest[:4], "little") % int(self.dim)
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def encode_many(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts], axis=0) if texts else np.zeros(
            (0, int(self.dim)), dtype=np.float32
        )
