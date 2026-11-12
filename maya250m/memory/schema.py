"""Schemas for the five memory layers.

* ``WorkingMemoryState``   — session-local task state (layer 3).
* ``MemoryRecord``         — one episodic/semantic/code record (layers 4 and 5).
* ``SemanticTriple``       — one relational fact with provenance and supersession.
* ``RepoStateRef``         — a pointer to a cached hybrid KV/GDN state per snapshot.

Records are plain dataclasses so they serialize into SQLite without pickle and
survive ``weights_only=True`` boundaries when embedded in checkpoints.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "MEMORY_KINDS",
    "MemoryRecord",
    "SemanticTriple",
    "WorkingMemoryState",
    "RepoStateRef",
    "stable_memory_id",
    "content_hash",
]

MEMORY_KINDS = ("episodic", "semantic", "code", "preference")


def content_hash(text: str) -> str:
    """Stable SHA-256 of normalised text (used for reversible deduplication)."""
    normalized = " ".join(str(text).split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_memory_id(kind: str, content: str) -> str:
    """Deterministic id so re-inserting identical content is idempotent."""
    digest = hashlib.sha256(f"{kind}\x00{content}".encode("utf-8")).hexdigest()[:32]
    return f"{kind[:4]}_{digest}"


@dataclass
class MemoryRecord:
    """One durable memory with provenance, confidence and supersession links."""

    memory_id: str
    kind: str
    content: str
    summary: str = ""
    source: str = ""
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_used_at: float = 0.0
    use_count: int = 0
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in MEMORY_KINDS:
            raise ValueError(f"kind must be one of {MEMORY_KINDS}, got {self.kind!r}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("MemoryRecord.content must be a non-empty string")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.memory_id:
            self.memory_id = stable_memory_id(self.kind, self.content)
        if not self.summary:
            self.summary = self.content.strip().splitlines()[0][:200]

    @property
    def content_hash(self) -> str:
        return content_hash(self.content)

    @property
    def is_superseded(self) -> bool:
        return self.superseded_by is not None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=str(payload.get("memory_id") or stable_memory_id(
                str(payload.get("kind", "semantic")), str(payload.get("content", ""))
            )),
            kind=str(payload.get("kind", "semantic")),
            content=str(payload.get("content", "")),
            summary=str(payload.get("summary", "")),
            source=str(payload.get("source", "")),
            confidence=float(payload.get("confidence", 0.5)),
            created_at=float(payload.get("created_at", time.time())),
            last_used_at=float(payload.get("last_used_at", 0.0)),
            use_count=int(payload.get("use_count", 0)),
            supersedes=payload.get("supersedes"),
            superseded_by=payload.get("superseded_by"),
            tags=list(payload.get("tags") or []),
            embedding=payload.get("embedding"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class SemanticTriple:
    """One relational fact (subject, predicate, object) with provenance."""

    subject: str
    predicate: str
    obj: str
    source_id: str = ""
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    superseded_by: Optional[str] = None
    triple_id: str = ""

    def __post_init__(self) -> None:
        for name in ("subject", "predicate", "obj"):
            if not getattr(self, name):
                raise ValueError(f"SemanticTriple.{name} must be non-empty")
        if not self.triple_id:
            digest = hashlib.sha256(
                f"{self.subject}\x00{self.predicate}\x00{self.obj}\x00{self.source_id}".encode(
                    "utf-8"
                )
            ).hexdigest()[:24]
            self.triple_id = f"tri_{digest}"
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepoStateRef:
    """Pointer to a cached hybrid KV/GDN state for one repository snapshot."""

    repo_root: str
    snapshot_hash: str
    cache_path: str
    config_hash: str
    position: int = 0
    num_loops: int = 1
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkingMemoryState:
    """Session-local working memory with explicit token-budgeted eviction."""

    objective: str = ""
    task_spec: Optional[Dict[str, Any]] = None
    constraints: List[str] = field(default_factory=list)
    known_facts: List[Dict[str, Any]] = field(default_factory=list)
    active_symbol_contracts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    failed_hypotheses: List[str] = field(default_factory=list)
    latest_failure: Optional[str] = None
    max_facts: int = 12

    def __post_init__(self) -> None:
        if int(self.max_facts) < 1:
            raise ValueError("max_facts must be >= 1")

    def add_fact(self, text: str, *, pinned: bool = False, source: str = "") -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("fact text must be a non-empty string")
        entry = {"text": text.strip(), "pinned": bool(pinned), "source": source, "step": None}
        self.known_facts.append(entry)
        self._evict()

    def _evict(self) -> None:
        """Drop oldest unpinned facts until within budget; pinned facts are never dropped."""
        while len(self.known_facts) > int(self.max_facts):
            for idx, fact in enumerate(self.known_facts):
                if not fact.get("pinned"):
                    del self.known_facts[idx]
                    break
            else:  # everything pinned: keep the newest max_facts
                self.known_facts = self.known_facts[-int(self.max_facts) :]
                break

    def pin(self, text: str, source: str = "") -> None:
        self.add_fact(text, pinned=True, source=source)

    def record_failure(self, description: str) -> None:
        if description and description not in self.failed_hypotheses:
            self.failed_hypotheses.append(description)
            if len(self.failed_hypotheses) > 8:
                self.failed_hypotheses = self.failed_hypotheses[-8:]
        self.latest_failure = description

    def set_contract(self, qualname: str, contract: Dict[str, Any]) -> None:
        self.active_symbol_contracts[qualname] = dict(contract)

    def add_evidence(self, evidence_id: str) -> None:
        if evidence_id and evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)

    def pinned_context(self) -> str:
        lines = []
        if self.objective:
            lines.append(f"OBJECTIVE: {self.objective}")
        if self.constraints:
            lines.append("CONSTRAINTS: " + " | ".join(self.constraints))
        if self.active_symbol_contracts:
            rendered = "; ".join(
                f"{name}(min={c.get('min_arity')},max={c.get('max_arity')})"
                for name, c in list(self.active_symbol_contracts.items())[:8]
            )
            lines.append("CONTRACTS: " + rendered)
        if self.latest_failure:
            lines.append(f"LATEST_FAILURE: {self.latest_failure}")
        for fact in self.known_facts:
            if fact.get("pinned"):
                lines.append(f"KNOWN: {fact['text']}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "WorkingMemoryState":
        return cls(
            objective=str(payload.get("objective", "")),
            task_spec=payload.get("task_spec"),
            constraints=list(payload.get("constraints") or []),
            known_facts=list(payload.get("known_facts") or []),
            active_symbol_contracts=dict(payload.get("active_symbol_contracts") or {}),
            evidence_ids=list(payload.get("evidence_ids") or []),
            failed_hypotheses=list(payload.get("failed_hypotheses") or []),
            latest_failure=payload.get("latest_failure"),
            max_facts=int(payload.get("max_facts", 12)),
        )


def new_episode_id() -> str:
    """Opaque but sortable episode identifier."""
    return f"ep_{uuid.uuid4().hex[:16]}"
