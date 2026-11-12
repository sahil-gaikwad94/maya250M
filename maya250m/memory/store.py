"""Durable SQLite memory store with an FTS5 lexical index and versioned schema.

Design constraints:

* **No external services.** A single SQLite file (or ``:memory:``) holds episodes,
  semantic records, relational triples and the FTS5 index, so the same store works
  on Kaggle, a laptop and a sandbox.
* **Additive migrations.** ``SCHEMA_VERSION`` is stored in ``meta`` and bumped only
  by additive migrations, so older stores keep loading.
* **Reversible consolidation.** Supersession never deletes rows; it links
  ``superseded_by`` so a wrong merge can be undone.
* **Fail-soft FTS5.** If the SQLite build lacks FTS5 the store degrades to a
  deterministic ``LIKE`` scan and reports ``lexical_backend="like"`` instead of
  pretending lexical search works.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import MemoryRecord, SemanticTriple, content_hash

__all__ = ["MayaMemoryStore", "SCHEMA_VERSION"]

SCHEMA_VERSION = 3

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


class MayaMemoryStore:
    """SQLite-backed episodic/semantic/code memory with FTS5 lexical search."""

    def __init__(self, path: str = ":memory:", *, check_same_thread: bool = True):
        self.path = path
        if path not in (":memory:", "") and os.path.dirname(os.path.abspath(path)):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.lexical_backend = "none"
        self._init_schema()

    # ------------------------------------------------------------------ schema

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id      TEXT PRIMARY KEY,
                kind           TEXT NOT NULL,
                content        TEXT NOT NULL,
                summary        TEXT NOT NULL DEFAULT '',
                source         TEXT NOT NULL DEFAULT '',
                confidence     REAL NOT NULL DEFAULT 0.5,
                created_at     REAL NOT NULL,
                last_used_at   REAL NOT NULL DEFAULT 0.0,
                use_count      INTEGER NOT NULL DEFAULT 0,
                supersedes     TEXT,
                superseded_by  TEXT,
                tags           TEXT NOT NULL DEFAULT '[]',
                embedding      BLOB,
                metadata       TEXT NOT NULL DEFAULT '{}',
                content_hash   TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_kind ON memories(kind)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(content_hash)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_triples (
                triple_id      TEXT PRIMARY KEY,
                subject        TEXT NOT NULL,
                predicate      TEXT NOT NULL,
                obj            TEXT NOT NULL,
                source_id      TEXT NOT NULL DEFAULT '',
                confidence     REAL NOT NULL DEFAULT 0.5,
                created_at     REAL NOT NULL,
                superseded_by  TEXT
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_tri_spo ON semantic_triples(subject, predicate)"
        )
        try:
            cur.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5("
                "memory_id UNINDEXED, content, summary, tokenize='porter unicode61')"
            )
            self.lexical_backend = "fts5"
        except sqlite3.OperationalError:
            self.lexical_backend = "like"
        current = self._meta_get("schema_version")
        if current is None:
            self._meta_set("schema_version", str(SCHEMA_VERSION))
        elif int(current) > SCHEMA_VERSION:
            raise RuntimeError(
                f"memory store schema {current} is newer than supported {SCHEMA_VERSION}"
            )
        else:
            # Additive migrations live here; version 3 adds the embedding column.
            self._meta_set("schema_version", str(SCHEMA_VERSION))
        self.conn.commit()

    def _meta_get(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _meta_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------ writes

    def add_memory(self, record: MemoryRecord, *, replace: bool = True) -> str:
        """Insert or update a memory record and keep the FTS index in sync."""
        payload = (
            record.memory_id,
            record.kind,
            record.content,
            record.summary,
            record.source,
            float(record.confidence),
            float(record.created_at),
            float(record.last_used_at),
            int(record.use_count),
            record.supersedes,
            record.superseded_by,
            json.dumps(list(record.tags)),
            self._encode_embedding(record.embedding),
            json.dumps(record.metadata, sort_keys=True),
            content_hash(record.content),
        )
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        self.conn.execute(
            f"""
            {verb} INTO memories (
                memory_id, kind, content, summary, source, confidence, created_at,
                last_used_at, use_count, supersedes, superseded_by, tags, embedding,
                metadata, content_hash
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            payload,
        )
        if self.lexical_backend == "fts5":
            self.conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (record.memory_id,))
            self.conn.execute(
                "INSERT INTO memory_fts(memory_id, content, summary) VALUES (?,?,?)",
                (record.memory_id, record.content, record.summary),
            )
        self.conn.commit()
        return record.memory_id

    def add_many(self, records: Iterable[MemoryRecord]) -> int:
        count = 0
        for record in records:
            self.add_memory(record)
            count += 1
        return count

    def mark_used(self, memory_ids: Sequence[str], now: Optional[float] = None) -> None:
        """Record retrieval usage (powers the freshness/recency reranker)."""
        now = float(time.time() if now is None else now)
        self.conn.executemany(
            "UPDATE memories SET last_used_at = ?, use_count = use_count + 1 WHERE memory_id = ?",
            [(now, mid) for mid in memory_ids],
        )
        self.conn.commit()

    def supersede(self, old_id: str, new_id: str) -> bool:
        """Link ``old_id`` as superseded by ``new_id`` (reversible, non-destructive)."""
        cur = self.conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE memory_id = ? AND superseded_by IS NULL",
            (new_id, old_id),
        )
        self.conn.execute(
            "UPDATE memories SET supersedes = ? WHERE memory_id = ?", (old_id, new_id)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def restore_supersession(self, memory_id: str) -> bool:
        """Undo a supersession link (consolidation must stay reversible)."""
        cur = self.conn.execute(
            "UPDATE memories SET superseded_by = NULL WHERE memory_id = ?", (memory_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def add_triple(self, triple: SemanticTriple) -> str:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO semantic_triples (
                triple_id, subject, predicate, obj, source_id, confidence,
                created_at, superseded_by
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                triple.triple_id,
                triple.subject,
                triple.predicate,
                triple.obj,
                triple.source_id,
                float(triple.confidence),
                float(triple.created_at),
                triple.superseded_by,
            ),
        )
        self.conn.commit()
        return triple.triple_id

    # ------------------------------------------------------------------- reads

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return None if row is None else self._row_to_record(row)

    def find_by_hash(self, text: str, kind: Optional[str] = None) -> List[MemoryRecord]:
        digest = content_hash(text)
        if kind is None:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE content_hash = ? ORDER BY created_at",
                (digest,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE content_hash = ? AND kind = ? ORDER BY created_at",
                (digest, kind),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def all_memories(
        self, kind: Optional[str] = None, include_superseded: bool = True
    ) -> List[MemoryRecord]:
        clauses, params = [], []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM memories {where} ORDER BY created_at", params
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def lex_search(
        self, query: str, limit: int = 20, kind: Optional[str] = None
    ) -> List[Tuple[MemoryRecord, float]]:
        """Lexical search. Returns ``(record, rank_score)`` with higher meaning better."""
        terms = self._tokens(query)
        if not terms:
            return []
        if self.lexical_backend == "fts5":
            match_expr = " OR ".join(f'"{t}"' for t in terms)
            sql = (
                "SELECT m.*, bm25(memory_fts) AS score FROM memory_fts "
                "JOIN memories m ON m.memory_id = memory_fts.memory_id "
                "WHERE memory_fts MATCH ?"
            )
            params: List[Any] = [match_expr]
            if kind is not None:
                sql += " AND m.kind = ?"
                params.append(kind)
            sql += " ORDER BY score ASC, m.created_at DESC LIMIT ?"
            params.append(int(limit))
            try:
                rows = self.conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                rows = []
            # bm25 is negative-is-better: convert to a positive relevance score
            return [(self._row_to_record(r), -float(r["score"] or 0.0)) for r in rows]

        sql = "SELECT * FROM memories WHERE (" + " OR ".join(
            "lower(content) LIKE ?" for _ in terms
        ) + ")"
        params = [f"%{t.lower()}%" for t in terms]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        rows = self.conn.execute(sql, params).fetchall()
        scored: List[Tuple[MemoryRecord, float]] = []
        for row in rows:
            record = self._row_to_record(row)
            hits = sum(1 for t in terms if t.lower() in record.content.lower())
            scored.append((record, float(hits)))
        scored.sort(key=lambda pair: (-pair[1], -pair[0].created_at))
        return scored[: int(limit)]

    def query_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        include_superseded: bool = False,
    ) -> List[SemanticTriple]:
        clauses, params = [], []
        if subject is not None:
            clauses.append("subject = ?")
            params.append(subject)
        if predicate is not None:
            clauses.append("predicate = ?")
            params.append(predicate)
        if obj is not None:
            clauses.append("obj = ?")
            params.append(obj)
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM semantic_triples {where} ORDER BY created_at DESC", params
        ).fetchall()
        return [
            SemanticTriple(
                subject=r["subject"],
                predicate=r["predicate"],
                obj=r["obj"],
                source_id=r["source_id"],
                confidence=float(r["confidence"]),
                created_at=float(r["created_at"]),
                superseded_by=r["superseded_by"],
                triple_id=r["triple_id"],
            )
            for r in rows
        ]

    def stats(self) -> Dict[str, Any]:
        mem = self.conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
        by_kind = {
            r["kind"]: int(r["c"])
            for r in self.conn.execute(
                "SELECT kind, COUNT(*) AS c FROM memories GROUP BY kind"
            ).fetchall()
        }
        tri = self.conn.execute("SELECT COUNT(*) AS c FROM semantic_triples").fetchone()["c"]
        superseded = self.conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE superseded_by IS NOT NULL"
        ).fetchone()["c"]
        return {
            "schema_version": int(self._meta_get("schema_version") or SCHEMA_VERSION),
            "memories": int(mem),
            "by_kind": by_kind,
            "triples": int(tri),
            "superseded": int(superseded),
            "lexical_backend": self.lexical_backend,
            "path": self.path,
        }

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _tokens(query: str) -> List[str]:
        return [t.lower() for t in _TOKEN_RE.findall(query)]

    @staticmethod
    def _encode_embedding(embedding: Optional[Sequence[float]]) -> Optional[bytes]:
        if embedding is None:
            return None
        import numpy as np

        arr = np.asarray(embedding, dtype=np.float32)
        return arr.tobytes()

    @staticmethod
    def _decode_embedding(blob: Optional[bytes]) -> Optional[List[float]]:
        if blob is None:
            return None
        import numpy as np

        arr = np.frombuffer(blob, dtype=np.float32)
        return [float(x) for x in arr]

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            kind=row["kind"],
            content=row["content"],
            summary=row["summary"],
            source=row["source"],
            confidence=float(row["confidence"]),
            created_at=float(row["created_at"]),
            last_used_at=float(row["last_used_at"]),
            use_count=int(row["use_count"]),
            supersedes=row["supersedes"],
            superseded_by=row["superseded_by"],
            tags=list(json.loads(row["tags"] or "[]")),
            embedding=self._decode_embedding(row["embedding"]),
            metadata=dict(json.loads(row["metadata"] or "{}")),
        )

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "MayaMemoryStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
