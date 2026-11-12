"""Reversible deduplication, temporal supersession and conflict tagging.

Consolidation never destroys data. Duplicates and stale facts are *linked* through
``superseded_by``, and every operation is recorded in a report that
:meth:`MemoryConsolidator.undo` can replay backwards. That makes an automated
memory merge auditable and safe to run on the user's durable store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .schema import MemoryRecord, SemanticTriple, content_hash
from .store import MayaMemoryStore

__all__ = ["ConsolidationReport", "MemoryConsolidator"]


@dataclass
class ConsolidationReport:
    """What a consolidation pass changed, and how to undo it."""

    duplicates_merged: int = 0
    triples_superseded: int = 0
    conflicts: List[Dict[str, str]] = field(default_factory=list)
    supersession_ops: List[Dict[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duplicates_merged": self.duplicates_merged,
            "triples_superseded": self.triples_superseded,
            "conflicts": list(self.conflicts),
            "supersession_ops": list(self.supersession_ops),
            "notes": list(self.notes),
        }


class MemoryConsolidator:
    """Deduplicate memories and resolve stale relational facts, reversibly."""

    def __init__(self, store: MayaMemoryStore):
        self.store = store

    # ---------------------------------------------------------------- memories

    def dedupe(self, kind: Optional[str] = None) -> ConsolidationReport:
        """Merge records with identical normalised content.

        The survivor is the *earliest* record (stable provenance); later copies are
        superseded by it so retrieval stops returning duplicates.
        """
        report = ConsolidationReport()
        records = self.store.all_memories(kind=kind)
        groups: Dict[str, List[MemoryRecord]] = {}
        for record in records:
            groups.setdefault(content_hash(record.content), []).append(record)

        for digest, group in groups.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda r: (r.created_at, r.memory_id))
            survivor = group[0]
            for duplicate in group[1:]:
                if duplicate.superseded_by is not None:
                    continue
                if self.store.supersede(duplicate.memory_id, survivor.memory_id):
                    report.duplicates_merged += 1
                    report.supersession_ops.append(
                        {"old": duplicate.memory_id, "new": survivor.memory_id, "what": "memory"}
                    )
        return report

    # ----------------------------------------------------------------- triples

    def consolidate_triples(self, conflict_confidence_gap: float = 0.1) -> ConsolidationReport:
        """Newest fact wins per ``(subject, predicate)``; close calls are conflicts."""
        report = ConsolidationReport()
        triples = self.store.query_triples(include_superseded=False)
        groups: Dict[tuple, List[SemanticTriple]] = {}
        for triple in triples:
            groups.setdefault((triple.subject, triple.predicate), []).append(triple)

        for (_subject, _predicate), group in groups.items():
            distinct = {t.obj for t in group}
            if len(distinct) < 2:
                continue
            group.sort(key=lambda t: (t.created_at, t.triple_id), reverse=True)
            newest = group[0]
            for older in group[1:]:
                if older.obj == newest.obj:
                    # same value, older provenance: supersede silently
                    if self._supersede_triple(older.triple_id, newest.triple_id):
                        report.triples_superseded += 1
                        report.supersession_ops.append(
                            {
                                "old": older.triple_id,
                                "new": newest.triple_id,
                                "what": "triple",
                            }
                        )
                    continue
                report.conflicts.append(
                    {
                        "subject": newest.subject,
                        "predicate": newest.predicate,
                        "older_obj": older.obj,
                        "newer_obj": newest.obj,
                        "kind": "value_conflict",
                    }
                )
                if abs(float(newest.confidence) - float(older.confidence)) >= float(
                    conflict_confidence_gap
                ):
                    if self._supersede_triple(older.triple_id, newest.triple_id):
                        report.triples_superseded += 1
                        report.supersession_ops.append(
                            {
                                "old": older.triple_id,
                                "new": newest.triple_id,
                                "what": "triple",
                            }
                        )
                else:
                    report.notes.append(
                        f"ambiguous conflict on ({newest.subject}, {newest.predicate}): "
                        "confidence gap too small to auto-supersede"
                    )
        return report

    def _supersede_triple(self, old_id: str, new_id: str) -> bool:
        cur = self.store.conn.execute(
            "UPDATE semantic_triples SET superseded_by = ? "
            "WHERE triple_id = ? AND superseded_by IS NULL",
            (new_id, old_id),
        )
        self.store.conn.commit()
        return cur.rowcount > 0

    def find_conflicts(self) -> List[Dict[str, str]]:
        """Report every ``(subject, predicate)`` currently holding >1 distinct value."""
        conflicts: List[Dict[str, str]] = []
        groups: Dict[tuple, set] = {}
        for triple in self.store.query_triples(include_superseded=False):
            groups.setdefault((triple.subject, triple.predicate), set()).add(triple.obj)
        for (subject, predicate), objects in groups.items():
            if len(objects) > 1:
                conflicts.append(
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "values": " | ".join(sorted(objects)),
                    }
                )
        return conflicts

    # -------------------------------------------------------------------- undo

    def undo(self, report: ConsolidationReport) -> int:
        """Reverse every supersession recorded in ``report`` (returns count undone)."""
        undone = 0
        for op in report.supersession_ops:
            if op.get("what") == "memory":
                if self.store.restore_supersession(op["old"]):
                    undone += 1
            else:
                cur = self.store.conn.execute(
                    "UPDATE semantic_triples SET superseded_by = NULL WHERE triple_id = ?",
                    (op["old"],),
                )
                self.store.conn.commit()
                undone += int(cur.rowcount > 0)
        return undone
