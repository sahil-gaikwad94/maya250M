"""Data provenance, deduplication and leakage prevention (v3 §8.1–8.3).

A 250M model cannot hide a contaminated corpus behind parameter count, so this
module makes every record carry provenance and makes the dedup/leakage decisions
explicit and reproducible:

* exact SHA-256 deduplication;
* normalized-text deduplication (case, whitespace, punctuation insensitive);
* near-duplicate detection with MinHash + LSH (documented, deterministic);
* code-clone detection that strips comments/docstrings/blank lines but keeps
  identifiers and literals, so semantically different code is not merged;
* repository/document-family grouping so a held-out split cannot leak through a
  sibling record from the same repository or paper family;
* a report that records *why* each record was dropped instead of silently
  discarding raw inputs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "DOMAINS",
    "Provenance",
    "ProvenanceError",
    "DataRecord",
    "build_record",
    "normalize_text",
    "code_clone_key",
    "DedupReport",
    "dedup_records",
    "near_duplicate_groups",
    "leakage_report",
    "LeakageError",
    "assert_no_group_leakage",
]

DOMAINS = ("code", "math", "science", "general", "research", "engineering")
_KNOWN_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "public-domain",
    "odc-by",
    "cdla-permissive-1.0",
    "proprietary-authorized",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_CODE_COMMENT_RE = re.compile(r"#[^\n]*")
_CODE_BLOCK_COMMENT_RE = re.compile(r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'')
_CODE_DOCSTRING_LINE_RE = re.compile(r'^[ \t]*("""|\'\'\').*$')


class ProvenanceError(ValueError):
    """A record cannot be used because provenance is missing or invalid."""


@dataclass(frozen=True)
class Provenance:
    """Per-record provenance required before a record may enter training."""

    source_id: str
    license: str
    domain: str
    language: str = "en"
    source_uri: Optional[str] = None
    quality_score: float = 0.0
    timestamp: Optional[str] = None
    content_sha256: str = ""
    split_group: str = ""
    is_synthetic: bool = False
    generator_version: Optional[str] = None
    verifier_version: Optional[str] = None

    def validate(self, *, allow_unknown_license: bool = False) -> None:
        if not self.source_id:
            raise ProvenanceError("source_id is required")
        if not self.split_group:
            raise ProvenanceError(
                f"record {self.source_id!r} has no split_group (repository/document family)"
            )
        if self.domain not in DOMAINS:
            raise ProvenanceError(
                f"record {self.source_id!r} has unknown domain {self.domain!r}; "
                f"expected one of {DOMAINS}"
            )
        license_name = (self.license or "").strip().lower()
        if not license_name:
            raise ProvenanceError(f"record {self.source_id!r} has no license")
        if not allow_unknown_license and license_name not in _KNOWN_LICENSES:
            raise ProvenanceError(
                f"record {self.source_id!r} carries unverified license {self.license!r}; "
                "refusing to train on it"
            )
        if not 0.0 <= float(self.quality_score) <= 1.0:
            raise ProvenanceError("quality_score must be in [0, 1]")
        if self.content_sha256 and not _SHA256_RE.match(self.content_sha256):
            raise ProvenanceError("content_sha256 must be 64 lowercase hex characters")
        if self.is_synthetic and not self.generator_version:
            raise ProvenanceError(
                f"synthetic record {self.source_id!r} must name generator_version"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_uri": self.source_uri,
            "license": self.license,
            "language": self.language,
            "domain": self.domain,
            "quality_score": float(self.quality_score),
            "timestamp": self.timestamp,
            "content_sha256": self.content_sha256,
            "split_group": self.split_group,
            "is_synthetic": bool(self.is_synthetic),
            "generator_version": self.generator_version,
            "verifier_version": self.verifier_version,
        }


@dataclass
class DataRecord:
    """One training document with provenance and an explicit split."""

    record_id: str
    text: str
    provenance: Provenance
    split: str = "train"
    quality_flags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ProvenanceError("record_id is required")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ProvenanceError(f"record {self.record_id!r} is empty")

    @property
    def domain(self) -> str:
        return self.provenance.domain

    @property
    def split_group(self) -> str:
        return self.provenance.split_group

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.strip().encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "split": self.split,
            "domain": self.domain,
            "text_sha256": self.content_sha256,
            "provenance": self.provenance.to_dict(),
            "quality_flags": list(self.quality_flags),
        }


def build_record(
    text: str,
    *,
    source_id: str,
    license: str,
    domain: str,
    split_group: str,
    record_id: Optional[str] = None,
    split: str = "train",
    quality_score: float = 0.5,
    is_synthetic: bool = False,
    generator_version: Optional[str] = None,
    allow_unknown_license: bool = False,
    **kwargs: Any,
) -> DataRecord:
    """Validate provenance and attach the content hash."""
    provenance = Provenance(
        source_id=source_id,
        license=license,
        domain=domain,
        split_group=split_group,
        quality_score=quality_score,
        is_synthetic=is_synthetic,
        generator_version=generator_version,
        content_sha256=hashlib.sha256(text.strip().encode("utf-8")).hexdigest(),
        **kwargs,
    )
    provenance.validate(allow_unknown_license=allow_unknown_license)
    return DataRecord(
        record_id=record_id or provenance.content_sha256[:16],
        text=text,
        provenance=provenance,
        split=split,
    )


# ----------------------------------------------------------------- normalisation


def normalize_text(text: str) -> str:
    """Case/whitespace/punctuation-insensitive representation for dedup."""
    lowered = text.lower().replace("\r\n", "\n")
    ascii_only = _NON_ALNUM_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", ascii_only).strip()


def code_clone_key(text: str) -> str:
    """Clone key that ignores comments/formatting but keeps semantics.

    Identifiers and literals are preserved on purpose: two functions that differ
    only in a variable name are *not* duplicates, while the same function with
    different comments is.
    """
    without_blocks = _CODE_BLOCK_COMMENT_RE.sub("", text)
    lines = []
    for line in without_blocks.replace("\r\n", "\n").splitlines():
        if _CODE_DOCSTRING_LINE_RE.match(line):
            continue
        stripped = _CODE_COMMENT_RE.sub("", line).rstrip()
        if not stripped.strip():
            continue
        lines.append(_WS_RE.sub(" ", stripped.strip()))
    return "\n".join(lines)


def _dedup_key(record: DataRecord, *, domain_aware: bool = True) -> str:
    if domain_aware and record.domain == "code":
        return code_clone_key(record.text)
    return normalize_text(record.text)


# --------------------------------------------------------------------- MinHash


def _shingles(text: str, size: int = 5) -> Set[str]:
    tokens = text.split()
    if len(tokens) <= size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _minhash(shingles: Set[str], permutations: int = 64) -> Tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in range(permutations))
    signature: List[int] = []
    for seed in range(permutations):
        prefix = f"{seed}:".encode("utf-8")
        signature.append(
            min(
                int.from_bytes(hashlib.blake2b(prefix + shingle.encode("utf-8"), digest_size=8).digest(), "big")
                for shingle in shingles
            )
        )
    return tuple(signature)


def near_duplicate_groups(
    records: Sequence[DataRecord],
    *,
    threshold: float = 0.85,
    shingle_size: int = 5,
    permutations: int = 64,
    bands: int = 16,
) -> List[List[int]]:
    """MinHash/LSH near-duplicate groups (indices into ``records``)."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if permutations % bands:
        rows = max(1, permutations // bands)
        permutations = rows * bands
    rows = permutations // bands

    signatures = [
        _minhash(_shingles(_dedup_key(record), shingle_size), permutations)
        for record in records
    ]
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    for index, signature in enumerate(signatures):
        for band in range(bands):
            key = (band, signature[band * rows : (band + 1) * rows])
            buckets.setdefault(key, []).append(index)

    parent = list(range(len(records)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    comparable = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                comparable.append((members[i], members[j]))
    for left, right in comparable:
        same = sum(
            1 for a, b in zip(signatures[left], signatures[right]) if a == b
        )
        if same / permutations >= threshold:
            union(left, right)

    grouped: Dict[int, List[int]] = {}
    for index in range(len(records)):
        grouped.setdefault(find(index), []).append(index)
    return [members for members in grouped.values() if len(members) > 1]


@dataclass
class DedupReport:
    """What was kept, what was dropped and why."""

    kept: List[DataRecord] = field(default_factory=list)
    exact_duplicates: List[str] = field(default_factory=list)
    near_duplicates: List[Dict[str, Any]] = field(default_factory=list)
    group_duplicates: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_kept": len(self.kept),
            "exact_duplicates": list(self.exact_duplicates),
            "near_duplicates": list(self.near_duplicates),
            "group_duplicates": list(self.group_duplicates),
            "stats": dict(self.stats),
            "kept_ids": [record.record_id for record in self.kept],
            "corpus_sha256": corpus_hash(self.kept),
        }


def corpus_hash(records: Sequence[DataRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.record_id):
        digest.update(f"{record.record_id}\x00{record.content_sha256}\n".encode("utf-8"))
    return digest.hexdigest()


def dedup_records(
    records: Sequence[DataRecord],
    *,
    near_threshold: float = 0.85,
    shingle_size: int = 5,
    group_dedup: bool = False,
) -> DedupReport:
    """Exact + normalized + near-duplicate + group deduplication."""
    report = DedupReport()
    seen: Set[str] = set()
    unique: List[DataRecord] = []
    for record in records:
        record.provenance.validate()
        key = _dedup_key(record)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if digest in seen:
            report.exact_duplicates.append(record.record_id)
            continue
        seen.add(digest)
        unique.append(record)

    groups = near_duplicate_groups(
        unique, threshold=near_threshold, shingle_size=shingle_size
    )
    dropped: Set[int] = set()
    for members in groups:
        ordered = sorted(members, key=lambda index: (-unique[index].provenance.quality_score, index))
        keeper = ordered[0]
        for index in ordered[1:]:
            if index in dropped:
                continue
            dropped.add(index)
            report.near_duplicates.append(
                {
                    "record_id": unique[index].record_id,
                    "near_duplicate_of": unique[keeper].record_id,
                    "group_size": len(members),
                    "reason": "minhash_lsh",
                }
            )
    kept = [record for index, record in enumerate(unique) if index not in dropped]

    if group_dedup:
        # Group-level duplication is not "one document per repository": it only
        # collapses records that are the same normalized document twice inside one
        # group (already exact-deduped above), so this stays an explicit no-op hook
        # rather than an aggressive filter that would delete real training data.
        pass

    report.kept = kept
    report.stats = {
        "n_input": len(records),
        "n_kept": len(kept),
        "n_exact_duplicates": len(report.exact_duplicates),
        "n_near_duplicates": len(report.near_duplicates),
        "n_group_duplicates": len(report.group_duplicates),
        "near_threshold": near_threshold,
        "shingle_size": shingle_size,
    }
    return report


# -------------------------------------------------------------------- leakage


class LeakageError(ValueError):
    """A held-out group also appears in training data."""


def leakage_report(
    records: Sequence[DataRecord], heldout_splits: Sequence[str] = ("heldout", "test")
) -> Dict[str, Any]:
    """Groups shared between training and held-out splits (must be empty)."""
    train_groups: Set[str] = set()
    heldout_groups: Set[str] = set()
    heldout_ids: Dict[str, List[str]] = {}
    for record in records:
        if record.split in heldout_splits:
            heldout_groups.add(record.split_group)
            heldout_ids.setdefault(record.split_group, []).append(record.record_id)
        else:
            train_groups.add(record.split_group)
    shared = sorted(train_groups & heldout_groups)
    return {
        "shared_groups": shared,
        "n_train_groups": len(train_groups),
        "n_heldout_groups": len(heldout_groups),
        "heldout_record_ids": {group: ids for group, ids in heldout_ids.items() if group in shared},
        "clean": not shared,
    }


def assert_no_group_leakage(
    records: Sequence[DataRecord], heldout_splits: Sequence[str] = ("heldout", "test")
) -> Dict[str, Any]:
    """Raise :class:`LeakageError` when a group crosses the split boundary."""
    report = leakage_report(records, heldout_splits)
    if not report["clean"]:
        raise LeakageError(
            "held-out split groups also appear in training data: "
            f"{report['shared_groups']}"
        )
    return report
