"""Benchmark / dataset contamination scanning (v3 §8.6).

A contamination check is the difference between a benchmark score and a
memorisation score. This module makes that check explicit and reproducible:

* exact normalised containment (a held-out document that is a substring of, or
  equal to, a training document after normalisation);
* n-gram overlap scanning (shared n-grams above a threshold);
* a report that names *which* records overlap and *why*;
* :func:`protect_heldout`, which removes contaminated training records instead
  of silently trusting the benchmark.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from ..training.dedup import DataRecord, normalize_text

__all__ = [
    "ContaminationError",
    "ContaminationReport",
    "ngram_set",
    "ngram_overlap",
    "exact_containment",
    "contamination_scan",
    "scan_contamination",
    "detect_contamination",
    "assert_no_contamination",
    "protect_heldout",
    "ProtectedDataset",
]


class ContaminationError(ValueError):
    """Raised when contamination is detected or inputs are invalid."""


@dataclass
class ContaminationReport:
    """What overlapped between training data and held-out/benchmark data."""

    train_size: int = 0
    heldout_size: int = 0
    benchmark_size: int = 0
    ngram_n: int = 5
    min_overlap: int = 1
    exact_matches: List[Dict[str, Any]] = field(default_factory=list)
    ngram_overlaps: List[Dict[str, Any]] = field(default_factory=list)
    contaminated_train_ids: List[str] = field(default_factory=list)
    contaminated_heldout_ids: List[str] = field(default_factory=list)
    contaminated_benchmark_ids: List[str] = field(default_factory=list)
    shared_ngrams: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def contaminated(self) -> bool:
        return bool(
            self.exact_matches
            or self.ngram_overlaps
            or self.contaminated_train_ids
            or self.contaminated_heldout_ids
            or self.contaminated_benchmark_ids
        )

    @property
    def clean(self) -> bool:
        return not self.contaminated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_size": self.train_size,
            "heldout_size": self.heldout_size,
            "benchmark_size": self.benchmark_size,
            "ngram_n": self.ngram_n,
            "min_overlap": self.min_overlap,
            "contaminated": self.contaminated,
            "clean": self.clean,
            "exact_matches": list(self.exact_matches),
            "ngram_overlaps": list(self.ngram_overlaps),
            "contaminated_train_ids": list(self.contaminated_train_ids),
            "contaminated_heldout_ids": list(self.contaminated_heldout_ids),
            "contaminated_benchmark_ids": list(self.contaminated_benchmark_ids),
            "shared_ngrams": list(self.shared_ngrams),
            "details": dict(self.details),
        }


@dataclass
class ProtectedDataset(list):
    """A filtered training list that also carries its contamination report."""

    report: Optional[ContaminationReport] = None


# ----------------------------------------------------------------- helpers


def _extract(item: Any) -> Tuple[str, str]:
    """Return ``(id, text)`` for a record, string, or dict."""
    if isinstance(item, DataRecord):
        return str(item.record_id), str(item.text)
    if isinstance(item, str):
        return "", item
    if isinstance(item, dict):
        text = item.get("text") or item.get("content") or item.get("document") or ""
        rid = item.get("id") or item.get("record_id") or item.get("name") or ""
        return str(rid), str(text)
    rid = getattr(item, "record_id", None) or getattr(item, "id", None) or ""
    text = getattr(item, "text", None) or getattr(item, "content", None) or str(item)
    return str(rid), str(text)


def _ids(items: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for index, item in enumerate(items):
        rid, _ = _extract(item)
        out.append(rid if rid else f"idx:{index}")
    return out


def ngram_set(text: str, n: int = 5, *, char: bool = False) -> Set[str]:
    """Return the set of n-grams of ``text`` (word-level by default)."""
    if n <= 0:
        raise ValueError("n must be positive")
    normalized = normalize_text(text)
    if char:
        tokens = list(normalized.replace(" ", ""))
    else:
        tokens = normalized.split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def ngram_overlap(
    left: str, right: str, n: int = 5, *, char: bool = False
) -> Dict[str, Any]:
    """Shared n-grams between two texts."""
    left_grams = ngram_set(left, n, char=char)
    right_grams = ngram_set(right, n, char=char)
    shared = sorted(left_grams & right_grams)
    union = left_grams | right_grams
    jaccard = (len(shared) / len(union)) if union else 0.0
    return {
        "shared_count": len(shared),
        "shared_ngrams": shared,
        "jaccard": jaccard,
        "left_size": len(left_grams),
        "right_size": len(right_grams),
    }


def exact_containment(left: str, right: str) -> bool:
    """True if either normalised text contains the other (non-empty)."""
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return False
    if a == b:
        return True
    return a in b or b in a


# ----------------------------------------------------------------- resolution


def _resolve_inputs(
    train: Any,
    heldout: Any,
    benchmarks: Any,
    kwargs: Dict[str, Any],
) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
    if train is None:
        train = kwargs.pop("train_records", None) or kwargs.pop("train_texts", None) or kwargs.pop("training", None)
    if heldout is None:
        heldout = kwargs.pop("heldout_records", None) or kwargs.pop("heldout_texts", None) or kwargs.pop("heldout", None) or kwargs.pop("validation", None)
    if benchmarks is None:
        benchmarks = (
            kwargs.pop("benchmark_records", None)
            or kwargs.pop("benchmark_texts", None)
            or kwargs.pop("benchmarks", None)
            or kwargs.pop("benchmark", None)
        )
    train = train if train is not None else []
    heldout = heldout if heldout is not None else []
    benchmarks = benchmarks if benchmarks is not None else []
    return train, heldout, benchmarks


def contamination_scan(
    train: Any = None,
    heldout: Any = None,
    benchmarks: Any = None,
    *,
    n: int = 5,
    ngram_n: Optional[int] = None,
    min_overlap: int = 1,
    char_ngrams: bool = False,
    train_records: Any = None,
    heldout_records: Any = None,
    benchmark_records: Any = None,
    train_texts: Any = None,
    heldout_texts: Any = None,
    benchmark_texts: Any = None,
    **kwargs: Any,
) -> ContaminationReport:
    """Scan training data for contamination against held-out and benchmark data.

    Accepts records, dicts, or raw strings. Returns a :class:`ContaminationReport`.
    """
    if ngram_n is not None:
        n = int(ngram_n)
    if n <= 0:
        raise ContaminationError("n must be positive")
    if min_overlap <= 0:
        raise ContaminationError("min_overlap must be positive")

    # Allow keyword aliases through kwargs too.
    kwargs.setdefault("train_records", train_records)
    kwargs.setdefault("heldout_records", heldout_records)
    kwargs.setdefault("benchmark_records", benchmark_records)
    kwargs.setdefault("train_texts", train_texts)
    kwargs.setdefault("heldout_texts", heldout_texts)
    kwargs.setdefault("benchmark_texts", benchmark_texts)

    train_seq, heldout_seq, benchmark_seq = _resolve_inputs(train, heldout, benchmarks, kwargs)

    train_items = list(train_seq)
    heldout_items = list(heldout_seq)
    benchmark_items = list(benchmark_seq)

    report = ContaminationReport(
        train_size=len(train_items),
        heldout_size=len(heldout_items),
        benchmark_size=len(benchmark_items),
        ngram_n=n,
        min_overlap=min_overlap,
    )

    train_norms = [_extract(item) for item in train_items]
    heldout_norms = [_extract(item) for item in heldout_items]
    benchmark_norms = [_extract(item) for item in benchmark_items]

    contaminated_train: Set[str] = set()
    contaminated_heldout: Set[str] = set()
    contaminated_benchmark: Set[str] = set()
    shared_grams: Set[str] = set()

    # --- exact normalised containment --------------------------------------
    for ti, (t_id, t_text) in enumerate(train_norms):
        t_norm = normalize_text(t_text)
        for hi, (h_id, h_text) in enumerate(heldout_norms):
            if exact_containment(t_text, h_text):
                report.exact_matches.append(
                    {
                        "train_id": t_id if t_id else f"train:{ti}",
                        "heldout_id": h_id if h_id else f"heldout:{hi}",
                        "kind": "heldout",
                        "reason": "exact_normalized_containment",
                    }
                )
                contaminated_train.add(t_id if t_id else f"train:{ti}")
                contaminated_heldout.add(h_id if h_id else f"heldout:{hi}")
        for bi, (b_id, b_text) in enumerate(benchmark_norms):
            if exact_containment(t_text, b_text):
                report.exact_matches.append(
                    {
                        "train_id": t_id if t_id else f"train:{ti}",
                        "benchmark_id": b_id if b_id else f"benchmark:{bi}",
                        "kind": "benchmark",
                        "reason": "exact_normalized_containment",
                    }
                )
                contaminated_train.add(t_id if t_id else f"train:{ti}")
                contaminated_benchmark.add(b_id if b_id else f"benchmark:{bi}")

    # --- n-gram overlap -----------------------------------------------------
    train_grams = [ngram_set(text, n, char=char_ngrams) for _, text in train_norms]
    for hi, (h_id, h_text) in enumerate(heldout_norms):
        h_grams = ngram_set(h_text, n, char=char_ngrams)
        for ti, (t_id, t_text) in enumerate(train_norms):
            shared = train_grams[ti] & h_grams
            if len(shared) >= min_overlap:
                t_key = t_id if t_id else f"train:{ti}"
                h_key = h_id if h_id else f"heldout:{hi}"
                report.ngram_overlaps.append(
                    {
                        "train_id": t_key,
                        "heldout_id": h_key,
                        "kind": "heldout",
                        "shared_count": len(shared),
                        "shared_ngrams": sorted(shared)[:50],
                        "reason": "ngram_overlap",
                    }
                )
                contaminated_train.add(t_key)
                contaminated_heldout.add(h_key)
                shared_grams.update(shared)

    for bi, (b_id, b_text) in enumerate(benchmark_norms):
        b_grams = ngram_set(b_text, n, char=char_ngrams)
        for ti, (t_id, t_text) in enumerate(train_norms):
            shared = train_grams[ti] & b_grams
            if len(shared) >= min_overlap:
                t_key = t_id if t_id else f"train:{ti}"
                b_key = b_id if b_id else f"benchmark:{bi}"
                report.ngram_overlaps.append(
                    {
                        "train_id": t_key,
                        "benchmark_id": b_key,
                        "kind": "benchmark",
                        "shared_count": len(shared),
                        "shared_ngrams": sorted(shared)[:50],
                        "reason": "ngram_overlap",
                    }
                )
                contaminated_train.add(t_key)
                contaminated_benchmark.add(b_key)
                shared_grams.update(shared)

    report.contaminated_train_ids = sorted(contaminated_train)
    report.contaminated_heldout_ids = sorted(contaminated_heldout)
    report.contaminated_benchmark_ids = sorted(contaminated_benchmark)
    report.shared_ngrams = sorted(shared_grams)[:200]
    report.details = {
        "n_train_grams": [len(g) for g in train_grams],
        "normalization": "lowercase, punctuation->space, whitespace collapse",
    }
    return report


# Aliases for callers using different naming conventions.
scan_contamination = contamination_scan
detect_contamination = contamination_scan


def assert_no_contamination(*args: Any, **kwargs: Any) -> ContaminationReport:
    """Raise :class:`ContaminationError` if any contamination is found.

    Accepts either a :class:`ContaminationReport` or raw inputs (forwarded to
    :func:`contamination_scan`).
    """
    if args and isinstance(args[0], ContaminationReport):
        report = args[0]
    else:
        report = contamination_scan(*args, **kwargs)
    if report.contaminated:
        raise ContaminationError(
            "contamination detected: "
            f"{len(report.contaminated_train_ids)} train, "
            f"{len(report.contaminated_heldout_ids)} held-out, "
            f"{len(report.contaminated_benchmark_ids)} benchmark records; "
            f"exact={len(report.exact_matches)}, ngram={len(report.ngram_overlaps)}"
        )
    return report


def protect_heldout(
    train: Any = None,
    heldout: Any = None,
    benchmarks: Any = None,
    *,
    n: int = 5,
    ngram_n: Optional[int] = None,
    min_overlap: int = 1,
    char_ngrams: bool = False,
    remove_contaminated: bool = True,
    **kwargs: Any,
) -> ProtectedDataset:
    """Return training records with contaminated entries removed.

    The returned list subclass carries ``.report`` so callers can inspect what
    was dropped. Inputs may be records, dicts, or strings.
    """
    if ngram_n is not None:
        n = int(ngram_n)
    report = contamination_scan(
        train, heldout, benchmarks, n=n, min_overlap=min_overlap, char_ngrams=char_ngrams, **kwargs
    )
    train_items = list(train if train is not None else kwargs.get("train_records") or kwargs.get("train_texts") or [])
    # If train not provided positionally but in kwargs, resolve.
    if not train_items:
        tr, _, _ = _resolve_inputs(None, None, None, kwargs)
        if tr:
            train_items = list(tr)

    contaminated_ids = set(report.contaminated_train_ids)
    cleaned: List[Any] = []
    for index, item in enumerate(train_items):
        rid, _ = _extract(item)
        key = rid if rid else f"train:{index}"
        if remove_contaminated and key in contaminated_ids:
            continue
        # Also drop by index-key fallback used when no id.
        if remove_contaminated and not rid and f"train:{index}" in contaminated_ids:
            continue
        cleaned.append(item)

    protected = ProtectedDataset(cleaned)
    protected.report = report
    return protected
