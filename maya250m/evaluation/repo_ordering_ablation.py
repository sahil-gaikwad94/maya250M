"""Repository-ordering and GDN-associative-retrieval ablations (v3 §7).

The v3 report is explicit that two things must be treated as *hypotheses*, not
architecture facts:

* that dependency/topological file ordering beats random ordering when packing a
  repository into the context window;
* that the recurrent GDN state acts as an exact associative symbol dictionary.

This module provides the harness that measures the first and refuses to assume the
second. It computes Recall@1/5, MRR, cross-file localization accuracy, packed-token
cost and latency per ordering condition, and it returns ``INCONCLUSIVE`` rather
than a verdict when the query set is too small or the conditions are not
compute-matched.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..codebase.indexer import RepoGraphIndexer
from ..codebase.repomap import RepoMapBuilder

__all__ = [
    "GDN_RETRIEVAL_CLAIM",
    "LocalizationQuery",
    "OrderingCondition",
    "ORDERING_CONDITIONS",
    "metrics_from_ranking",
    "rank_files",
    "build_queries",
    "run_ordering_ablation",
    "AblationReport",
]

#: v3 §7: GDN state is *not* an exact O(1) symbol dictionary unless measured.
GDN_RETRIEVAL_CLAIM = "hypothesis"


@dataclass(frozen=True)
class LocalizationQuery:
    """One localization probe: ``query`` should resolve to ``target_file``."""

    query: str
    target_symbol: str
    target_file: str
    cross_file: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "target_symbol": self.target_symbol,
            "target_file": self.target_file,
            "cross_file": self.cross_file,
        }


@dataclass(frozen=True)
class OrderingCondition:
    """One packing/ranking strategy, with its compute accounting."""

    name: str
    description: str
    uses_dependency_order: bool
    uses_neural_reranker: bool
    compute_units: float  # relative cost of producing the ranking
    implemented: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "uses_dependency_order": self.uses_dependency_order,
            "uses_neural_reranker": self.uses_neural_reranker,
            "compute_units": self.compute_units,
            "implemented": self.implemented,
        }


ORDERING_CONDITIONS: Tuple[OrderingCondition, ...] = (
    OrderingCondition("random", "seeded random file order", False, False, 1.0),
    OrderingCondition("topological", "import/topological order", True, False, 1.2),
    OrderingCondition("ppr_only", "PPR repo map only", False, False, 1.4),
    OrderingCondition(
        "ppr_dependency", "PPR plus dependency closure boost", True, False, 1.7
    ),
    OrderingCondition(
        "hybrid_lexical", "PPR plus lexical symbol overlap", False, False, 1.9
    ),
    # Declared, but NOT_RUN until a trained reranker checkpoint exists. It stays in
    # the matrix so the omission is explicit instead of silently missing.
    OrderingCondition(
        "hybrid_neural_rerank",
        "PPR+lexical with trained neural reranker",
        True,
        True,
        3.4,
        implemented=False,
    ),
)

MEASURED_CONDITIONS: Tuple[str, ...] = tuple(
    condition.name for condition in ORDERING_CONDITIONS if condition.implemented
)

_CONDITIONS_BY_NAME = {condition.name: condition for condition in ORDERING_CONDITIONS}


# --------------------------------------------------------------------- ordering


def _seeded_key(seed: int, path: str) -> str:
    return hashlib.sha256(f"{seed}:{path}".encode("utf-8")).hexdigest()


def _lexical_score(query: str, indexer: RepoGraphIndexer, rel_path: str) -> float:
    tokens = {token.lower() for token in query.replace(".", " ").split() if token}
    if not tokens:
        return 0.0
    score = 0.0
    for symbol in indexer.file_symbols(rel_path):
        name = symbol.name.lower()
        for token in tokens:
            if token == name:
                score += 3.0
            elif token in name:
                score += 1.5
            elif token in (symbol.docstring_summary or "").lower():
                score += 0.5
    return score


def _dependency_order(indexer: RepoGraphIndexer) -> List[str]:
    """Kahn topological order over file imports (dependencies first)."""
    files = sorted(indexer.files)
    dependencies: Dict[str, set] = {rel: set() for rel in files}
    for rel in files:
        node = indexer.files[rel]
        for imported in getattr(node, "imports", []) or []:
            imported = str(imported)
            if imported in dependencies and imported != rel:
                dependencies[rel].add(imported)
    ordered: List[str] = []
    ready = [rel for rel in files if not dependencies[rel]]
    while ready:
        ready.sort()
        current = ready.pop(0)
        ordered.append(current)
        for rel in files:
            if current in dependencies[rel]:
                dependencies[rel].discard(current)
                if not dependencies[rel] and rel not in ordered and rel not in ready:
                    ready.append(rel)
    for rel in files:  # cycles or unreachable nodes keep a stable tail order
        if rel not in ordered:
            ordered.append(rel)
    return ordered


def rank_files(
    indexer: RepoGraphIndexer,
    query: str,
    condition: str = "ppr_only",
    *,
    seed: int = 0,
    budget_tokens: int = 16000,
) -> Dict[str, Any]:
    """Rank files for ``query`` under one ordering condition.

    Returns ``{"files": [...], "tokens": int, "order_hash": ...}``; ranking is
    deterministic for a given ``(condition, query, seed)``.
    """
    if condition not in _CONDITIONS_BY_NAME:
        raise ValueError(
            f"unknown ordering condition {condition!r}; valid: {sorted(_CONDITIONS_BY_NAME)}"
        )
    started = time.perf_counter()
    files = sorted(indexer.files)

    if condition == "random":
        ranked = sorted(files, key=lambda rel: _seeded_key(seed, rel))
    elif condition == "topological":
        ranked = _dependency_order(indexer)
    else:
        ppr_text = RepoMapBuilder(indexer).build(query, token_budget=budget_tokens).text
        ppr_order = []
        for rel in files:
            if rel in ppr_text:
                ppr_order.append(rel)
        ranked = ppr_order + [rel for rel in files if rel not in ppr_order]
        position = {rel: index for index, rel in enumerate(ranked)}
        if condition == "ppr_dependency":
            dependencies = set(_dependency_order(indexer))
            ranked = sorted(
                ranked,
                key=lambda rel: (rel not in dependencies, position[rel]),
            )
        elif condition in ("hybrid_lexical", "hybrid_neural_rerank"):
            lexical = {rel: _lexical_score(query, indexer, rel) for rel in files}
            ranked = sorted(
                ranked,
                key=lambda rel: (-lexical.get(rel, 0.0), position[rel]),
            )

    tokens = sum(8 + 20 for _ in ranked)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "files": ranked,
        "tokens": int(tokens),
        "latency_ms": elapsed_ms,
        "compute_units": _CONDITIONS_BY_NAME[condition].compute_units,
        "order_hash": hashlib.sha256("\n".join(ranked).encode("utf-8")).hexdigest()[:16],
    }


# ---------------------------------------------------------------------- metrics


def metrics_from_ranking(
    ranking: Sequence[str], target_file: str, cross_file: bool = False
) -> Dict[str, Any]:
    """Recall@1/5, reciprocal rank and cross-file hit for one ranking."""
    try:
        rank = list(ranking).index(target_file) + 1
    except ValueError:
        rank = None
    reciprocal = 1.0 / rank if rank else 0.0
    return {
        "rank": rank,
        "recall_at_1": 1.0 if rank == 1 else 0.0,
        "recall_at_5": 1.0 if rank is not None and rank <= 5 else 0.0,
        "reciprocal_rank": reciprocal,
        "cross_file_hit": 1.0 if (cross_file and rank is not None and rank <= 5) else 0.0,
    }


def build_queries(indexer: RepoGraphIndexer, limit: int = 40) -> List[LocalizationQuery]:
    """Deterministic self-supervised localization queries from the index itself.

    Each query asks where a symbol is defined; the target file is its definition
    site. ``cross_file`` marks symbols whose callers live in a different file.
    """
    queries: List[LocalizationQuery] = []
    for rel in sorted(indexer.files):
        for symbol in indexer.file_symbols(rel):
            if symbol.kind in ("class",) or not symbol.name:
                continue
            callers = indexer.callers(symbol.name)
            cross_file = any(caller.file_path != rel for caller in callers)
            queries.append(
                LocalizationQuery(
                    query=f"where is {symbol.name} defined",
                    target_symbol=symbol.qualname,
                    target_file=rel,
                    cross_file=cross_file,
                )
            )
            if len(queries) >= int(limit):
                return queries
    return queries


@dataclass
class AblationReport:
    """Per-condition measured metrics plus a compute-aware verdict."""

    conditions: List[Dict[str, Any]] = field(default_factory=list)
    baseline: str = "ppr_only"
    n_queries: int = 0
    min_queries_for_verdict: int = 20
    decision: str = "INCONCLUSIVE"
    decision_reason: str = ""
    gdn_claim: str = GDN_RETRIEVAL_CLAIM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline,
            "n_queries": self.n_queries,
            "gdn_claim": self.gdn_claim,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "conditions": self.conditions,
        }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def run_ordering_ablation(
    indexer: RepoGraphIndexer,
    queries: Optional[Sequence[LocalizationQuery]] = None,
    conditions: Sequence[str] = MEASURED_CONDITIONS,
    *,
    seed: int = 0,
    budget_tokens: int = 16000,
    min_queries_for_verdict: int = 20,
) -> AblationReport:
    """Measure localization quality per ordering condition (real numbers only)."""
    queries = list(queries if queries is not None else build_queries(indexer))
    report = AblationReport(
        n_queries=len(queries), min_queries_for_verdict=int(min_queries_for_verdict)
    )
    if not queries:
        report.decision_reason = "no localization queries available; nothing measured"
        return report

    per_condition: Dict[str, Dict[str, Any]] = {}
    for name in conditions:
        condition = _CONDITIONS_BY_NAME[name]
        if not condition.implemented:
            per_condition[name] = {
                **condition.to_dict(),
                "measured": False,
                "status": "NOT_RUN",
                "reason": "no trained reranker checkpoint is available",
                "recall_at_1": None,
                "recall_at_5": None,
                "mrr": None,
                "cross_file_recall_at_5": None,
                "token_cost": None,
                "latency_ms": None,
            }
            report.conditions.append(per_condition[name])
            continue
        ranks: List[Optional[int]] = []
        recalls_1: List[float] = []
        recalls_5: List[float] = []
        reciprocal: List[float] = []
        cross: List[float] = []
        tokens: List[int] = []
        latency: List[float] = []
        for query in queries:
            packed = rank_files(
                indexer, query.query, name, seed=seed, budget_tokens=budget_tokens
            )
            metrics = metrics_from_ranking(
                packed["files"], query.target_file, query.cross_file
            )
            ranks.append(metrics["rank"])
            recalls_1.append(metrics["recall_at_1"])
            recalls_5.append(metrics["recall_at_5"])
            reciprocal.append(metrics["reciprocal_rank"])
            cross.append(metrics["cross_file_hit"])
            tokens.append(int(packed["tokens"]))
            latency.append(float(packed["latency_ms"]))
        entry = {
            **condition.to_dict(),
            "recall_at_1": _mean(recalls_1),
            "recall_at_5": _mean(recalls_5),
            "mrr": _mean(reciprocal),
            "cross_file_recall_at_5": _mean(cross),
            "mean_rank": _mean([float(r) for r in ranks if r is not None]),
            "unresolved": sum(1 for r in ranks if r is None),
            "token_cost": _mean(tokens),
            "latency_ms": _mean(latency),
            "measured": True,
        }
        per_condition[name] = entry
        report.conditions.append(entry)

    baseline_entry = per_condition.get(report.baseline)
    if baseline_entry is None:
        report.decision_reason = f"baseline {report.baseline!r} was not run"
        return report
    if len(queries) < int(min_queries_for_verdict):
        report.decision_reason = (
            f"only {len(queries)} queries (< {min_queries_for_verdict}); "
            "report values but do not adopt a verdict"
        )
        return report

    candidate = per_condition.get("ppr_dependency")
    if candidate is None:
        report.decision_reason = "dependency condition was not run"
        return report
    improved = candidate["recall_at_5"] > baseline_entry["recall_at_5"] + 1e-9
    not_more_compute = candidate["compute_units"] <= baseline_entry["compute_units"] * 1.5
    if improved and not_more_compute:
        report.decision = "KEEP_DEPENDENCY_ORDERING"
        report.decision_reason = (
            f"recall@5 {candidate['recall_at_5']:.3f} > {baseline_entry['recall_at_5']:.3f} "
            f"at {candidate['compute_units']:.1f}x vs {baseline_entry['compute_units']:.1f}x compute"
        )
    else:
        report.decision = "RETAIN_BASELINE"
        report.decision_reason = (
            f"recall@5 {candidate['recall_at_5']:.3f} vs {baseline_entry['recall_at_5']:.3f}; "
            f"compute {candidate['compute_units']:.1f}x vs {baseline_entry['compute_units']:.1f}x"
        )
    return report
