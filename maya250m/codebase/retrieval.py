"""Hybrid repository retrieval: BM25 lexical, PPR structural, optional dense.

Localizing an issue in a 100k-line repository cannot rely on one signal. The
issue text names identifiers (lexical), those identifiers live in a structural
neighbourhood of callers/importers (graph), and a semantic embedder could add
paraphrase matching (dense). The three rankings are fused with Reciprocal Rank
Fusion, which combines *ranks* instead of raw scores, so a BM25 score of 12.4 and
a PPR mass of 0.003 can be merged without inventing a normalization.

Honesty constraint: nobody in this project has trained or attached an embedding
model. ``dense`` therefore stays absent unless a ranker is supplied, and every
report says ``dense_ranked: False`` rather than implying semantic retrieval that
never ran. The structural signal is Personalized PageRank over the real symbol
graph, not a placeholder.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .indexer import RepoGraphIndexer, SymbolDef
from .repomap import RepoMapBuilder

__all__ = [
    "RRF_K",
    "DenseRanker",
    "HybridRepoRetriever",
    "RetrievalCandidate",
    "RetrievalContext",
    "RetrievalResult",
    "bm25_scores",
    "reciprocal_rank_fusion",
    "tokenize",
    "tokenize_identifier",
]

#: Standard RRF damping constant; ranks are 1-based.
RRF_K = 60

#: ``(query, documents) -> document indices, best first``.
DenseRanker = Callable[[str, Sequence[str]], Sequence[int]]

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize_identifier(name: str) -> List[str]:
    """Split ``snake_case`` and ``camelCase`` into lowercase sub-tokens."""
    out: List[str] = []
    for part in str(name).split("_"):
        out.extend(match.group(0).lower() for match in _CAMEL.finditer(part))
    return [token for token in out if token]


def tokenize(text: str) -> List[str]:
    """Tokenize source/query text, keeping identifier sub-tokens."""
    tokens: List[str] = []
    for match in _WORD.finditer(text or ""):
        tokens.extend(tokenize_identifier(match.group(0)))
    return tokens


def bm25_scores(
    query_tokens: Sequence[str],
    documents: Sequence[str],
    k1: float = 1.2,
    b: float = 0.75,
) -> List[float]:
    """Okapi BM25 over pre-split ``documents`` (pure Python, deterministic)."""
    n_docs = len(documents)
    if n_docs == 0:
        return []
    doc_tokens = [tokenize(doc) for doc in documents]
    avg_len = (sum(len(toks) for toks in doc_tokens) / n_docs) or 1.0
    doc_freq: Counter = Counter()
    for toks in doc_tokens:
        for term in set(toks):
            doc_freq[term] += 1

    scores = [0.0] * n_docs
    for term in set(query_tokens):
        df = doc_freq.get(term, 0)
        if df == 0:
            continue
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for index, toks in enumerate(doc_tokens):
            tf = toks.count(term)
            if tf == 0:
                continue
            norm = tf + k1 * (1.0 - b + b * len(toks) / avg_len)
            scores[index] += idf * tf * (k1 + 1.0) / norm
    return scores


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = RRF_K
) -> Dict[str, float]:
    """Fuse ranked id lists: ``sum(1 / (k + rank))`` with 1-based ranks."""
    fused: Dict[str, float] = {}
    for ranking in rankings:
        seen = set()
        unique = []
        for key in ranking:
            # Deduplicate before ranking: a repeated id must not push every
            # later id one position down inside the same signal.
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)
        for position, key in enumerate(unique, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + position)
    return fused


@dataclass
class RetrievalCandidate:
    """One ranked symbol span."""

    symbol_id: str
    qualname: str
    kind: str
    file_path: str
    start_line: int
    end_line: int
    score: float
    ranks: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol_id": self.symbol_id,
            "qualname": self.qualname,
            "kind": self.kind,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(float(self.score), 6),
            "ranks": dict(self.ranks),
        }


@dataclass
class RetrievalResult:
    """Fused ranking plus which signals actually ran."""

    query: str
    candidates: List[RetrievalCandidate] = field(default_factory=list)
    signals: Tuple[str, ...] = ()
    dense_ranked: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "query": self.query,
            "candidates": [c.to_dict() for c in self.candidates],
            "signals": list(self.signals),
            "dense_ranked": bool(self.dense_ranked),
        }


@dataclass
class RetrievalContext:
    """Rendered source spans ready to be placed in the model's context window."""

    text: str
    candidates: List[RetrievalCandidate] = field(default_factory=list)
    tokens_used: int = 0
    truncated: bool = False
    dense_ranked: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "candidates": [c.to_dict() for c in self.candidates],
            "tokens_used": int(self.tokens_used),
            "truncated": bool(self.truncated),
            "dense_ranked": bool(self.dense_ranked),
        }


class HybridRepoRetriever:
    """Fuse lexical BM25, structural PPR and an optional dense ranker."""

    def __init__(
        self,
        indexer: RepoGraphIndexer,
        dense_ranker: Optional[DenseRanker] = None,
        rrf_k: int = RRF_K,
        k1: float = 1.2,
        b: float = 0.75,
        structural_iterations: int = 15,
        max_span_lines: int = 400,
    ):
        self.indexer = indexer
        self.dense_ranker = dense_ranker
        self.rrf_k = int(rrf_k)
        self.k1 = float(k1)
        self.b = float(b)
        self.structural_iterations = int(structural_iterations)
        self.max_span_lines = int(max_span_lines)

        self._symbols: List[SymbolDef] = sorted(
            (symbol for node in indexer.files.values() for symbol in node.symbols),
            key=lambda symbol: symbol.symbol_id,
        )
        self._symbol_by_id: Dict[str, SymbolDef] = {
            symbol.symbol_id: symbol for symbol in self._symbols
        }
        self._index_by_id: Dict[str, int] = {
            symbol.symbol_id: position for position, symbol in enumerate(self._symbols)
        }
        self._documents: List[str] = [self._document_text(s) for s in self._symbols]
        self._builder: Optional[RepoMapBuilder] = None

    # -------------------------------------------------------------- preparation

    @staticmethod
    def _document_text(symbol: SymbolDef) -> str:
        return "\n".join(
            part
            for part in (
                symbol.qualname.replace(".", " "),
                symbol.name,
                symbol.signature,
                symbol.docstring_summary,
                symbol.file_path,
            )
            if part
        )

    def _structural_builder(self) -> RepoMapBuilder:
        if self._builder is None:
            self._builder = RepoMapBuilder(
                self.indexer, iterations=self.structural_iterations
            )
        return self._builder

    @staticmethod
    def _count(text: str, tokenizer=None) -> int:
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text, add_bos=False, add_eos=False))
            except Exception:  # pragma: no cover - tokenizer without that API
                pass
        return max(1, (len(text) + 3) // 4)

    # ------------------------------------------------------------------ signals

    def lexical_ranking(self, query: str, limit: Optional[int] = None) -> List[int]:
        """BM25 over symbol documents; only positive scores are returned."""
        query_tokens = tokenize(query)
        if not query_tokens or not self._documents:
            return []
        scores = bm25_scores(query_tokens, self._documents, self.k1, self.b)
        order = sorted(
            (i for i, score in enumerate(scores) if score > 0.0),
            key=lambda i: (-scores[i], self._symbols[i].symbol_id),
        )
        return order[: int(limit)] if limit is not None else order

    def structural_ranking(self, query: str, limit: Optional[int] = None) -> List[int]:
        """Personalized PageRank, keeping symbols above the uniform baseline."""
        if not self._symbols:
            return []
        builder = self._structural_builder()
        ranks = builder.pagerank(builder.personalize(query))
        baseline = 1.0 / max(1, len(ranks))
        scored = [
            (ranks.get(symbol.symbol_id, 0.0), symbol.symbol_id, position)
            for position, symbol in enumerate(self._symbols)
        ]
        scored = [item for item in scored if item[0] > baseline]
        scored.sort(key=lambda item: (-item[0], item[1]))
        indices = [item[2] for item in scored]
        return indices[: int(limit)] if limit is not None else indices

    def dense_ranking(self, query: str, limit: Optional[int] = None) -> Optional[List[int]]:
        """Dense ranking, or ``None`` when no embedder is attached (NOT_RUN)."""
        if self.dense_ranker is None:
            return None
        raw = self.dense_ranker(query, list(self._documents)) or ()
        seen = set()
        indices: List[int] = []
        for item in raw:
            try:
                index = int(item)
            except (TypeError, ValueError):  # pragma: no cover - bad ranker output
                continue
            if 0 <= index < len(self._symbols) and index not in seen:
                seen.add(index)
                indices.append(index)
        return indices[: int(limit)] if limit is not None else indices

    # ------------------------------------------------------------------- fusion

    def search(
        self, query: str, k: int = 8, limit_per_signal: Optional[int] = None
    ) -> RetrievalResult:
        """Rank symbol spans for ``query`` by reciprocal rank fusion."""
        limit = limit_per_signal if limit_per_signal is not None else max(int(k) * 4, 24)
        signals: Dict[str, List[int]] = {
            "lexical": self.lexical_ranking(query, limit),
            "structural": self.structural_ranking(query, limit),
        }
        dense = self.dense_ranking(query, limit)
        if dense is not None:
            signals["dense"] = dense

        rankings = [
            [self._symbols[index].symbol_id for index in indices]
            for indices in signals.values()
        ]
        fused = reciprocal_rank_fusion(rankings, self.rrf_k)

        ranks_by_signal: Dict[str, Dict[str, int]] = {}
        for name, indices in signals.items():
            for position, index in enumerate(indices, start=1):
                ranks_by_signal.setdefault(self._symbols[index].symbol_id, {})[name] = position

        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        candidates: List[RetrievalCandidate] = []
        for symbol_id, score in ordered[: int(k)]:
            symbol = self._symbol_by_id[symbol_id]
            candidates.append(
                RetrievalCandidate(
                    symbol_id=symbol.symbol_id,
                    qualname=symbol.qualname,
                    kind=symbol.kind,
                    file_path=symbol.file_path,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    score=float(score),
                    ranks=dict(ranks_by_signal.get(symbol_id, {})),
                )
            )
        return RetrievalResult(
            query=query,
            candidates=candidates,
            signals=tuple(signals),
            dense_ranked=dense is not None,
        )

    # ---------------------------------------------------------------- rendering

    def _read_span(self, file_path: str, start_line: int, end_line: int) -> str:
        full = os.path.join(self.indexer.root, file_path)
        try:
            with open(full, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except (OSError, UnicodeDecodeError):
            return ""
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line), start + self.max_span_lines - 1)
        if end < start:
            return ""
        return "".join(lines[start - 1 : end])

    def render_context(
        self,
        query: str,
        token_budget: int = 2048,
        k: int = 8,
        tokenizer=None,
        max_chars_per_span: Optional[int] = None,
    ) -> RetrievalContext:
        """Render the fused spans as source text within ``token_budget``."""
        result = self.search(query, k=k)
        chunks: List[str] = []
        kept: List[RetrievalCandidate] = []
        used = 0
        truncated = False
        for candidate in result.candidates:
            source = self._read_span(
                candidate.file_path, candidate.start_line, candidate.end_line
            )
            if not source.strip():
                continue
            if max_chars_per_span is not None and len(source) > int(max_chars_per_span):
                source = source[: int(max_chars_per_span)].rstrip() + "\n# ... [truncated]"
            header = (
                f"# File: {candidate.file_path} "
                f"(lines {candidate.start_line}-{candidate.end_line}, "
                f"{candidate.kind} {candidate.qualname})"
            )
            block = header + "\n" + source.rstrip() + "\n"
            cost = self._count(block, tokenizer)
            if used + cost > int(token_budget):
                # Skip this span but keep trying smaller ones (no starvation).
                truncated = True
                continue
            chunks.append(block)
            kept.append(candidate)
            used += cost
        return RetrievalContext(
            text="\n".join(chunks).strip(),
            candidates=kept,
            tokens_used=used,
            truncated=truncated,
            dense_ranked=result.dense_ranked,
        )
