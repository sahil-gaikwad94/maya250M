"""Personalized-PageRank repository skeleton map.

Compresses a whole repository into a 1.5k-3k token "repo map": file paths, class
hierarchies and function signatures ranked by Personalized PageRank seeded on the
identifiers in the user's issue. Bodies are elided, so the map states *structure*
and *where things live* without spending the context window on implementation.

The graph has three edge families:

* containment  ``file -> symbol``
* calls        ``symbol -> callee symbol`` (resolved by name, preferring same file)
* imports      ``file -> imported file``

Personalization gives exact query matches ``seed_weight`` (default 10) and every
other node a small uniform mass (1), which is what makes the ranking
*personalized* to the question rather than to raw repository size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .indexer import RepoGraphIndexer, SymbolDef

__all__ = ["RepoMapEntry", "RepoMap", "RepoMapBuilder"]

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_FILE_HINT = re.compile(r"[\w./-]+\.py")
_TRACEBACK_FRAME = re.compile(r'File "([^"]+)", line (\d+), in ([\w.<>]+)')


@dataclass
class RepoMapEntry:
    """One rendered symbol line in the map."""

    file_path: str
    symbol_id: str
    qualname: str
    kind: str
    score: float
    start_line: int
    end_line: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "symbol_id": self.symbol_id,
            "qualname": self.qualname,
            "kind": self.kind,
            "score": round(self.score, 6),
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass
class RepoMap:
    """Rendered repo map plus the ranking it came from."""

    text: str
    entries: List[RepoMapEntry] = field(default_factory=list)
    files_ranked: List[str] = field(default_factory=list)
    symbols_ranked: List[str] = field(default_factory=list)
    token_budget: int = 2048
    tokens_used: int = 0
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entries": [e.to_dict() for e in self.entries],
            "files_ranked": list(self.files_ranked),
            "symbols_ranked": list(self.symbols_ranked),
            "token_budget": int(self.token_budget),
            "tokens_used": int(self.tokens_used),
            "truncated": bool(self.truncated),
        }


class RepoMapBuilder:
    """Build and render a Personalized-PageRank repo skeleton map."""

    def __init__(
        self,
        indexer: RepoGraphIndexer,
        alpha: float = 0.85,
        iterations: int = 15,
        seed_weight: float = 10.0,
        default_weight: float = 1.0,
        max_map_chars_per_symbol: int = 200,
    ):
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.indexer = indexer
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.seed_weight = float(seed_weight)
        self.default_weight = float(default_weight)
        self.max_map_chars_per_symbol = int(max_map_chars_per_symbol)
        self.nodes: List[str] = []
        self.adjacency: Dict[str, List[str]] = {}
        self._build_graph()

    # -------------------------------------------------------------------- graph

    def _file_node(self, path: str) -> str:
        return f"file::{path}"

    def _build_graph(self) -> None:
        adjacency: Dict[str, Set[str]] = {}
        self._symbol_by_id: Dict[str, SymbolDef] = {}

        def add_edge(src: str, dst: str) -> None:
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set())

        for path, node in self.indexer.files.items():
            fnode = self._file_node(path)
            adjacency.setdefault(fnode, set())
            for symbol in node.symbols:
                self._symbol_by_id[symbol.symbol_id] = symbol
                add_edge(fnode, symbol.symbol_id)
            for imported in node.imports:
                if imported in self.indexer.files:
                    add_edge(fnode, self._file_node(imported))

        # Call and inheritance edges between symbols.
        for symbol in self._symbol_by_id.values():
            for callee in symbol.calls:
                target = self._resolve_symbol(callee, prefer_file=symbol.file_path)
                if target is not None:
                    add_edge(symbol.symbol_id, target.symbol_id)
            for base in symbol.bases:
                target = self._resolve_symbol(base.split(".")[-1], prefer_file=symbol.file_path)
                if target is not None:
                    add_edge(symbol.symbol_id, target.symbol_id)

        self.adjacency = {node: sorted(edges) for node, edges in sorted(adjacency.items())}
        self.nodes = sorted(self.adjacency)

    def _resolve_symbol(self, name: str, prefer_file: Optional[str] = None) -> Optional[SymbolDef]:
        basename = name.rsplit(".", 1)[-1]
        candidates = self.indexer.symbols_named(basename)
        if not candidates:
            return None
        if prefer_file is not None:
            same_file = [c for c in candidates if c.file_path == prefer_file]
            if same_file:
                return same_file[0]
        return candidates[0]

    # ------------------------------------------------------------ personalization

    def query_terms(self, query: str) -> Set[str]:
        """Extract identifiers, file hints and traceback symbols from a query."""
        terms: Set[str] = set()
        for match in _IDENT.finditer(query or ""):
            terms.add(match.group(0))
        for match in _FILE_HINT.finditer(query or ""):
            path = match.group(0).lstrip("./")
            terms.add(path)
            terms.add(path.replace("/", ".").removesuffix(".py"))
            terms.add(path.rsplit("/", 1)[-1].removesuffix(".py"))
        for match in _TRACEBACK_FRAME.finditer(query or ""):
            terms.add(match.group(1))
            terms.add(match.group(3).rsplit(".", 1)[-1])
        return {t for t in terms if t}

    def personalize(self, query: str) -> Dict[str, float]:
        """Build the PPR personalization vector over graph nodes."""
        terms = self.query_terms(query)
        lowered_terms = {t.lower() for t in terms}
        weights: Dict[str, float] = {}
        for node in self.nodes:
            weight = self.default_weight
            if node.startswith("file::"):
                path = node[len("file::") :]
                stem = path.rsplit("/", 1)[-1].removesuffix(".py")
                if path in terms or stem in terms:
                    weight = self.seed_weight
                elif any(t in path.lower() for t in lowered_terms if len(t) >= 4):
                    weight = self.seed_weight / 2.0
            else:
                symbol = self._symbol_by_id.get(node)
                if symbol is not None and (
                    symbol.name in terms or symbol.qualname in terms
                ):
                    weight = self.seed_weight
                elif symbol is not None and symbol.name.lower() in lowered_terms:
                    weight = self.seed_weight / 2.0
            weights[node] = weight
        total = sum(weights.values()) or 1.0
        return {node: w / total for node, w in weights.items()}

    # ---------------------------------------------------------------- pagerank

    def pagerank(self, personalization: Dict[str, float]) -> Dict[str, float]:
        """Power-method PPR (15 iterations, alpha=0.85 by default)."""
        ranks = {node: personalization.get(node, 0.0) for node in self.nodes}
        out_degree = {n: len(self.adjacency.get(n, ())) for n in self.nodes}
        for _ in range(self.iterations):
            incoming = {node: 0.0 for node in self.nodes}
            dangling_mass = 0.0
            for node in self.nodes:
                rank = ranks[node]
                degree = out_degree[node]
                if degree == 0:
                    dangling_mass += rank
                    continue
                share = rank / degree
                for target in self.adjacency[node]:
                    incoming[target] += share
            ranks = {
                node: self.alpha * (incoming[node] + dangling_mass * personalization.get(node, 0.0))
                + (1.0 - self.alpha) * personalization.get(node, 0.0)
                for node in self.nodes
            }
        return ranks

    # ---------------------------------------------------------------- rendering

    def _count_tokens(self, text: str, tokenizer=None) -> int:
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text, add_bos=False, add_eos=False))
            except Exception:  # pragma: no cover
                pass
        return max(1, (len(text) + 3) // 4)

    def build(
        self,
        query: str,
        token_budget: int = 2048,
        tokenizer=None,
        max_files: Optional[int] = None,
        max_symbols_per_file: int = 24,
    ) -> RepoMap:
        """Render the highest-scoring structure until ``token_budget`` is reached."""
        personalization = self.personalize(query)
        ranks = self.pagerank(personalization)

        file_scores: Dict[str, float] = {}
        symbol_entries: List[Tuple[float, SymbolDef]] = []
        for node, score in ranks.items():
            if node.startswith("file::"):
                file_scores[node[len("file::") :]] = score
            else:
                symbol = self._symbol_by_id.get(node)
                if symbol is not None:
                    symbol_entries.append((score, symbol))
                    file_scores[symbol.file_path] = max(
                        file_scores.get(symbol.file_path, 0.0), score * 0.5
                    )

        files_ranked = [
            path
            for path, _score in sorted(
                file_scores.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ]
        if max_files is not None:
            files_ranked = files_ranked[: int(max_files)]
        allowed = set(files_ranked)

        symbol_entries = [
            (score, symbol) for score, symbol in symbol_entries if symbol.file_path in allowed
        ]
        symbol_entries.sort(key=lambda pair: (-pair[0], pair[1].symbol_id))

        used_tokens = 0
        rendered: List[str] = []
        entries: List[RepoMapEntry] = []
        truncated = False
        current_file: Optional[str] = None
        per_file_count: Dict[str, int] = {}

        for score, symbol in symbol_entries:
            line = symbol.signature
            if symbol.kind in ("function", "async_function", "method", "async_method"):
                line = symbol.signature.rstrip() + " ..."
            if len(line) > self.max_map_chars_per_symbol:
                line = line[: self.max_map_chars_per_symbol - 3] + "..."
            indent = "    " if symbol.qualname.count(".") > 0 else ""
            body = f"{indent}{line}"

            block = []
            if symbol.file_path != current_file:
                block.append("")
                block.append(
                    f"# File: {symbol.file_path} "
                    f"(lines {self._file_line_span(symbol.file_path)})"
                )
            block.append(body)
            candidate_text = "\n".join(block)
            cost = self._count_tokens(candidate_text, tokenizer)

            if used_tokens + cost > int(token_budget):
                truncated = True
                break
            if per_file_count.get(symbol.file_path, 0) >= int(max_symbols_per_file):
                continue
            rendered.extend(block)
            used_tokens += cost
            current_file = symbol.file_path
            per_file_count[symbol.file_path] = per_file_count.get(symbol.file_path, 0) + 1
            entries.append(
                RepoMapEntry(
                    file_path=symbol.file_path,
                    symbol_id=symbol.symbol_id,
                    qualname=symbol.qualname,
                    kind=symbol.kind,
                    score=score,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                )
            )

        text = "\n".join(rendered).strip()
        return RepoMap(
            text=text,
            entries=entries,
            files_ranked=files_ranked,
            symbols_ranked=[e.symbol_id for e in entries],
            token_budget=int(token_budget),
            tokens_used=used_tokens if used_tokens else self._count_tokens(text, tokenizer),
            truncated=truncated,
        )

    def _file_line_span(self, path: str) -> str:
        node = self.indexer.files.get(path)
        if node is None:
            return "?"
        return f"1-{node.line_count}"

    def localize(self, query: str, k: int = 5) -> List[str]:
        """Ranked candidate files for an issue (used by localization evaluation)."""
        return self.build(query, token_budget=2048, max_files=k).files_ranked[:k]
