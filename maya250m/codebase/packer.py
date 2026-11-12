"""Topological repository packing and structure-aware Fill-in-the-Middle.

Two independent pieces:

* :class:`TopologicalRepoPacker` orders files so callees/imported foundations come
  *before* their callers/importers. With 24 of 32 layers being Gated DeltaNet, this
  ordering lets the recurrent state learn ``symbol -> signature`` bindings before
  the call site needs them. Cycles are handled by the minimum-in-degree rule rather
  than by failing.
* :func:`fim_example` builds 50% PSM / 50% SPM Fill-in-the-Middle examples whose
  middle span is aligned to an AST node, so a token is never cut mid-identifier.
"""

from __future__ import annotations

import ast
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .indexer import RepoGraphIndexer

__all__ = [
    "RepoTokens",
    "PackedRepo",
    "TopologicalRepoPacker",
    "FIM_MARKERS",
    "FIMSpan",
    "ast_span_candidates",
    "fim_example",
]

FILE_SEP = "# <|file_sep|> "
FIM_MARKERS = {
    "prefix": "# <|fim_prefix|>",
    "suffix": "# <|fim_suffix|>",
    "middle": "# <|fim_middle|>",
    "repo_context": "# <|repo_context|>",
}


@dataclass
class RepoTokens:
    """A packed file sequence with its accounting."""

    text: str
    order: List[str] = field(default_factory=list)
    tokens_used: int = 0
    truncated: bool = False


#: Backwards-compatible alias used by the specification's target layout.
PackedRepo = RepoTokens


class TopologicalRepoPacker:
    """Order repository files by minimum in-degree (imports before importers)."""

    def __init__(self, indexer: RepoGraphIndexer, file_sep: str = FILE_SEP):
        self.indexer = indexer
        self.file_sep = file_sep

    def dependency_graph(self, files: Optional[Iterable[str]] = None) -> Dict[str, List[str]]:
        """``file -> [imported files]`` restricted to files present in the repository."""
        selected = (
            set(files) if files is not None else set(self.indexer.files.keys())
        )
        graph: Dict[str, List[str]] = {}
        for path in sorted(selected):
            node = self.indexer.files.get(path)
            deps = []
            if node is not None:
                for imported in node.imports:
                    if imported in selected and imported != path:
                        deps.append(imported)
            graph[path] = sorted(set(deps))
        return graph

    def topological_order(self, files: Optional[Iterable[str]] = None) -> List[str]:
        """Greedy min-in-degree ordering with lexicographic tie-breaking."""
        graph = self.dependency_graph(files)
        remaining = dict(graph)
        order: List[str] = []
        while remaining:
            indegree = {
                path: sum(1 for dep in deps if dep in remaining)
                for path, deps in remaining.items()
            }
            # min in-degree, then lexicographic path (spec 21A.2)
            chosen = min(
                remaining.keys(), key=lambda path: (indegree[path], path)
            )
            order.append(chosen)
            del remaining[chosen]
        return order

    def pack(
        self,
        contents: Optional[Dict[str, str]] = None,
        order: Optional[Sequence[str]] = None,
        token_budget: Optional[int] = None,
        tokenizer=None,
        max_file_chars: Optional[int] = None,
    ) -> RepoTokens:
        """Render the ordered stream with explicit ``# <|file_sep|> path`` headers."""
        order = list(order) if order is not None else self.topological_order()
        contents = contents or {}
        chunks: List[str] = []
        used_paths: List[str] = []
        used_tokens = 0
        truncated = False

        def count(text: str) -> int:
            if tokenizer is not None:
                try:
                    return len(tokenizer.encode(text, add_bos=False, add_eos=False))
                except Exception:  # pragma: no cover
                    pass
            return max(1, (len(text) + 3) // 4)

        for path in order:
            if path in contents:
                body = contents[path]
            else:
                full = os.path.join(self.indexer.root, path)
                try:
                    with open(full, "r", encoding="utf-8") as handle:
                        body = handle.read()
                except OSError:
                    continue
            if max_file_chars is not None and len(body) > int(max_file_chars):
                body = body[: int(max_file_chars)].rstrip() + "\n# ... [truncated]\n"
            block = f"{self.file_sep}{path}\n{body.rstrip()}\n"
            cost = count(block)
            if token_budget is not None and used_tokens + cost > int(token_budget):
                truncated = True
                break
            chunks.append(block)
            used_paths.append(path)
            used_tokens += cost

        return RepoTokens(
            text="\n".join(chunks).strip() + "\n",
            order=used_paths,
            tokens_used=used_tokens,
            truncated=truncated,
        )


@dataclass
class FIMSpan:
    """One AST-aligned span usable as a FIM middle section."""

    kind: str
    start_line: int  # 1-indexed inclusive
    end_line: int  # 1-indexed inclusive


_SPAN_NODE_KINDS = {
    "FunctionDef": "function_body",
    "AsyncFunctionDef": "function_body",
    "For": "loop_body",
    "AsyncFor": "loop_body",
    "While": "loop_body",
    "If": "branch",
    "Assign": "assignment",
    "With": "block",
    "Try": "block",
}


def ast_span_candidates(code: str, min_lines: int = 1, max_lines: int = 120) -> List[FIMSpan]:
    """Enumerate AST-aligned candidate spans (multi-line nodes only)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    spans: List[FIMSpan] = []
    for node in ast.walk(tree):
        kind = _SPAN_NODE_KINDS.get(type(node).__name__)
        if kind is None:
            continue
        start = int(getattr(node, "lineno", 0) or 0)
        end = int(getattr(node, "end_lineno", 0) or 0)
        if start <= 0 or end <= 0:
            continue
        length = end - start + 1
        if length < int(min_lines) or length > int(max_lines):
            continue
        spans.append(FIMSpan(kind=kind, start_line=start, end_line=end))
    spans.sort(key=lambda s: (s.start_line, s.end_line, s.kind))
    return spans


def fim_example(
    code: str,
    rng: Optional[random.Random] = None,
    mode: Optional[str] = None,
    repo_context: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build one structure-aware FIM example, or ``None`` if no span is available.

    ``mode`` is ``"psm"`` (prefix-suffix-middle tokens) or ``"spm"``
    (suffix-prefix-middle); ``None`` samples 50/50.
    """
    spans = ast_span_candidates(code)
    if not spans:
        return None
    rng = rng or random.Random(0)
    span = rng.choice(spans)
    lines = code.splitlines(keepends=True)
    prefix = "".join(lines[: span.start_line - 1])
    middle = "".join(lines[span.start_line - 1 : span.end_line])
    suffix = "".join(lines[span.end_line :])
    if not middle.strip():
        return None

    mode = mode or ("psm" if rng.random() < 0.5 else "spm")
    if mode not in ("psm", "spm"):
        raise ValueError("mode must be 'psm', 'spm' or None")
    context = ""
    if repo_context:
        context = FIM_MARKERS["repo_context"] + "\n" + repo_context.rstrip() + "\n"

    if mode == "psm":
        text = (
            context
            + FIM_MARKERS["prefix"]
            + "\n"
            + prefix
            + FIM_MARKERS["suffix"]
            + "\n"
            + suffix
            + FIM_MARKERS["middle"]
            + "\n"
            + middle
        )
    else:
        text = (
            context
            + FIM_MARKERS["suffix"]
            + "\n"
            + suffix
            + FIM_MARKERS["prefix"]
            + "\n"
            + prefix
            + FIM_MARKERS["middle"]
            + "\n"
            + middle
        )
    return {
        "text": text,
        "mode": mode,
        "span_kind": span.kind,
        "start_line": span.start_line,
        "end_line": span.end_line,
        "prefix": prefix,
        "middle": middle,
        "suffix": suffix,
    }
