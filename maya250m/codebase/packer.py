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
    "TaskContextPacker",
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
    skipped: List[str] = field(default_factory=list)


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
        """Render the ordered stream with explicit ``# <|file_sep|> path`` headers.

        A file that does not fit is *skipped*, never used to abort the walk: letting
        one enormous file end packing would starve every smaller foundation behind
        it, which is exactly the wrong trade for a callee-before-caller stream. After
        the whole-file pass, leftover budget is filled with trimmed bodies of the
        skipped files, so the window is still used and the tail is visibly marked.
        """
        order = list(order) if order is not None else self.topological_order()
        contents = contents or {}
        chunks: List[str] = []
        used_paths: List[str] = []
        used_tokens = 0
        truncated = False
        oversized: List[Tuple[str, str]] = []

        def count(text: str) -> int:
            if tokenizer is not None:
                try:
                    return len(tokenizer.encode(text, add_bos=False, add_eos=False))
                except Exception:  # pragma: no cover
                    pass
            return max(1, (len(text) + 3) // 4)

        for path in order:
            body = self._read_body(path, contents)
            if body is None:
                continue
            if max_file_chars is not None and len(body) > int(max_file_chars):
                body = body[: int(max_file_chars)].rstrip() + "\n# ... [truncated]\n"
                truncated = True
            block = self._render_block(path, body)
            cost = count(block)
            if token_budget is not None and used_tokens + cost > int(token_budget):
                truncated = True
                oversized.append((path, body))
                continue
            chunks.append(block)
            used_paths.append(path)
            used_tokens += cost

        if oversized and token_budget is not None:
            for path, body in oversized:
                room = int(token_budget) - used_tokens
                if room <= 0:
                    break
                fitted = self._trim_block(path, body, room, count)
                if fitted is None:
                    continue
                block, cost = fitted
                chunks.append(block)
                used_paths.append(path)
                used_tokens += cost

        packed = set(used_paths)
        skipped = [path for path, _body in oversized if path not in packed]
        return RepoTokens(
            text="\n".join(chunks).strip() + "\n",
            order=used_paths,
            tokens_used=used_tokens,
            truncated=truncated,
            skipped=skipped,
        )

    # ------------------------------------------------------------------ helpers

    def _read_body(self, path: str, contents: Dict[str, str]) -> Optional[str]:
        """Source for ``path``: supplied content first, then the repository."""
        if path in contents:
            return contents[path]
        full = os.path.join(self.indexer.root, path)
        try:
            with open(full, "r", encoding="utf-8") as handle:
                return handle.read()
        except (OSError, UnicodeDecodeError):
            return None

    def _render_block(self, path: str, body: str) -> str:
        return f"{self.file_sep}{path}\n{body.rstrip()}\n"

    def _trim_block(
        self,
        path: str,
        body: str,
        room: int,
        count,
        min_body_chars: int = 80,
    ) -> Optional[Tuple[str, int]]:
        """Largest line-aligned prefix of ``body`` that fits in ``room`` tokens.

        Returns ``None`` when not even a useful stub fits, so a header is never
        emitted claiming a file whose content was entirely dropped.
        """
        header = f"{self.file_sep}{path}\n"
        marker = "\n# ... [truncated]\n"
        if count(header + marker) > room:
            return None

        best_keep = 0
        low, high = 0, len(body)
        while low <= high:
            mid = (low + high) // 2
            candidate = header + body[:mid].rstrip() + marker
            if count(candidate) <= room:
                best_keep = mid
                low = mid + 1
            else:
                high = mid - 1
        if best_keep <= 0:
            return None

        # Back off to the last complete line so a token is not cut mid-identifier.
        boundary = body.rfind("\n", 0, best_keep)
        if boundary > 0:
            best_keep = boundary
        kept = body[:best_keep].rstrip()
        if len(kept) < int(min_body_chars):
            return None
        block = header + kept + marker
        cost = count(block)
        if cost > room:  # pragma: no cover - binary search already guaranteed this
            return None
        return block, cost


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


class TaskContextPacker:
    """Role-budgeted repository context packer with 1-hop dependency expansion.

    Allocates ``token_budget`` across four evidence roles rather than greedily
    filling the window with the first N files:
    1. ``implementation`` (target files & symbols);
    2. ``contracts_and_callers`` (1-hop callers, callees, and imported dependencies);
    3. ``tests`` (test files importing or calling the target symbols);
    4. ``repomap`` (compact symbol signatures for global navigation).
    """

    def __init__(self, indexer: RepoGraphIndexer):
        self.indexer = indexer
        self.topo_packer = TopologicalRepoPacker(indexer)

    def expand_neighborhood(
        self, seed_files: Sequence[str] = (), seed_symbols: Sequence[str] = ()
    ) -> Dict[str, List[str]]:
        """ Partition relevant files into implementation, 1-hop neighbors, and tests."""
        from .transaction import is_test_path

        impl_files: List[str] = []
        neighbor_files: List[str] = []
        test_files: List[str] = []

        for path in seed_files:
            if path in self.indexer.files and path not in impl_files:
                if is_test_path(path):
                    test_files.append(path)
                else:
                    impl_files.append(path)

        for sym in seed_symbols:
            for defn in (self.indexer.definitions(sym) or self.indexer.symbols_named(sym)):
                if defn.file_path not in impl_files and not is_test_path(defn.file_path):
                    impl_files.append(defn.file_path)
                for caller in self.indexer.callers(defn.name):
                    c_path = caller.file_path
                    if is_test_path(c_path):
                        if c_path not in test_files:
                            test_files.append(c_path)
                    elif c_path not in impl_files and c_path not in neighbor_files:
                        neighbor_files.append(c_path)

        for path in list(impl_files):
            node = self.indexer.files.get(path)
            if node is None:
                continue
            for imp in node.imports:
                if imp in self.indexer.files and imp not in impl_files and imp not in neighbor_files:
                    if is_test_path(imp):
                        if imp not in test_files:
                            test_files.append(imp)
                    else:
                        neighbor_files.append(imp)

        # Discover any test files importing the implementation files
        for path, node in sorted(self.indexer.files.items()):
            if is_test_path(path) and path not in test_files:
                if any(imp in impl_files or imp in neighbor_files for imp in node.imports):
                    test_files.append(path)

        return {
            "implementation": impl_files,
            "contracts_and_callers": neighbor_files,
            "tests": test_files,
        }

    def pack_for_task(
        self,
        seed_files: Sequence[str] = (),
        seed_symbols: Sequence[str] = (),
        token_budget: int = 2048,
        tokenizer=None,
    ) -> Dict[str, Any]:
        """Pack role-balanced context for an engineering task."""
        roles = self.expand_neighborhood(seed_files=seed_files, seed_symbols=seed_symbols)
        ordered_candidates: List[str] = []
        for group in (roles["implementation"], roles["contracts_and_callers"], roles["tests"]):
            for p in self.topo_packer.topological_order(group) if group else []:
                if p not in ordered_candidates:
                    ordered_candidates.append(p)
        if not ordered_candidates:
            ordered_candidates = self.topo_packer.topological_order()

        packed = self.topo_packer.pack(
            order=ordered_candidates, token_budget=token_budget, tokenizer=tokenizer
        )
        return {
            "text": packed.text,
            "order": packed.order,
            "tokens_used": packed.tokens_used,
            "truncated": packed.truncated,
            "skipped": packed.skipped,
            "roles": roles,
        }
