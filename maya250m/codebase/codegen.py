"""Content-anchored ``SEARCH/REPLACE`` patch engine.

Small models fail up to 40% of unified diffs purely from line-number arithmetic.
Content-anchored edits remove that failure mode: the model quotes the exact code it
wants to change, and the engine refuses ambiguous anchors instead of guessing.

Invariants enforced here:

1. exact substring match first, then a whitespace-insensitive line match that
   preserves relative indentation;
2. a fail-closed uniqueness guard (0 matches -> ``patch_not_found``, >=2 ->
   ``patch_ambiguous``);
3. atomic verification before commit: ``ast.parse`` + Vexx AST policy +
   cross-file arity impact, all checked on an in-memory buffer;
4. the file is only written when every check passes.
"""

from __future__ import annotations

import ast
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..verification import VerificationResult, failure, ok
from .indexer import RepoGraphIndexer
from .multifile import verify_cross_file_impact

__all__ = [
    "PatchBlock",
    "PatchError",
    "parse_patch_blocks",
    "apply_search_replace",
    "SearchReplacePatchEngine",
]

EDIT_FILE_MARKER = "# <|edit_file|>"

_HUNK_RE = re.compile(
    r"<<<<<<< SEARCH\r?\n(?P<search>.*?)\r?\n?=======\r?\n(?P<replace>.*?)\r?\n?>>>>>>> REPLACE",
    re.DOTALL,
)
_FILE_MARKER_RE = re.compile(rf"^[ \t]*{re.escape(EDIT_FILE_MARKER)}[ \t]*(?P<path>[^\n]+)$", re.MULTILINE)


@dataclass
class PatchBlock:
    """One parsed ``SEARCH/REPLACE`` hunk with its target file."""

    file_path: str
    search: str
    replace: str
    start_offset: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "search": self.search,
            "replace": self.replace,
        }


class PatchError(Exception):
    """Typed patch failure carrying a canonical failure category."""

    def __init__(self, category: str, message: str, **details: Any):
        super().__init__(message)
        self.category = category
        self.message = message
        self.details = details


def parse_patch_blocks(text: str, default_file: str = "") -> List[PatchBlock]:
    """Parse ``# <|edit_file|> path`` + ``SEARCH/REPLACE`` hunks from model output."""
    if not isinstance(text, str):
        raise TypeError("patch text must be a string")
    markers = [
        (m.start(), m.group("path").strip())
        for m in _FILE_MARKER_RE.finditer(text)
    ]
    blocks: List[PatchBlock] = []
    for match in _HUNK_RE.finditer(text):
        offset = match.start()
        target = default_file
        for marker_pos, marker_path in markers:
            if marker_pos < offset:
                target = marker_path
            else:
                break
        blocks.append(
            PatchBlock(
                file_path=target,
                search=match.group("search"),
                replace=match.group("replace"),
                start_offset=offset,
            )
        )
    if not blocks and "<<<<<<< SEARCH" in text:
        raise PatchError("patch_error", "malformed SEARCH/REPLACE block")
    return blocks


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _line_window_match(
    text_lines: Sequence[str], search_lines: Sequence[str]
) -> List[int]:
    """Return start indices of windows that match ``search_lines`` ignoring indentation."""
    if not search_lines:
        return []
    needle = [line.strip() for line in search_lines]
    starts: List[int] = []
    for start in range(0, len(text_lines) - len(needle) + 1):
        window = [line.strip() for line in text_lines[start : start + len(needle)]]
        if window == needle:
            starts.append(start)
    return starts


def apply_search_replace(text: str, search: str, replace: str) -> str:
    """Apply one content-anchored edit, or raise :class:`PatchError`."""
    if not search:
        raise PatchError("patch_error", "empty SEARCH block")
    occurrences = text.count(search)
    if occurrences == 1:
        return text.replace(search, replace, 1)
    if occurrences > 1:
        raise PatchError(
            "patch_ambiguous",
            f"Ambiguous SEARCH block matched {occurrences} locations; "
            "include the enclosing function header",
            matches=occurrences,
        )

    # Fallback: indent-insensitive line match that preserves the file's indentation.
    text_lines = text.splitlines(keepends=True)
    search_lines = search.splitlines()
    starts = _line_window_match(text_lines, search_lines)
    if not starts:
        raise PatchError(
            "patch_not_found",
            "SEARCH block not found in file; include 2-3 surrounding context lines",
        )
    if len(starts) > 1:
        raise PatchError(
            "patch_ambiguous",
            f"Ambiguous SEARCH block matched {len(starts)} locations; "
            "include enclosing function header",
            matches=len(starts),
        )

    start = starts[0]
    base_indent = _indent_of(text_lines[start])
    search_indent = _indent_of(search_lines[0]) if search_lines else 0
    delta = base_indent - search_indent
    reindented: List[str] = []
    for line in replace.splitlines(keepends=True):
        if not line.strip():
            reindented.append(line)
            continue
        target_indent = max(0, _indent_of(line) + delta)
        reindented.append(" " * target_indent + line.lstrip(" \t"))
    new_lines = text_lines[:start] + reindented + text_lines[start + len(search_lines) :]
    return "".join(new_lines)


def _signature_table_from_source(source: str) -> Dict[str, Dict[str, Any]]:
    """Per-function contracts from a source string (for old/new comparison)."""
    table: Dict[str, Dict[str, Any]] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return table
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        positional = list(args.posonlyargs) + list(args.args)
        if positional and positional[0].arg in ("self", "cls"):
            positional = positional[1:]
        n_pos = len(positional)
        n_def = len(args.defaults)
        has_vararg = args.vararg is not None or args.kwarg is not None
        kw_required = sum(
            1 for arg, default in zip(args.kwonlyargs, args.kw_defaults) if default is None
        )
        table[node.name] = {
            "min_arity": max(0, n_pos - n_def),
            "max_arity": None if has_vararg else n_pos + kw_required,
            "return_type": ast.unparse(node.returns) if node.returns is not None else "any",
        }
    return table


@dataclass
class SearchReplacePatchEngine:
    """Apply and atomically verify content-anchored patches."""

    indexer: Optional[RepoGraphIndexer] = None
    vexx: Any = None
    root: Optional[str] = None
    check_cross_file: bool = True
    last_new_text: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.root is None and self.indexer is not None:
            self.root = self.indexer.root
        if self.root is None:
            self.root = os.getcwd()

    def _resolve(self, file_path: str) -> str:
        if os.path.isabs(file_path):
            return file_path
        return os.path.join(self.root, file_path)

    def read(self, file_path: str) -> str:
        with open(self._resolve(file_path), "r", encoding="utf-8") as handle:
            return handle.read()

    def preview(self, file_path: str, search: str, replace: str) -> str:
        """Apply in memory only; raises :class:`PatchError` on anchor failure."""
        return apply_search_replace(self.read(file_path), search, replace)

    def apply_and_verify(
        self,
        file_path: str,
        search: str,
        replace: str,
        changed_files: Sequence[str] = (),
        commit: bool = False,
        extra_changed_signatures: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    ) -> VerificationResult:
        """Verify the patch on a buffer, then optionally write it atomically."""
        rel_path = file_path.replace("\\", "/")
        try:
            original = self.read(file_path)
        except OSError as exc:
            return failure("patch_error", f"cannot read {file_path}: {exc}")

        try:
            new_text = apply_search_replace(original, search, replace)
        except PatchError as exc:
            return failure(exc.category, exc.message, **exc.details)

        try:
            ast.parse(new_text)
        except SyntaxError as exc:
            return failure(
                "syntax_error",
                f"patched buffer does not parse: {exc.msg} (line {exc.lineno})",
                line=exc.lineno,
                offset=exc.offset,
            )

        if self.vexx is not None:
            try:
                self.vexx._check_ast_policy(new_text)
            except SyntaxError as exc:
                return failure("syntax_error", f"policy parse failed: {exc}")
            except PermissionError as exc:
                return failure("ast_policy_violation", str(exc))

        if self.check_cross_file and self.indexer is not None:
            old_table = _signature_table_from_source(original)
            new_table = _signature_table_from_source(new_text)
            changed: Dict[str, Optional[Dict[str, Any]]] = {}
            for name, contract in old_table.items():
                if name not in new_table:
                    changed[name] = None
                elif new_table[name] != contract:
                    changed[name] = new_table[name]
            if extra_changed_signatures:
                changed.update(extra_changed_signatures)
            if changed:
                impacted = verify_cross_file_impact(
                    self.indexer,
                    changed,
                    changed_files=list(changed_files) + [rel_path],
                )
                if not impacted.passed:
                    return impacted

        self.last_new_text = new_text
        self.history.append(
            {
                "file_path": rel_path,
                "search_lines": search.count("\n") + 1,
                "replace_lines": replace.count("\n") + 1,
                "committed": bool(commit),
            }
        )

        if commit:
            directory = os.path.dirname(os.path.abspath(self._resolve(file_path))) or "."
            fd, tmp_path = tempfile.mkstemp(prefix=".maya_patch_", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(new_text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, self._resolve(file_path))
            except Exception as exc:  # pragma: no cover - filesystem failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                return failure("internal_error", f"atomic write failed: {exc}")
            if self.indexer is not None:
                try:
                    self.indexer.index_file(rel_path)
                except Exception:  # pragma: no cover - index refresh is best-effort
                    pass

        return ok(
            "patch verified",
            file_path=rel_path,
            committed=bool(commit),
            changed_lines=replace.count("\n") + 1,
        )

    def apply_patch_text(
        self,
        patch_text: str,
        commit: bool = False,
        default_file: str = "",
    ) -> List[VerificationResult]:
        """Parse and apply a multi-hunk, multi-file patch, stopping at the first failure."""
        try:
            blocks = parse_patch_blocks(patch_text, default_file=default_file)
        except PatchError as exc:
            return [failure(exc.category, exc.message)]
        if not blocks:
            return [failure("patch_error", "no SEARCH/REPLACE blocks found")]

        changed = sorted({b.file_path for b in blocks if b.file_path})
        results: List[VerificationResult] = []
        for block in blocks:
            if not block.file_path:
                results.append(
                    failure(
                        "invalid_tool_call",
                        "patch block has no target file; emit '# <|edit_file|> path'",
                    )
                )
                return results
            result = self.apply_and_verify(
                block.file_path,
                block.search,
                block.replace,
                changed_files=changed,
                commit=commit,
            )
            results.append(result)
            if not result.passed:
                return results
        return results
