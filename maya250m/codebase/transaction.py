"""Atomic multi-file patch transactions with tamper detection and rollback.

Per-hunk verification (:class:`SearchReplacePatchEngine`) is not enough for a
multi-file change: a model can edit file A successfully, then break file B, leaving
the repository half-migrated. This module makes a set of edits *one decision*:

* every edit is applied to an in-memory overlay first (nothing touches disk);
* all overlay files are parsed and policy-checked;
* cross-file contract checks run against a temporary copy of the repository with
  the overlay applied, so caller edits inside the same transaction are visible;
* only when everything passes are the files written, each atomically (temp file +
  ``os.replace``);
* if a write fails midway, the already-written files are restored;
* the repository tree hash is captured before and after, and
  :meth:`PatchTransaction.rollback` restores the exact pre-transaction bytes, but
  only if no unrelated file changed in the meantime.

Test tampering: edits that target test files are refused unless the caller opts in,
and even then an edit that *reduces* the number of assertions/skips in a test file
is refused (weakening a test is the same failure as editing it).
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..verification import VerificationResult, failure, ok
from .codegen import PatchBlock, PatchError, apply_search_replace, parse_patch_blocks
from .indexer import DEFAULT_EXCLUDES, RepoGraphIndexer
from .multifile import verify_cross_file_impact

__all__ = [
    "FileEdit",
    "PatchTransaction",
    "is_test_path",
    "count_assertions",
    "tree_hash",
    "TEST_TAMPERING_CATEGORY",
]

TEST_TAMPERING_CATEGORY = "test_tampering_blocked"

_TEST_DIR_NAMES = {"tests", "test", "__tests__"}
_TEST_FILE_RE = re.compile(r"^(?:test_.*\.py|.*_test\.py|conftest\.py|tests?\.py)$")
_ASSERT_RE = re.compile(
    r"^[ \t]*(?:assert\b|self\.assert\w+|pytest\.raises|pytest\.fail|unittest\.skip)",
    re.MULTILINE,
)

_SKIP_DIRS = set(DEFAULT_EXCLUDES) | {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "site-packages",
}
_SKIP_SUFFIXES = (".pyc", ".pyo", ".so", ".dylib", ".pyd")


def is_test_path(rel_path: str) -> bool:
    """Whether ``rel_path`` is a test file that must not be silently edited."""
    normalized = rel_path.replace("\\", "/").lstrip("./")
    parts = normalized.split("/")
    if any(part in _TEST_DIR_NAMES for part in parts[:-1]):
        return True
    return bool(_TEST_FILE_RE.match(parts[-1]))


def count_assertions(text: str) -> int:
    """Assertions and skip/fail markers in a test source file."""
    return len(_ASSERT_RE.findall(text))


def tree_hash(root: str, skip_dirs: Optional[Iterable[str]] = None) -> str:
    """Content-addressed hash of every tracked file under ``root`` (order-stable)."""
    skip = set(skip_dirs) if skip_dirs is not None else _SKIP_DIRS
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in skip)
        for name in sorted(filenames):
            if name.endswith(_SKIP_SUFFIXES):
                continue
            absolute = os.path.join(dirpath, name)
            if not os.path.isfile(absolute) or os.path.islink(absolute):
                continue
            rel = os.path.relpath(absolute, root).replace(os.sep, "/")
            file_digest = hashlib.sha256()
            try:
                with open(absolute, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        file_digest.update(chunk)
            except OSError:  # pragma: no cover - unreadable file
                file_digest.update(b"<unreadable>")
            digest.update(f"{rel}\x00{file_digest.hexdigest()}\n".encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FileEdit:
    """One content-anchored edit, validated against its file at plan time."""

    file_path: str
    search: str
    replace: str

    def __post_init__(self) -> None:
        if not self.file_path:
            raise PatchError("invalid_tool_call", "patch edit has no target file")
        if not self.search:
            raise PatchError("patch_error", "empty SEARCH block")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "search_lines": self.search.count("\n") + 1,
            "replace_lines": self.replace.count("\n") + 1,
        }


@dataclass
class PatchTransaction:
    """All-or-nothing multi-file patch, verified on an in-memory overlay."""

    root: str
    edits: List[FileEdit] = field(default_factory=list)
    indexer: Optional[RepoGraphIndexer] = None
    vexx: Any = None
    allow_test_edits: bool = False
    allow_assert_removal: bool = False
    max_files: int = 64
    overlay_copy_limit: int = 4000

    _originals: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _overlay: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _snapshot_hash: Optional[str] = None
    _written_hash: Optional[str] = None
    _writer: Any = field(default=None, init=False, repr=False)
    last_details: Dict[str, Any] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ setup

    def add(self, file_path: str, search: str, replace: str) -> "PatchTransaction":
        self.edits.append(FileEdit(file_path=file_path, search=search, replace=replace))
        return self

    @classmethod
    def from_blocks(
        cls, blocks: Sequence[PatchBlock], root: str, **kwargs: Any
    ) -> "PatchTransaction":
        transaction = cls(root=root, **kwargs)
        for block in blocks:
            transaction.add(block.file_path, block.search, block.replace)
        return transaction

    @classmethod
    def from_patch_text(
        cls, text: str, root: str, default_file: str = "", **kwargs: Any
    ) -> "PatchTransaction":
        return cls.from_blocks(
            parse_patch_blocks(text, default_file=default_file), root=root, **kwargs
        )

    # ---------------------------------------------------------------- helpers

    def _resolve(self, rel_path: str) -> str:
        return rel_path if os.path.isabs(rel_path) else os.path.join(self.root, rel_path)

    def _relative(self, rel_path: str) -> str:
        if os.path.isabs(rel_path):
            return os.path.relpath(rel_path, self.root).replace(os.sep, "/")
        return rel_path.replace("\\", "/").lstrip("./")

    def _read(self, rel_path: str) -> str:
        with open(self._resolve(rel_path), "r", encoding="utf-8") as handle:
            return handle.read()

    def _write_atomic(self, rel_path: str, text: str) -> None:
        if self._writer is not None:
            self._writer(rel_path, text)
            return
        target = self._resolve(rel_path)
        directory = os.path.dirname(os.path.abspath(target)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".maya_txn_", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # --------------------------------------------------------------- planning

    def plan(self) -> Dict[str, str]:
        """Apply every edit to an in-memory overlay; raise :class:`PatchError`."""
        if not self.edits:
            raise PatchError("patch_error", "transaction contains no edits")
        by_file: Dict[str, List[FileEdit]] = {}
        for edit in self.edits:
            by_file.setdefault(self._relative(edit.file_path), []).append(edit)
        if len(by_file) > int(self.max_files):
            raise PatchError(
                "patch_error",
                f"transaction touches {len(by_file)} files (limit {self.max_files})",
            )

        originals: Dict[str, str] = {}
        overlay: Dict[str, str] = {}
        for rel_path, file_edits in by_file.items():
            try:
                original = self._read(rel_path)
            except OSError as exc:
                raise PatchError("patch_error", f"cannot read {rel_path}: {exc}") from exc
            text = original
            for edit in file_edits:
                try:
                    text = apply_search_replace(text, edit.search, edit.replace)
                except PatchError as exc:
                    raise PatchError(
                        exc.category,
                        f"{rel_path}: {exc.message}",
                        file_path=rel_path,
                        **exc.details,
                    ) from exc
            originals[rel_path] = original
            overlay[rel_path] = text

        if not self.allow_test_edits:
            touched = sorted(path for path in overlay if is_test_path(path))
            if touched:
                raise PatchError(
                    TEST_TAMPERING_CATEGORY,
                    f"refusing to edit test file(s) without explicit permission: {touched}",
                    files=touched,
                )

        # Weakening a test is test tampering even when edits are allowed.
        for rel_path, text in overlay.items():
            if not is_test_path(rel_path):
                continue
            before = count_assertions(originals[rel_path])
            after = count_assertions(text)
            if after < before and not self.allow_assert_removal:
                raise PatchError(
                    TEST_TAMPERING_CATEGORY,
                    (
                        f"{rel_path}: edit removes test strength "
                        f"({before} -> {after} assertions/markers)"
                    ),
                    file_path=rel_path,
                    assertions_before=before,
                    assertions_after=after,
                )

        for rel_path, text in overlay.items():
            try:
                ast.parse(text)
            except SyntaxError as exc:
                raise PatchError(
                    "syntax_error",
                    f"{rel_path}: patched buffer does not parse: {exc.msg} (line {exc.lineno})",
                    file_path=rel_path,
                    line=exc.lineno,
                    offset=exc.offset,
                ) from exc

        self._originals = originals
        self._overlay = overlay
        return dict(overlay)

    # ------------------------------------------------------------- verification

    def _changed_contracts(self) -> Dict[str, Optional[Dict[str, Any]]]:
        from .codegen import _signature_table_from_source

        changed: Dict[str, Optional[Dict[str, Any]]] = {}
        for rel_path, new_text in self._overlay.items():
            old_table = _signature_table_from_source(self._originals[rel_path])
            new_table = _signature_table_from_source(new_text)
            for name, contract in old_table.items():
                if name not in new_table or new_table[name] != contract:
                    changed[name] = new_table.get(name)
        return changed

    def _verify_cross_file(
        self, changed: Dict[str, Optional[Dict[str, Any]]]
    ) -> Optional[VerificationResult]:
        if not changed:
            return None
        if self.indexer is not None and self.indexer.root != self.root:
            # The index belongs to another tree; do not mix contracts across roots.
            self.indexer = None

        working_indexer = self.indexer
        overlay_copy = False
        temp_dir: Optional[str] = None
        if any(path.endswith(".py") for path in self._overlay):
            temp_dir = self._materialise_overlay()
            if temp_dir is not None:
                working_indexer = RepoGraphIndexer(temp_dir)
                working_indexer.index()
                overlay_copy = True

        if working_indexer is None:
            return None

        try:
            result = verify_cross_file_impact(
                working_indexer,
                changed,
                changed_files=list(self._overlay),
            )
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

        result.details["verified_against_overlay"] = overlay_copy
        return result

    def _materialise_overlay(self) -> Optional[str]:
        """Copy the repository and apply the overlay, or ``None`` if too large."""
        try:
            temp_dir = tempfile.mkdtemp(prefix="maya_txn_overlay_")
        except OSError:  # pragma: no cover - temp dir failure
            return None
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIRS)
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                absolute = os.path.join(dirpath, name)
                if not os.path.isfile(absolute) or os.path.islink(absolute):
                    continue
                count += 1
                if count > int(self.overlay_copy_limit):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return None
                rel = os.path.relpath(absolute, self.root)
                target = os.path.join(temp_dir, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                try:
                    shutil.copyfile(absolute, target)
                except OSError:  # pragma: no cover
                    continue
        for rel_path, text in self._overlay.items():
            if not rel_path.endswith(".py"):
                continue
            target = os.path.join(temp_dir, rel_path)
            os.makedirs(os.path.dirname(target) or temp_dir, exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(text)
        return temp_dir

    def verify(self) -> VerificationResult:
        """Full verification on the overlay; writes nothing."""
        try:
            self.plan()
        except PatchError as exc:
            return failure(exc.category, exc.message, **exc.details)

        if self.vexx is not None:
            for rel_path, text in self._overlay.items():
                try:
                    self.vexx._check_ast_policy(text)
                except SyntaxError as exc:
                    return failure(
                        "syntax_error", f"{rel_path}: policy parse failed: {exc}"
                    )
                except PermissionError as exc:
                    return failure("ast_policy_violation", f"{rel_path}: {exc}")

        changed = self._changed_contracts()
        if not changed:
            return ok(
                "transaction verified",
                files=sorted(self._overlay),
                changed_contracts=[],
            )
        checked = self._verify_cross_file(changed)
        if checked is not None and not checked.passed:
            return checked
        return ok(
            "transaction verified",
            files=sorted(self._overlay),
            changed_contracts=sorted(changed),
            overlay_verified=bool(checked and checked.details.get("verified_against_overlay")),
        )

    # --------------------------------------------------------------- execution

    def execute(self, commit: bool = True) -> VerificationResult:
        """Verify, then (when ``commit``) write every file atomically."""
        verified = self.verify()
        if not verified.passed:
            self.last_details = {"committed": False, "rolled_back": False}
            verified.details.update(self.last_details)
            return verified

        before = tree_hash(self.root)
        if not commit:
            self.last_details = {
                "committed": False,
                "rolled_back": False,
                "tree_hash_before": before,
                "tree_hash_after": before,
                "files": sorted(self._overlay),
            }
            self.last_details.update(verified.details)
            return ok("transaction verified (dry run)", **self.last_details)

        written: List[str] = []
        try:
            for rel_path, text in self._overlay.items():
                self._write_atomic(rel_path, text)
                written.append(rel_path)
        except Exception as exc:
            restored = self._restore(written)
            after = tree_hash(self.root)
            self.last_details = {
                "committed": False,
                "rolled_back": True,
                "restored": restored,
                "tree_hash_before": before,
                "tree_hash_after": after,
            }
            return failure(
                "internal_error",
                f"write failed after {len(written)} file(s): {exc}",
                **self.last_details,
            )

        after = tree_hash(self.root)
        self._snapshot_hash = before
        self._written_hash = after
        self.last_details = {
            "committed": True,
            "rolled_back": False,
            "files": sorted(written),
            "tree_hash_before": before,
            "tree_hash_after": after,
        }
        if self.indexer is not None:
            for rel_path in written:
                try:
                    self.indexer.index_file(rel_path)
                except Exception:  # pragma: no cover - index refresh is best-effort
                    pass
        return ok("transaction committed", **self.last_details)

    def _restore(self, rel_paths: Sequence[str]) -> List[str]:
        restored: List[str] = []
        for rel_path in rel_paths:
            original = self._originals.get(rel_path)
            if original is None:
                continue
            try:
                self._write_atomic(rel_path, original)
                restored.append(rel_path)
            except Exception:  # pragma: no cover - restore failure
                pass
        return restored

    def rollback(self) -> VerificationResult:
        """Restore the exact pre-transaction bytes, refusing on unrelated changes."""
        if self._snapshot_hash is None or self._written_hash is None:
            return failure(
                "internal_error",
                "no committed transaction to roll back",
            )
        current = tree_hash(self.root)
        if current != self._written_hash:
            return failure(
                "transaction_conflict",
                "repository changed after the transaction; refusing to roll back",
                expected_tree_hash=self._written_hash,
                current_tree_hash=current,
            )
        restored = self._restore(sorted(self._overlay))
        after = tree_hash(self.root)
        if after != self._snapshot_hash:
            return failure(
                "transaction_conflict",
                "rollback did not reproduce the original tree hash",
                expected_tree_hash=self._snapshot_hash,
                current_tree_hash=after,
                restored=restored,
            )
        if self.indexer is not None:
            for rel_path in restored:
                try:
                    self.indexer.index_file(rel_path)
                except Exception:  # pragma: no cover
                    pass
        self._snapshot_hash = None
        self._written_hash = None
        return ok("transaction rolled back", restored=restored, tree_hash=after)
