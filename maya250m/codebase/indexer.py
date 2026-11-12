"""Deterministic AST repository indexer.

Parses every Python file under a repository root into a structural graph of
definitions (functions, async functions, methods, classes, module constants),
their exact line spans, signatures, imports, call sites and inheritance edges.

Why this exists: a 100k-line repository is ~37x Maya's 32k-token window. Accurate
localization therefore cannot come from chunking raw text — it has to come from a
symbol graph that knows *where* a definition starts and ends, *who* calls it, and
*which* file it must be imported from. Everything here is deterministic and
content-addressed (SHA-256 per file) so an index can be cached and verified.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "SymbolDef",
    "CallSite",
    "FileNode",
    "RepoGraphIndexer",
    "DEFAULT_EXCLUDES",
    "index_repository",
]

DEFAULT_EXCLUDES = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "site-packages",
    ".eggs",
)


@dataclass
class SymbolDef:
    """One definition in one file."""

    symbol_id: str
    name: str
    qualname: str
    kind: str  # function | async_function | class | method | async_method | constant
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring_summary: str = ""
    calls: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    min_arity: int = 0
    max_arity: Optional[int] = 0
    return_type: str = "any"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SymbolDef":
        return cls(**payload)


@dataclass
class CallSite:
    """One call expression, with the information needed for arity checking."""

    callee: str
    file_path: str
    line: int
    n_positional: int
    has_starargs: bool = False
    has_kwargs: bool = False
    keyword_names: List[str] = field(default_factory=list)
    caller_qualname: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CallSite":
        return cls(**payload)


@dataclass
class FileNode:
    """One Python file and its extracted structure."""

    file_path: str
    sha256: str
    line_count: int
    imports: List[str] = field(default_factory=list)
    symbols: List[SymbolDef] = field(default_factory=list)
    calls: List[CallSite] = field(default_factory=list)
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "sha256": self.sha256,
            "line_count": self.line_count,
            "imports": list(self.imports),
            "symbols": [s.to_dict() for s in self.symbols],
            "calls": [c.to_dict() for c in self.calls],
            "parse_error": self.parse_error,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FileNode":
        return cls(
            file_path=payload["file_path"],
            sha256=payload["sha256"],
            line_count=int(payload["line_count"]),
            imports=list(payload.get("imports") or []),
            symbols=[SymbolDef.from_dict(s) for s in payload.get("symbols") or []],
            calls=[CallSite.from_dict(c) for c in payload.get("calls") or []],
            parse_error=payload.get("parse_error"),
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _format_signature(node: ast.AST, name: str) -> str:
    """Rebuild a readable one-line signature from an AST function node."""
    args = getattr(node, "args", None)
    if args is None:
        return f"class {name}:"
    parts: List[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    n_no_default = len(positional) - len(defaults)
    for idx, arg in enumerate(positional):
        rendered = arg.arg
        if arg.annotation is not None:
            rendered += f": {ast.unparse(arg.annotation)}"
        if idx >= n_no_default:
            rendered += f" = {ast.unparse(defaults[idx - n_no_default])}"
        parts.append(rendered)
        if args.posonlyargs and idx == len(args.posonlyargs) - 1:
            parts.append("/")
    if args.vararg is not None:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        rendered = arg.arg
        if arg.annotation is not None:
            rendered += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            rendered += f" = {ast.unparse(default)}"
        parts.append(rendered)
    if args.kwarg is not None:
        parts.append("**" + args.kwarg.arg)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) else ""
    return f"{prefix} {name}({', '.join(parts)}){returns}:"


def _arity(node: ast.AST) -> Tuple[int, Optional[int]]:
    args = getattr(node, "args", None)
    if args is None:
        return 0, 0
    positional = list(args.posonlyargs) + list(args.args)
    if positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
    n_pos = len(positional)
    n_def = len(getattr(args, "defaults", []) or [])
    has_vararg = getattr(args, "vararg", None) is not None
    has_kwarg = getattr(args, "kwarg", None) is not None
    # Keyword-only arguments without defaults are required too.
    kw_required = sum(
        1
        for arg, default in zip(
            getattr(args, "kwonlyargs", []) or [],
            getattr(args, "kw_defaults", []) or [],
        )
        if default is None
    )
    return max(0, n_pos - n_def), (None if (has_vararg or has_kwarg) else n_pos + kw_required)


def _docstring(node: ast.AST) -> str:
    doc = ast.get_docstring(node) if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ) else None
    if not doc:
        return ""
    return doc.strip().splitlines()[0][:200]


def _callee_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _callee_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return None


class RepoGraphIndexer:
    """Build, query, persist and incrementally update a repository symbol graph."""

    def __init__(
        self,
        root: str,
        excludes: Sequence[str] = DEFAULT_EXCLUDES,
        max_files: int = 20000,
        max_file_bytes: int = 2_000_000,
    ):
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise ValueError(f"repository root does not exist: {self.root}")
        self.excludes = tuple(excludes)
        self.max_files = int(max_files)
        self.max_file_bytes = int(max_file_bytes)
        self.files: Dict[str, FileNode] = {}
        self._by_qualname: Dict[str, List[SymbolDef]] = {}
        self._by_name: Dict[str, List[SymbolDef]] = {}

    # ------------------------------------------------------------------ walking

    def _iter_python_files(self) -> Iterable[str]:
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(
                d for d in dirnames if d not in self.excludes and not d.startswith(".")
            )
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                full = os.path.join(dirpath, filename)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                count += 1
                if count > self.max_files:
                    return
                yield rel

    # ----------------------------------------------------------------- indexing

    def index(self, paths: Optional[Sequence[str]] = None) -> Dict[str, FileNode]:
        """Index ``paths`` (relative) or the whole repository; returns the file map."""
        targets = list(paths) if paths is not None else list(self._iter_python_files())
        for rel in targets:
            self.index_file(rel)
        self._rebuild_lookup()
        return self.files

    def index_file(self, rel_path: str) -> Optional[FileNode]:
        """Parse and store one file. Returns ``None`` if it is skipped or too large."""
        rel = rel_path.replace(os.sep, "/").lstrip("./")
        full = os.path.join(self.root, rel)
        try:
            if os.path.getsize(full) > self.max_file_bytes:
                return None
            with open(full, "rb") as handle:
                raw = handle.read()
        except OSError:
            return None

        digest = _sha256_bytes(raw)
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
        line_count = source.count("\n") + (1 if source and not source.endswith("\n") else 0)

        node = FileNode(file_path=rel, sha256=digest, line_count=line_count)
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            node.parse_error = f"{exc.__class__.__name__}: {exc.msg} (line {exc.lineno})"
            self.files[rel] = node
            return node

        node.imports = self._extract_imports(tree, rel)
        self._walk_module(tree, rel, node)
        self.files[rel] = node
        self._rebuild_lookup()
        return node

    def _extract_imports(self, tree: ast.AST, rel: str) -> List[str]:
        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(self._resolve_module(alias.name, rel, 0))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                resolved = self._resolve_module(module, rel, node.level or 0)
                imports.append(resolved)
        deduped: List[str] = []
        for item in imports:
            if item and item not in deduped:
                deduped.append(item)
        return deduped

    def _resolve_module(self, module: str, from_file: str, level: int) -> str:
        """Resolve a module reference to a repository-relative ``.py`` path when possible."""
        if level > 0:
            base_dir = os.path.dirname(from_file)
            for _ in range(max(0, level - 1)):
                base_dir = os.path.dirname(base_dir)
            candidate_rel = (
                os.path.normpath(os.path.join(base_dir, module.replace(".", "/")))
                if module
                else base_dir
            )
        else:
            candidate_rel = os.path.normpath(module.replace(".", "/")) if module else ""
        candidate_rel = str(candidate_rel).replace(os.sep, "/").lstrip("./")
        if not candidate_rel:
            return ""
        for suffix in (".py", "/__init__.py"):
            candidate = candidate_rel + suffix
            if os.path.exists(os.path.join(self.root, candidate)):
                return candidate
        if level == 0:
            # Absolute import from the repository root (e.g. ``maya250m.train``).
            candidate = candidate_rel + ".py"
            if os.path.exists(os.path.join(self.root, candidate)):
                return candidate
        return module or ""

    def _walk_module(self, tree: ast.Module, rel: str, node: FileNode) -> None:
        def visit(body: Sequence[ast.stmt], class_prefix: str = "") -> None:
            for stmt in body:
                if isinstance(stmt, ast.ClassDef):
                    qualname = f"{class_prefix}{stmt.name}"
                    node.symbols.append(
                        SymbolDef(
                            symbol_id=f"{rel}::{qualname}",
                            name=stmt.name,
                            qualname=qualname,
                            kind="class",
                            file_path=rel,
                            start_line=int(stmt.lineno),
                            end_line=int(getattr(stmt, "end_lineno", stmt.lineno)),
                            signature=f"class {stmt.name}:",
                            docstring_summary=_docstring(stmt),
                            calls=[],
                            bases=[ast.unparse(b) for b in stmt.bases],
                        )
                    )
                    visit(stmt.body, class_prefix=qualname + ".")
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f"{class_prefix}{stmt.name}"
                    is_method = bool(class_prefix)
                    if isinstance(stmt, ast.AsyncFunctionDef):
                        kind = "async_method" if is_method else "async_function"
                    else:
                        kind = "method" if is_method else "function"
                    min_arity, max_arity = _arity(stmt)
                    returns = stmt.returns
                    node.symbols.append(
                        SymbolDef(
                            symbol_id=f"{rel}::{qualname}",
                            name=stmt.name,
                            qualname=qualname,
                            kind=kind,
                            file_path=rel,
                            start_line=int(stmt.lineno),
                            end_line=int(getattr(stmt, "end_lineno", stmt.lineno)),
                            signature=_format_signature(stmt, stmt.name),
                            docstring_summary=_docstring(stmt),
                            calls=self._collect_calls(stmt),
                            min_arity=min_arity,
                            max_arity=max_arity,
                            return_type=ast.unparse(returns) if returns is not None else "any",
                        )
                    )
                    node.calls.extend(self._collect_call_sites(stmt, rel, qualname))
                    visit(stmt.body, class_prefix=class_prefix)
                elif isinstance(stmt, ast.Assign) and not class_prefix:
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            node.symbols.append(
                                SymbolDef(
                                    symbol_id=f"{rel}::{target.id}",
                                    name=target.id,
                                    qualname=target.id,
                                    kind="constant",
                                    file_path=rel,
                                    start_line=int(stmt.lineno),
                                    end_line=int(getattr(stmt, "end_lineno", stmt.lineno)),
                                    signature=f"{target.id} = ...",
                                )
                            )
                elif isinstance(stmt, (ast.If, ast.With, ast.Try, ast.For, ast.While)):
                    # Nested control flow at module level can still define symbols.
                    for attr in ("body", "orelse", "finalbody"):
                        nested = getattr(stmt, attr, None)
                        if isinstance(nested, list):
                            visit(nested, class_prefix=class_prefix)
                    for handler in getattr(stmt, "handlers", []) or []:
                        visit(handler.body, class_prefix=class_prefix)

        visit(tree.body)

    def _collect_calls(self, node: ast.AST) -> List[str]:
        calls: List[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = _callee_name(child.func)
                if name and name not in calls:
                    calls.append(name)
        return calls

    def _collect_call_sites(self, node: ast.AST, rel: str, caller: str) -> List[CallSite]:
        sites: List[CallSite] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            name = _callee_name(child.func)
            if not name:
                continue
            has_star = any(isinstance(a, ast.Starred) for a in child.args)
            sites.append(
                CallSite(
                    callee=name,
                    file_path=rel,
                    line=int(getattr(child, "lineno", 0)),
                    n_positional=sum(1 for a in child.args if not isinstance(a, ast.Starred)),
                    has_starargs=has_star,
                    has_kwargs=bool(child.keywords and any(k.arg is None for k in child.keywords)),
                    keyword_names=[k.arg for k in child.keywords if k.arg],
                    caller_qualname=caller,
                )
            )
        return sites

    # ----------------------------------------------------------------- lookups

    def _rebuild_lookup(self) -> None:
        self._by_qualname = {}
        self._by_name = {}
        for node in self.files.values():
            for symbol in node.symbols:
                self._by_qualname.setdefault(symbol.qualname, []).append(symbol)
                self._by_name.setdefault(symbol.name, []).append(symbol)

    def definitions(self, qualname: str) -> List[SymbolDef]:
        return list(self._by_qualname.get(qualname, []))

    def symbols_named(self, name: str) -> List[SymbolDef]:
        return list(self._by_name.get(name, []))

    def callers(self, name: str) -> List[CallSite]:
        """Every recorded call site whose callee basename equals ``name``."""
        out: List[CallSite] = []
        for node in self.files.values():
            for site in node.calls:
                if site.callee == name or site.callee.rsplit(".", 1)[-1] == name:
                    out.append(site)
        return out

    def callees(self, qualname: str) -> List[str]:
        for symbol in self.definitions(qualname):
            return list(symbol.calls)
        return []

    def file_symbols(self, rel_path: str) -> List[SymbolDef]:
        node = self.files.get(rel_path.replace(os.sep, "/"))
        return list(node.symbols) if node else []

    def signature_table(self, names: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
        """``name -> {min_arity, max_arity, return_type}`` for Vexx in-flight masking."""
        table: Dict[str, Dict[str, Any]] = {}
        for name, symbols in self._by_name.items():
            if names is not None and name not in set(names):
                continue
            concrete = [s for s in symbols if s.kind != "class"]
            if not concrete:
                continue
            # Prefer the shallowest definition; ambiguous names get the widest bounds.
            concrete.sort(key=lambda s: (s.qualname.count("."), s.file_path))
            mins = [s.min_arity for s in concrete]
            maxes = [s.max_arity for s in concrete]
            table[name] = {
                "min_arity": min(mins),
                "max_arity": None if any(m is None for m in maxes) else max(maxes),
                "return_type": concrete[0].return_type or "any",
            }
        return table

    def summary(self) -> Dict[str, Any]:
        symbols = sum(len(node.symbols) for node in self.files.values())
        errors = {p: n.parse_error for p, n in self.files.items() if n.parse_error}
        return {
            "root": self.root,
            "files": len(self.files),
            "symbols": symbols,
            "call_sites": sum(len(n.calls) for n in self.files.values()),
            "parse_errors": errors,
            "lines": sum(n.line_count for n in self.files.values()),
        }

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "files": {path: node.to_dict() for path, node in sorted(self.files.items())},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RepoGraphIndexer":
        indexer = cls(payload["root"])
        indexer.files = {
            path: FileNode.from_dict(node) for path, node in payload["files"].items()
        }
        indexer._rebuild_lookup()
        return indexer

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "RepoGraphIndexer":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def stale_files(self) -> List[str]:
        """Files whose on-disk SHA-256 no longer matches the index."""
        stale: List[str] = []
        for rel, node in self.files.items():
            full = os.path.join(self.root, rel)
            try:
                with open(full, "rb") as handle:
                    if _sha256_bytes(handle.read()) != node.sha256:
                        stale.append(rel)
            except OSError:
                stale.append(rel)
        return sorted(stale)


def index_repository(root: str, **kwargs: Any) -> RepoGraphIndexer:
    """Convenience helper: construct an indexer and index the whole repository."""
    indexer = RepoGraphIndexer(root, **kwargs)
    indexer.index()
    return indexer
