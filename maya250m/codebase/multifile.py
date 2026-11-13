"""Cross-file interface contracts and impact verification.

The failure this prevents is *cross-file interface amnesia*: while editing
``train.py`` a small model changes ``train_maya``'s arity, and the call site in
``kaggle/train_maya250m_kaggle.py`` — long outside the attention window — silently
breaks.

Two operations:

* :func:`build_cross_file_symbol_table` — resolve a file's imports against the
  repository index and emit exact ``(min_arity, max_arity, return_type)`` contracts,
  ready to inject into ``Vexx35Controller.symbol_table`` for in-flight masking.
* :func:`verify_cross_file_impact` — after a change, check every recorded call site
  outside the changed file set and fail with the exact file/line of each break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..verification import VerificationResult, failure, ok
from .contracts import CallableContract, check_call
from .indexer import RepoGraphIndexer, SymbolDef

__all__ = [
    "build_cross_file_symbol_table",
    "attach_cross_file_symbols",
    "verify_cross_file_impact",
    "call_chain",
    "interface_contract",
]


def interface_contract(symbol: SymbolDef) -> Dict[str, Any]:
    """Contract for one symbol, in the shape Vexx's in-flight masker expects."""
    return {
        "min_arity": int(symbol.min_arity),
        "max_arity": symbol.max_arity,
        "return_type": symbol.return_type or "any",
    }


def build_cross_file_symbol_table(
    indexer: RepoGraphIndexer,
    file_path: str,
    include_same_file: bool = True,
    include_repo_globals: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Resolve ``file_path``'s imports into repository-wide symbol contracts."""
    rel = file_path.replace("\\", "/")
    node = indexer.files.get(rel)
    if node is None:
        raise KeyError(f"file not indexed: {file_path}")

    table: Dict[str, Dict[str, Any]] = {}
    for imported in node.imports:
        imported_node = indexer.files.get(imported)
        if imported_node is None:
            continue
        for symbol in imported_node.symbols:
            if symbol.kind == "constant":
                continue
            table.setdefault(symbol.name, interface_contract(symbol))

    if include_same_file:
        for symbol in node.symbols:
            if symbol.kind == "constant":
                continue
            table[symbol.name] = interface_contract(symbol)

    if include_repo_globals:
        for name, contract in indexer.signature_table().items():
            table.setdefault(name, contract)
    return table


def attach_cross_file_symbols(
    vexx: Any, indexer: RepoGraphIndexer, file_path: str, **kwargs: Any
) -> Dict[str, Dict[str, Any]]:
    """Populate ``vexx.symbol_table`` with cross-file contracts for ``file_path``."""
    table = build_cross_file_symbol_table(indexer, file_path, **kwargs)
    vexx.symbol_table.update(table)
    return table


def _callers_outside(
    indexer: RepoGraphIndexer, name: str, changed_files: Set[str]
) -> List[Any]:
    return [site for site in indexer.callers(name) if site.file_path not in changed_files]


def verify_cross_file_impact(
    indexer: RepoGraphIndexer,
    changed_signatures: Dict[str, Optional[Dict[str, Any]]],
    changed_files: Sequence[str] = (),
    max_reported: int = 10,
) -> VerificationResult:
    """Check every caller of the changed symbols against the new contracts.

    ``changed_signatures`` maps a symbol name to its new contract, or ``None`` when
    the symbol was removed. Call sites inside ``changed_files`` are assumed to be
    updated in the same patch set and are skipped.
    """
    changed = {path.replace("\\", "/") for path in changed_files}
    breakages: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for name, contract in changed_signatures.items():
        callers = _callers_outside(indexer, name, changed)
        if contract is None:
            for site in callers:
                missing.append(
                    {
                        "symbol": name,
                        "caller_file": site.file_path,
                        "line": site.line,
                        "caller": site.caller_qualname,
                        "reason": "symbol_removed",
                    }
                )
            continue

        typed: Optional[CallableContract] = None
        unverified = 0
        if isinstance(contract.get("parameters"), (list, tuple)):
            typed = CallableContract.from_dict(contract)

        for site in callers:
            if typed is not None:
                if not _attribute_call_is_statically_bound(site, indexer):
                    unverified += 1
                    continue
                check = check_call(
                    typed,
                    n_positional=int(site.n_positional),
                    keyword_names=list(site.keyword_names),
                    has_starargs=bool(site.has_starargs),
                    has_kwargs=bool(site.has_kwargs),
                )
                if check.passed or check.status == "unknown_dynamic_dispatch":
                    continue
                breakages.append(
                    {
                        "symbol": name,
                        "caller_file": site.file_path,
                        "line": site.line,
                        "caller": site.caller_qualname,
                        "n_positional": int(site.n_positional),
                        "keyword_names": list(site.keyword_names),
                        "kind": check.kind,
                        "status": check.status,
                        "message": check.message,
                    }
                )
                continue

            # Legacy contract dicts (no full parameter list) only carry arity.
            if site.has_starargs or site.has_kwargs:
                # Cannot statically prove a break: report as *unverified*, not failed.
                continue
            min_arity = int(contract.get("min_arity", 0))
            max_arity = contract.get("max_arity", None)
            n_args = int(site.n_positional)
            if n_args < min_arity or (max_arity is not None and n_args > max_arity):
                breakages.append(
                    {
                        "symbol": name,
                        "caller_file": site.file_path,
                        "line": site.line,
                        "caller": site.caller_qualname,
                        "n_positional": n_args,
                        "min_arity": min_arity,
                        "max_arity": max_arity,
                    }
                )

    if missing:
        return failure(
            "cross_file_missing_symbol",
            f"{len(missing)} call site(s) reference a removed symbol",
            violations=missing[:max_reported],
        )
    if breakages:
        first = breakages[0]
        if "status" in first:
            detail = first.get("message") or first["status"]
        else:
            detail = (
                f"passes {first['n_positional']} positional arg(s) but the contract is "
                f"[{first['min_arity']}, {first['max_arity']}]"
            )
        return failure(
            "cross_file_arity_breakage",
            (
                f"{len(breakages)} call site(s) break the new signature of "
                f"{first['symbol']!r}; first: {first['caller_file']}:{first['line']} "
                f"{detail}"
            ),
            violations=breakages[:max_reported],
            unverified_arity_checks=unverified,
        )
    return ok("no cross-file contract violations", checked_symbols=len(changed_signatures))


def _attribute_call_is_statically_bound(site: CallSite, indexer: RepoGraphIndexer) -> bool:
    """Only trust ``a.b(...)`` when the receiver is statically known.

    ``self.f`` / ``cls.f``, an imported module, or a class defined in the file can
    be resolved. A local variable, a parameter or a return value is dynamic
    dispatch: the call is counted as *unverified* rather than rejected.
    """
    callee = getattr(site, "callee", "")
    if "." not in callee:
        return True
    prefix, _, receiver = callee.rpartition(".")
    if receiver in ("self", "cls"):
        return True
    if prefix.split(".")[-1] == "super":
        return True
    node = indexer.files.get(site.file_path)
    if node is None:
        return False
    for imported in getattr(node, "imports", []) or []:
        imported = str(imported)
        base = imported.rstrip("/").rsplit("/", 1)[-1]
        if base == "__init__.py":
            parts = imported.rstrip("/").split("/")
            base = parts[-2] if len(parts) > 1 else ""
        if base.removesuffix(".py") == receiver:
            return True
        if imported.removesuffix(".py") == prefix.replace(".", "/"):
            return True
    for symbol in getattr(node, "symbols", []) or []:
        if symbol.kind == "class" and symbol.name == receiver:
            return True
    return False


def call_chain(
    indexer: RepoGraphIndexer, qualname: str, max_depth: int = 3
) -> List[Dict[str, Any]]:
    """Breadth-first call chain from ``qualname`` (used by MF-2 dataflow tracing)."""
    frontier: List[Tuple[str, int]] = [(qualname, 0)]
    seen: Set[str] = {qualname}
    chain: List[Dict[str, Any]] = []
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= int(max_depth):
            continue
        for callee in indexer.callees(current):
            basename = callee.rsplit(".", 1)[-1]
            definitions = indexer.symbols_named(basename)
            if not definitions:
                chain.append(
                    {
                        "from": current,
                        "to": callee,
                        "depth": depth + 1,
                        "resolved": False,
                    }
                )
                continue
            target = definitions[0]
            chain.append(
                {
                    "from": current,
                    "to": target.qualname,
                    "file": target.file_path,
                    "line": target.start_line,
                    "depth": depth + 1,
                    "resolved": True,
                }
            )
            if target.qualname not in seen:
                seen.add(target.qualname)
                frontier.append((target.qualname, depth + 1))
    return chain
