"""Large-codebase engineering: AST index, PPR repo map, packing, patches."""

from __future__ import annotations

from .codegen import (
    EDIT_FILE_MARKER,
    PatchBlock,
    PatchError,
    SearchReplacePatchEngine,
    apply_search_replace,
    parse_patch_blocks,
)
from .indexer import (
    DEFAULT_EXCLUDES,
    CallSite,
    FileNode,
    RepoGraphIndexer,
    SymbolDef,
    index_repository,
)
from .multifile import (
    attach_cross_file_symbols,
    build_cross_file_symbol_table,
    call_chain,
    interface_contract,
    verify_cross_file_impact,
)
from .packer import (
    FILE_SEP,
    FIM_MARKERS,
    FIMSpan,
    PackedRepo,
    RepoTokens,
    TopologicalRepoPacker,
    ast_span_candidates,
    fim_example,
)
from .repomap import RepoMap, RepoMapBuilder, RepoMapEntry

__all__ = [
    "EDIT_FILE_MARKER",
    "PatchBlock",
    "PatchError",
    "SearchReplacePatchEngine",
    "apply_search_replace",
    "parse_patch_blocks",
    "DEFAULT_EXCLUDES",
    "CallSite",
    "FileNode",
    "RepoGraphIndexer",
    "SymbolDef",
    "index_repository",
    "attach_cross_file_symbols",
    "build_cross_file_symbol_table",
    "call_chain",
    "interface_contract",
    "verify_cross_file_impact",
    "FILE_SEP",
    "FIM_MARKERS",
    "FIMSpan",
    "PackedRepo",
    "RepoTokens",
    "TopologicalRepoPacker",
    "ast_span_candidates",
    "fim_example",
    "RepoMap",
    "RepoMapBuilder",
    "RepoMapEntry",
]
