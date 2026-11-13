"""Repository-facing alias for the hybrid cache contract (v3 report section 3).

The implementation lives in :mod:`maya250m.cache` because it is a model-runtime
concern, not a code-intelligence one. The specification lists this path, so it is
re-exported here rather than duplicated.
"""

from __future__ import annotations

from ..cache import (
    CACHE_SCHEMA_VERSION,
    LEGACY_CACHE_SCHEMA_VERSION,
    CacheError,
    CacheIdentity,
    CacheIdentityError,
    CacheRewindError,
    CacheSnapshotStore,
    build_identity,
    cache_memory_ledger,
    cache_memory_table,
    deep_clone_cache,
    rewind_cache,
    stamp_bundle,
    validate_bundle,
)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "LEGACY_CACHE_SCHEMA_VERSION",
    "CacheError",
    "CacheIdentity",
    "CacheIdentityError",
    "CacheRewindError",
    "CacheSnapshotStore",
    "build_identity",
    "cache_memory_ledger",
    "cache_memory_table",
    "deep_clone_cache",
    "rewind_cache",
    "stamp_bundle",
    "validate_bundle",
]
