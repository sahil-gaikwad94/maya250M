"""Hybrid-cache identity, snapshot/replay rewind and exact memory accounting.

v3 requires more than truncating K/V tensors: a hybrid model carries FP32 GDN
state, convolutional history and document history, none of which can be
"sliced" to rewind time. This module therefore provides:

* :class:`CacheIdentity` + ``stamp_bundle`` / ``validate_bundle`` — a cache is
  rejected when the config, weights, tokenizer, batch size, dtype, device,
  schema or loop count does not match the live model;
* :class:`CacheSnapshotStore` — periodic deep snapshots of the *complete* state;
* :func:`rewind_cache` — restore the closest snapshot at or before a target and
  (optionally) replay the intervening tokens deterministically;
* :func:`cache_memory_ledger` — bytes derived from instantiated cache tensors,
  not from a hand-written formula, for R = 1..4.

Never slice a recurrent state matrix to move backwards in time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "LEGACY_CACHE_SCHEMA_VERSION",
    "CacheError",
    "CacheIdentityError",
    "CacheRewindError",
    "CacheIdentity",
    "build_identity",
    "stamp_bundle",
    "validate_bundle",
    "deep_clone_cache",
    "CacheSnapshotStore",
    "rewind_cache",
    "cache_memory_ledger",
    "cache_memory_table",
]

CACHE_SCHEMA_VERSION = 3
LEGACY_CACHE_SCHEMA_VERSION = 2


class CacheError(RuntimeError):
    """Base class for cache contract violations."""


class CacheIdentityError(CacheError):
    """Raised when a cache does not belong to the model that would consume it."""


class CacheRewindError(CacheError):
    """Raised when a rewind cannot be performed safely."""


def deep_clone_cache(value: Any) -> Any:
    """Clone a cache bundle (tensors cloned, containers rebuilt)."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: deep_clone_cache(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deep_clone_cache(item) for item in value]
    if isinstance(value, tuple):
        return tuple(deep_clone_cache(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class CacheIdentity:
    """Everything a cache must agree with before it can be reused."""

    config_hash: str
    batch_size: int
    num_loops: int
    dtype: str
    device_type: str
    tokenizer_hash: Optional[str] = None
    checkpoint_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "batch_size": int(self.batch_size),
            "num_loops": int(self.num_loops),
            "dtype": self.dtype,
            "device_type": self.device_type,
            "tokenizer_hash": self.tokenizer_hash,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CacheIdentity":
        return cls(
            config_hash=str(payload.get("config_hash", "")),
            batch_size=int(payload.get("batch_size", 0)),
            num_loops=int(payload.get("num_loops", 0)),
            dtype=str(payload.get("dtype", "")),
            device_type=str(payload.get("device_type", "")),
            tokenizer_hash=payload.get("tokenizer_hash"),
            checkpoint_sha256=payload.get("checkpoint_sha256"),
        )


def build_identity(
    model,
    tokenizer: Any = None,
    batch_size: Optional[int] = None,
    num_loops: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    checkpoint_sha256: Optional[str] = None,
) -> CacheIdentity:
    """Derive the identity of a cache produced by ``model`` right now."""
    try:
        parameter = next(model.parameters())
    except StopIteration:  # pragma: no cover - a parameterless model is invalid
        parameter = torch.zeros(1)
    tok_hash = (
        tokenizer.tokenizer_hash()
        if tokenizer is not None and hasattr(tokenizer, "tokenizer_hash")
        else getattr(model, "tokenizer_hash", None)
    )
    return CacheIdentity(
        config_hash=model.config_hash(),
        batch_size=int(batch_size if batch_size is not None else 1),
        num_loops=int(num_loops if num_loops is not None else getattr(model, "recurrent_loops", 1)),
        dtype=str(dtype if dtype is not None else parameter.dtype),
        device_type=str(parameter.device.type),
        tokenizer_hash=tok_hash,
        checkpoint_sha256=(
            checkpoint_sha256
            if checkpoint_sha256 is not None
            else getattr(model, "checkpoint_artifact_id", None)
        ),
    )


def stamp_bundle(bundle: Dict[str, Any], identity: CacheIdentity) -> Dict[str, Any]:
    """Return ``bundle`` stamped as a schema-3 cache with full identity metadata."""
    if not isinstance(bundle, dict):
        raise TypeError("bundle must be a dict")
    stamped = dict(bundle)
    stamped["schema_version"] = CACHE_SCHEMA_VERSION
    stamped.update(identity.to_dict())
    stamped.setdefault("doc_state", {})
    return stamped


def validate_bundle(
    bundle: Any,
    identity: Optional[CacheIdentity] = None,
    allow_legacy: bool = True,
) -> int:
    """Validate a cache bundle; returns its schema version (2 or 3)."""
    if not isinstance(bundle, dict):
        raise CacheIdentityError("cache must be a dict bundle")
    schema = int(bundle.get("schema_version", 0) or 0)
    allowed = (CACHE_SCHEMA_VERSION, LEGACY_CACHE_SCHEMA_VERSION) if allow_legacy else (
        CACHE_SCHEMA_VERSION,
    )
    if schema not in allowed:
        raise CacheIdentityError(
            f"unsupported cache schema {schema!r}; expected one of {allowed}"
        )

    for key in ("num_loops", "batch_size", "position"):
        if key not in bundle:
            raise CacheIdentityError(f"cache bundle missing required field {key!r}")

    if identity is None:
        return schema

    problems: List[str] = []
    if str(bundle.get("config_hash", "")) != identity.config_hash:
        problems.append("config_hash")
    if int(bundle.get("batch_size", -1)) != int(identity.batch_size):
        problems.append("batch_size")
    if int(bundle.get("num_loops", -1)) != int(identity.num_loops):
        problems.append("num_loops")

    if schema == CACHE_SCHEMA_VERSION:
        if str(bundle.get("dtype", "")) != identity.dtype:
            problems.append("dtype")
        if str(bundle.get("device_type", "")) != identity.device_type:
            problems.append("device_type")
        for key in ("tokenizer_hash", "checkpoint_sha256"):
            cached = bundle.get(key)
            live = getattr(identity, key)
            if cached is not None and live is not None and str(cached) != str(live):
                problems.append(key)

    if problems:
        raise CacheIdentityError(
            "cache does not belong to this model/run; mismatched "
            + ", ".join(problems)
            + " — rebuild the cache after changing weights, tokenizer, batch or depth"
        )
    return schema


@dataclass
class CacheSnapshotStore:
    """Bounded store of complete cache snapshots for deterministic rewind."""

    interval: int = 64
    max_snapshots: int = 128
    snapshots: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("interval must be >= 1")
        if self.max_snapshots < 1:
            raise ValueError("max_snapshots must be >= 1")

    def snapshot(self, cache: Dict[str, Any], position: Optional[int] = None, force: bool = False) -> bool:
        """Store a snapshot; returns True when one was taken."""
        if not isinstance(cache, dict):
            raise CacheError("can only snapshot a cache bundle dict")
        pos = int(cache.get("position", 0) if position is None else position)
        if not force and pos % self.interval != 0 and pos != 0:
            return False
        self.snapshots[pos] = deep_clone_cache(cache)
        if len(self.snapshots) > self.max_snapshots:
            oldest = min(self.snapshots)
            del self.snapshots[oldest]
        return True

    def floor(self, position: int) -> Optional[int]:
        """Largest snapshot position <= ``position``."""
        candidates = [pos for pos in self.snapshots if pos <= int(position)]
        return max(candidates) if candidates else None

    def get(self, position: int) -> Dict[str, Any]:
        if int(position) not in self.snapshots:
            raise CacheRewindError(f"no snapshot at position {position}")
        return deep_clone_cache(self.snapshots[int(position)])

    def memory_bytes(self) -> int:
        total = 0
        for bundle in self.snapshots.values():
            total += _tensor_bytes(bundle)
        return total

    def clear(self) -> None:
        self.snapshots.clear()


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _as_token_tensor(replay_ids: Any) -> torch.Tensor:
    if isinstance(replay_ids, torch.Tensor):
        tensor = replay_ids
    else:
        tensor = torch.tensor(list(replay_ids), dtype=torch.long)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2:
        raise CacheRewindError("replay_ids must be [T] or [1, T]")
    return tensor


def rewind_cache(
    cache: Dict[str, Any],
    target_position: int,
    store: CacheSnapshotStore,
    model: Any = None,
    replay_ids: Any = None,
    identity: Optional[CacheIdentity] = None,
) -> Dict[str, Any]:
    """Rewind ``cache`` to ``target_position`` by snapshot restore (+ replay)."""
    validate_bundle(cache, identity)
    target = int(target_position)
    position = int(cache["position"])
    if target < 0:
        raise CacheRewindError("target_position must be >= 0")
    if target > position:
        raise CacheRewindError(
            f"cannot rewind forwards: target {target} > current position {position}"
        )
    if target == position:
        return deep_clone_cache(cache)

    start = store.floor(target)
    if start is None:
        raise CacheRewindError(
            f"no snapshot at or before position {target}; take snapshots before decoding"
        )
    restored = store.get(start)
    if start == target:
        return restored

    if model is None or replay_ids is None:
        raise CacheRewindError(
            f"rewind to {target} needs replay from snapshot {start}; "
            "pass the model and the intervening token ids"
        )

    tokens = _as_token_tensor(replay_ids)
    needed = target - start
    if tokens.shape[1] < needed:
        raise CacheRewindError(
            f"replay requires {needed} tokens, received {tokens.shape[1]}"
        )
    bundle = restored
    for index in range(needed):
        token = tokens[:, index : index + 1]
        _logits, bundle = model.decode_step(token, bundle)
    if int(bundle["position"]) != target:
        raise CacheRewindError(
            f"replay ended at {bundle['position']} instead of {target}"
        )
    return bundle


# --------------------------------------------------------------- memory ledger


def _classify_entries(value: Any, position: int) -> Tuple[int, int]:
    """Return ``(kv_bytes, recurrent_state_bytes)`` for one cache section."""
    kv_bytes = 0
    state_bytes = 0

    def walk(node: Any) -> None:
        nonlocal kv_bytes, state_bytes
        if isinstance(node, torch.Tensor):
            state_bytes += int(node.numel() * node.element_size())
            return
        if isinstance(node, dict):
            key_tensor = node.get("k")
            value_tensor = node.get("v")
            if (
                isinstance(key_tensor, torch.Tensor)
                and isinstance(value_tensor, torch.Tensor)
                and key_tensor.dim() == 4
                and int(key_tensor.shape[2]) == int(position)
            ):
                kv_bytes += int(
                    (key_tensor.numel() + value_tensor.numel()) * key_tensor.element_size()
                )
                for name, item in node.items():
                    if name not in ("k", "v"):
                        walk(item)
                return
            for item in node.values():
                walk(item)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(value)
    return kv_bytes, state_bytes


def cache_memory_ledger(
    model,
    batch_size: int = 1,
    ctx: Optional[int] = None,
    loops: Optional[int] = None,
    kv_dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Exact cache bytes for one loop count, measured from instantiated tensors.

    The measurement uses a one-token prefill, so K/V tensors hold exactly one
    position and can be scaled linearly to the target context; recurrent state and
    convolutional history are context-independent and counted as-is.
    """
    model_dtype = next(model.parameters()).dtype
    target_ctx = int(ctx if ctx is not None else model.max_ctx)
    circles = int(loops if loops is not None else model.recurrent_loops)

    probe = torch.zeros((int(batch_size), 1), dtype=torch.long, device=device or "cpu")
    with torch.no_grad():
        _logits, bundle, _info = model.prefill(probe, num_loops=circles)
    position = int(bundle.get("position", 1)) or 1

    prelude_kv, prelude_state = _classify_entries(bundle.get("prelude"), position)
    coda_kv, coda_state = _classify_entries(bundle.get("coda"), position)
    core_pass = (bundle.get("core_loops") or [[]])[0]
    core_kv_per_pass, core_state_per_pass = _classify_entries(core_pass, position)

    kv_scale = 1.0
    if kv_dtype is not None and kv_dtype != model_dtype:
        kv_scale = float(torch.tensor([], dtype=kv_dtype).element_size()) / float(
            torch.tensor([], dtype=model_dtype).element_size()
        )

    kv_bytes_per_token = (
        prelude_kv + coda_kv + circles * core_kv_per_pass
    ) / float(position)
    kv_total = int(round(kv_bytes_per_token * target_ctx * kv_scale))
    recurrent_total = int(prelude_state + coda_state + circles * core_state_per_pass)
    combined = kv_total + recurrent_total

    n_full_core = 0
    n_gdn_core = 0
    for block in list(model.blocks)[model.prelude_layers : model.prelude_layers + model.core_layers]:
        if hasattr(block.mixer, "A_log"):
            n_gdn_core += 1
        else:
            n_full_core += 1

    return {
        "loops": circles,
        "max_ctx": target_ctx,
        "batch_size": int(batch_size),
        "model_dtype": str(model_dtype),
        "kv_dtype": str(kv_dtype if kv_dtype is not None else model_dtype),
        "kv_cache_bytes": kv_total,
        "recurrent_state_bytes": recurrent_total,
        "combined_bytes": combined,
        "combined_mib": round(combined / (1024.0 * 1024.0), 3),
        "components": {
            "kv_per_token_bytes": round(kv_bytes_per_token, 1),
            "prelude_coda_kv_bytes_per_token": int((prelude_kv + coda_kv) / float(position)),
            "core_kv_bytes_per_pass_per_token": int(core_kv_per_pass / float(position)),
            "prelude_coda_state_bytes": int(prelude_state + coda_state),
            "core_state_bytes_per_pass": int(core_state_per_pass),
            "core_full_attention_layers": n_full_core,
            "core_gdn_layers": n_gdn_core,
        },
    }


def cache_memory_table(
    model,
    loop_counts: Iterable[int] = (1, 2, 3, 4),
    batch_size: int = 1,
    ctx: Optional[int] = None,
    kv_dtype: Optional[torch.dtype] = None,
) -> List[Dict[str, Any]]:
    """Ledger for R = 1..4 (or any supplied loop counts)."""
    return [
        cache_memory_ledger(
            model, batch_size=batch_size, ctx=ctx, loops=int(loops), kv_dtype=kv_dtype
        )
        for loops in loop_counts
    ]
