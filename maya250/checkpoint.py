"""Maya-250M safe, hash-verified, atomic and resumable checkpoints (Schema 9).

Serializes RNG state using only weights-only-safe primitives and tensors so
``torch.load(..., weights_only=True)`` always succeeds across Kaggle GPU and
TPU sessions.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import tempfile
from typing import Optional

import numpy as np
import torch

from .model import MayaModel, hash_config

VERSION = 9


def _artifact_id(path: str) -> str:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def capture_rng_state() -> dict:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(np_state[0]),
            "keys": [int(x) for x in np_state[1].tolist()],
            "position": int(np_state[2]),
            "has_gaussian": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(tuple(state["python"]))
    np_state = state.get("numpy")
    if np_state:
        np.random.set_state((
            str(np_state["algorithm"]),
            np.asarray(np_state["keys"], dtype=np.uint32),
            int(np_state["position"]),
            int(np_state["has_gaussian"]),
            float(np_state["cached_gaussian"]),
        ))
    torch_state = state.get("torch_cpu")
    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _scheduler_state(scheduler):
    if scheduler is None:
        return None
    if hasattr(scheduler, "state_dict"):
        return scheduler.state_dict()
    return {
        "step_num": int(getattr(scheduler, "step_num", 0)),
        "base_lrs": list(getattr(scheduler, "base_lrs", [])),
    }


def save_checkpoint_maya(
    model: MayaModel,
    optimizer,
    scheduler,
    trainer_state: dict,
    path: str,
    extra: Optional[dict] = None,
) -> None:
    """Atomically write a Schema-9 weights-only-safe checkpoint."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = {
        "version": VERSION,
        "rng_format": 2,
        "config": model.config,
        "config_hash": hash_config(model.config),
        "tokenizer_hash": model.tokenizer_hash,
        "model_status": model.model_status,
        "weights_origin": getattr(model, "weights_origin", "random_init"),
        "optimizer_updates": int(getattr(model, "optimizer_updates", 0)),
        "tokens_seen": int(getattr(model, "tokens_seen", 0)),
        "checkpoint_artifact_id": getattr(model, "checkpoint_artifact_id", None),
        "capability_evaluation_status": getattr(
            model, "capability_evaluation_status", "not_evaluated"
        ),
        "trainer_state": trainer_state,
        "rng": capture_rng_state(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": _scheduler_state(scheduler),
        "extra": dict(extra or {}),
    }
    fd, tmp_path = tempfile.mkstemp(prefix=".maya_ckpt_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint_maya(
    path: str,
    model: MayaModel,
    tokenizer=None,
    optimizer=None,
    scheduler=None,
    restore_rng: bool = True,
) -> dict:
    """Load a Schema-9 checkpoint with strict hash and shape verification."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise ValueError("Checkpoint failed safe weights_only=True load") from exc

    required = ("version", "config", "config_hash", "tokenizer_hash", "model")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Checkpoint missing required fields: {missing}")

    if int(payload["version"]) != VERSION:
        raise ValueError(f"Unsupported checkpoint version {payload['version']} != {VERSION}")
    if hash_config(payload["config"]) != payload["config_hash"]:
        raise ValueError("Checkpoint config does not match config_hash")
    if payload["config_hash"] != model.config_hash():
        raise ValueError("Checkpoint config_hash mismatch with live model")

    ckpt_tok_hash = payload.get("tokenizer_hash")
    if not ckpt_tok_hash:
        raise ValueError("Checkpoint missing tokenizer_hash")
    if tokenizer is not None:
        if model.vocab != tokenizer.vocab_size:
            raise ValueError("Vocab mismatch between model and tokenizer")
        actual_tok_hash = tokenizer.tokenizer_hash()
    elif model.tokenizer_hash is not None:
        actual_tok_hash = model.tokenizer_hash
    else:
        raise ValueError("No tokenizer supplied to verify checkpoint")
    if ckpt_tok_hash != actual_tok_hash:
        raise ValueError("Checkpoint tokenizer_hash mismatch")

    current_state = model.state_dict()
    incoming_state = payload["model"]
    if set(current_state) != set(incoming_state):
        raise ValueError("Checkpoint state_dict keys do not match model")
    for key, target in current_state.items():
        source = incoming_state[key]
        if not isinstance(source, torch.Tensor) or tuple(source.shape) != tuple(target.shape):
            raise ValueError(f"Tensor shape mismatch at {key}")

    model.load_state_dict(incoming_state, strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        state = payload["scheduler"]
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(state)
        else:
            scheduler.step_num = int(state.get("step_num", 0))
            if state.get("base_lrs") is not None:
                scheduler.base_lrs = list(state["base_lrs"])

    if restore_rng:
        restore_rng_state(payload.get("rng") or {})

    model.checkpoint_loaded = True
    model.model_status = "trained_checkpoint_loaded"
    model.weights_origin = payload.get("weights_origin", "random_init")
    model.optimizer_updates = int(payload.get("optimizer_updates", 0))
    model.tokens_seen = int(payload.get("tokens_seen", 0))
    model.capability_evaluation_status = payload.get(
        "capability_evaluation_status", "not_evaluated"
    )
    model.checkpoint_artifact_id = payload.get("checkpoint_artifact_id") or _artifact_id(path)

    trainer_state = dict(payload.get("trainer_state") or {})
    if payload.get("extra"):
        trainer_state["checkpoint_extra"] = dict(payload["extra"])
    return trainer_state
