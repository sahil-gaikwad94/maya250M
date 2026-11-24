"""Kaggle Free-Quota Auto-Chaining Trainer for Maya-250M.

Supports both Kaggle free weekly accelerators seamlessly from the same Schema-9
checkpoint:
1. Kaggle TPU v3-8 / v5e-8 (20 hours/week quota, 9h session limit -> 8.25h safe cutoff).
2. Kaggle Dual Tesla T4 x2 GPU (30 hours/week quota, 12h session limit -> 11.25h safe cutoff).

When the wall-clock cutoff is reached, this script atomically writes
``/kaggle/working/ckpt/maya250m_last.pt``, writes ``chain_state.json``, and
optionally invokes the Kaggle CLI to version the checkpoint dataset so the
next scheduled/chained notebook run resumes at the exact next batch in the
epoch permutation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    class _StubCuda:
        @staticmethod
        def is_available() -> bool:
            return False
        @staticmethod
        def device_count() -> int:
            return 0
    class _StubTorch:
        cuda = _StubCuda()
        class device:
            def __init__(self, name: str):
                self._name = str(name)
            def __str__(self) -> str:
                return self._name
    torch = _StubTorch()  # type: ignore[assignment]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m import (
    BPETokenizer,
    MemmapShardDataset,
    StreamPackedDataset,
    deduplicate_and_split,
    get_config,
)
try:
    from maya250m import build_maya_model, train_maya
except ImportError:  # pragma: no cover
    pass


def detect_kaggle_accelerator() -> dict:
    """Detect whether the current Kaggle session is running on TPU, Dual-T4, or CPU."""
    if os.environ.get("PJRT_DEVICE") == "TPU" or os.environ.get("TPU_NAME"):
        return {
            "backend": "tpu",
            "default_wall_hours": 8.25,
            "batch_size": 8,
            "grad_accum": 4,
            "dtype": "bfloat16",
        }
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        # Size batch_size and grad_accum for the single active GPU process rather than
        # multiplying by n_gpus before multi-GPU DDP is active.
        return {
            "backend": f"cuda_x{n_gpus}",
            "default_wall_hours": 11.25,
            "batch_size": 4,
            "grad_accum": 16,
            "detected_devices": int(n_gpus),
            "active_devices": 1,
            "dtype": "float16",
        }
    return {
        "backend": "cpu",
        "default_wall_hours": 11.5,
        "batch_size": 2,
        "grad_accum": 2,
        "dtype": "float32",
    }


def resolve_training_device(hw: Dict[str, object]) -> Tuple[torch.device, Dict[str, object]]:
    """Map a detected accelerator onto a real device plus a strategy record.

    A TPU session used to fall through to CPU while still logging
    ``accelerator: tpu``: the run burned weekly quota at CPU speed and its
    throughput figure was meaningless. A missing torch_xla is therefore a hard
    error, never a silent fallback.
    """
    backend = str(hw.get("backend", "cpu"))
    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    if world_size > 1:
        raise SystemExit(
            f"NOT_RUN: WORLD_SIZE={world_size} requested on backend='{backend}', but "
            "train_maya250m_kaggle.py is a single-process trainer without DDP/PJRT "
            "all-reduce gradient synchronization."
        )

    if backend == "tpu":
        try:
            import torch_xla.core.xla_model as xm  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SystemExit(
                "detected a TPU session but torch_xla is not importable "
                f"({exc}); install it or unset PJRT_DEVICE/TPU_NAME — refusing to "
                "silently train on CPU"
            )
        if world_size > 1:
            raise SystemExit(
                f"multi-process TPU training (WORLD_SIZE={world_size}) is NOT_RUN in "
                "this single-process trainer; refusing to fake multi-core XLA execution"
            )
        return xm.xla_device(), {
            "backend": backend,
            "device": "xla",
            "parallel": "xla_single",
            "world_size": 1,
            "active_devices": 1,
        }

    if backend.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA backend detected but torch.cuda.is_available() is False; "
                "refusing to silently train on CPU"
            )
        if world_size > 1:
            raise SystemExit(
                f"multi-process CUDA DDP (WORLD_SIZE={world_size}) is NOT_RUN in "
                "this single-process trainer; refusing to fake multi-GPU DDP on a single rank"
            )
        local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
        return torch.device("cuda", local_rank), {
            "backend": backend,
            "device": f"cuda:{local_rank}",
            "parallel": "single_gpu",
            "world_size": 1,
            "active_devices": 1,
        }

    return torch.device("cpu"), {
        "backend": "cpu",
        "device": "cpu",
        "parallel": "single_cpu",
        "world_size": 1,
    }


def push_kaggle_checkpoint_dataset(ckpt_dir: str, dataset_slug: str, step: int) -> bool:
    """Version a private Kaggle dataset with the latest Schema-9 checkpoint."""
    if not dataset_slug:
        return False
    meta_path = os.path.join(ckpt_dir, "dataset-metadata.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "title": dataset_slug.split("/")[-1],
                    "id": dataset_slug,
                    "licenses": [{"name": "Apache-2.0"}],
                },
                f,
                indent=2,
            )
    try:
        res = subprocess.run(
            ["kaggle", "datasets", "version", "-p", ckpt_dir, "-m", f"Auto-checkpoint step {step}"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return res.returncode == 0
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Maya-250M Kaggle Free-Quota Auto-Chaining Trainer")
    parser.add_argument("--preset", type=str, default="MAYA_250M")
    parser.add_argument("--shard-prefix", type=str, default="")
    parser.add_argument("--val-shard-prefix", type=str, default="")
    parser.add_argument("--tokenizer-path", type=str, required=False, default="")
    parser.add_argument("--ckpt-dir", type=str, default="/kaggle/working/ckpt")
    parser.add_argument("--resume-ckpt", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=180000)
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--kaggle-dataset-slug", type=str, default="")
    parser.add_argument(
        "--smoke-test-synthetic",
        action="store_true",
        help=(
            "explicitly run the 384-vocab synthetic fixture smoke test; without "
            "this flag the flagship trainer requires a 32000-vocab tokenizer and "
            "real train/validation shards"
        ),
    )
    args = parser.parse_args()

    hw = detect_kaggle_accelerator()
    wall_hours = args.max_wall_hours if args.max_wall_hours is not None else hw["default_wall_hours"]

    # Validate data inputs before touching the filesystem or constructing a model, so
    # an accidental flagship invocation on a non-Kaggle host exits cleanly.
    if not args.smoke_test_synthetic:
        # The flagship path must never silently fall back to a 384-token synthetic
        # fixture: that produced a checkpoint whose tokenizer did not match any real
        # corpus and quietly replaced the preset vocabulary.
        missing = [
            flag
            for flag, value in (
                ("--tokenizer-path", args.tokenizer_path),
                ("--shard-prefix", args.shard_prefix),
                ("--val-shard-prefix", args.val_shard_prefix),
            )
            if not value
        ]
        if missing:
            print(
                json.dumps(
                    {
                        "error": "flagship training requires explicit data inputs",
                        "missing": missing,
                        "hint": (
                            "pass --tokenizer-path, --shard-prefix and "
                            "--val-shard-prefix, or pass --smoke-test-synthetic for "
                            "the synthetic fixture smoke test"
                        ),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2
        if not os.path.exists(args.tokenizer_path):
            raise SystemExit(f"--tokenizer-path does not exist: {args.tokenizer_path}")
        for label, prefix in (
            ("train", args.shard_prefix),
            ("validation", args.val_shard_prefix),
        ):
            if not os.path.exists(f"{prefix}.meta.json"):
                raise SystemExit(f"{label} shard metadata missing: {prefix}.meta.json")

        tok = BPETokenizer.load(args.tokenizer_path)
        cfg = get_config(args.preset)
        expected_vocab = 32000
        if int(tok.vocab_size) != expected_vocab:
            raise SystemExit(
                f"flagship training requires vocab_size {expected_vocab}; tokenizer "
                f"{args.tokenizer_path} has {tok.vocab_size}"
            )
        if int(cfg.get("vocab", expected_vocab)) != expected_vocab:
            raise SystemExit(
                f"preset {args.preset} declares vocab {cfg.get('vocab')}, expected "
                f"{expected_vocab} for flagship training"
            )
        cfg["vocab"] = int(tok.vocab_size)
        tokenizer_hash = tok.tokenizer_hash()
        for label, prefix in (
            ("train", args.shard_prefix),
            ("validation", args.val_shard_prefix),
        ):
            with open(f"{prefix}.meta.json", "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            declared = meta.get("tokenizer_hash")
            if declared != tokenizer_hash:
                raise SystemExit(
                    f"{label} shard tokenizer_hash {declared!r} does not match the "
                    f"tokenizer {tokenizer_hash!r}; refusing to train on mismatched ids"
                )
        model = build_maya_model(cfg, tok)
        train_ds = MemmapShardDataset(args.shard_prefix)
        val_ds = MemmapShardDataset(args.val_shard_prefix)
        checkpoint_role = "flagship"
    else:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        from run_maya250m import SYNTHETIC_RESEARCH_CORPUS

        train_t, val_t, _test_t, _m = deduplicate_and_split(SYNTHETIC_RESEARCH_CORPUS)
        tok = BPETokenizer.train(train_t, vocab_size=384)
        tok.save(os.path.join(args.ckpt_dir, "tokenizer.json"))
        cfg = get_config(args.preset)
        cfg["vocab"] = tok.vocab_size
        model = build_maya_model(cfg, tok)
        train_ds = StreamPackedDataset(tok, train_t, seq_len=64)
        val_ds = StreamPackedDataset(tok, val_t, seq_len=64)
        checkpoint_role = "smoke_test_only"

    last_ckpt = os.path.join(args.ckpt_dir, "maya250m_last.pt")
    best_ckpt = os.path.join(args.ckpt_dir, "maya250m_best.pt")
    resume_path = args.resume_ckpt if (args.resume_ckpt and os.path.exists(args.resume_ckpt)) else (
        last_ckpt if os.path.exists(last_ckpt) else None
    )

    device, strategy = resolve_training_device(hw)
    if strategy["parallel"] == "ddp_not_implemented":
        raise SystemExit(
            "WORLD_SIZE>1 requested but distributed data-parallel training is "
            "NOT_RUN in this repository; launch one process single-device, or "
            "implement and verify DDP before quoting multi-GPU scaling"
        )
    state = train_maya(
        model,
        train_ds,
        val_dataset=val_ds,
        batch_size=hw["batch_size"],
        grad_accum_steps=hw["grad_accum"],
        max_steps=args.max_steps,
        mtp_weight=0.2,
        intermediate_supervision_weight=0.05,
        use_mtp_curriculum=True,
        checkpoint_path=last_ckpt,
        best_checkpoint_path=best_ckpt,
        resume_from=resume_path,
        tokenizer=tok,
        max_wall_hours=wall_hours,
        device=device,
        checkpoint_role=checkpoint_role,
    )

    chain_status = {
        "accelerator": hw["backend"],
        "device": strategy["device"],
        "parallel": strategy["parallel"],
        "world_size": strategy["world_size"],
        "step": state["step"],
        "epoch": state["epoch"],
        "next_batch_cursor": state["next_batch_cursor"],
        "tokens_seen": model.tokens_seen,
        "stopped_by_wall_clock": state["stopped_by_wall_clock"],
        "chain_complete": state["step"] >= args.max_steps,
        "checkpoint_role": checkpoint_role,
    }
    with open(os.path.join(args.ckpt_dir, "chain_state.json"), "w", encoding="utf-8") as f:
        json.dump(chain_status, f, indent=2)

    if args.kaggle_dataset_slug:
        push_kaggle_checkpoint_dataset(args.ckpt_dir, args.kaggle_dataset_slug, state["step"])

    print(json.dumps(chain_status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
