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

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m import (
    BPETokenizer,
    MemmapShardDataset,
    StreamPackedDataset,
    build_maya_model,
    deduplicate_and_split,
    get_config,
    train_maya,
)


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
        return {
            "backend": f"cuda_x{n_gpus}",
            "default_wall_hours": 11.25,
            "batch_size": 4 * max(1, n_gpus),
            "grad_accum": max(1, 32 // max(1, n_gpus)),
            "dtype": "float16",
        }
    return {
        "backend": "cpu",
        "default_wall_hours": 11.5,
        "batch_size": 2,
        "grad_accum": 2,
        "dtype": "float32",
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
    args = parser.parse_args()

    hw = detect_kaggle_accelerator()
    wall_hours = args.max_wall_hours if args.max_wall_hours is not None else hw["default_wall_hours"]
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        tok = BPETokenizer.load(args.tokenizer_path)
    else:
        from run_maya250m import SYNTHETIC_RESEARCH_CORPUS
        train_t, val_t, _test_t, _m = deduplicate_and_split(SYNTHETIC_RESEARCH_CORPUS)
        tok = BPETokenizer.train(train_t, vocab_size=384)
        tok.save(os.path.join(args.ckpt_dir, "tokenizer.json"))

    cfg = get_config(args.preset)
    cfg["vocab"] = tok.vocab_size
    model = build_maya_model(cfg, tok)

    if args.shard_prefix and os.path.exists(f"{args.shard_prefix}.meta.json"):
        train_ds = MemmapShardDataset(args.shard_prefix)
        val_ds = (
            MemmapShardDataset(args.val_shard_prefix)
            if args.val_shard_prefix and os.path.exists(f"{args.val_shard_prefix}.meta.json")
            else None
        )
    else:
        from run_maya250m import SYNTHETIC_RESEARCH_CORPUS
        train_t, val_t, _test_t, _m = deduplicate_and_split(SYNTHETIC_RESEARCH_CORPUS)
        train_ds = StreamPackedDataset(tok, train_t, seq_len=64)
        val_ds = StreamPackedDataset(tok, val_t, seq_len=64)

    last_ckpt = os.path.join(args.ckpt_dir, "maya250m_last.pt")
    best_ckpt = os.path.join(args.ckpt_dir, "maya250m_best.pt")
    resume_path = args.resume_ckpt if (args.resume_ckpt and os.path.exists(args.resume_ckpt)) else (
        last_ckpt if os.path.exists(last_ckpt) else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    )

    chain_status = {
        "accelerator": hw["backend"],
        "step": state["step"],
        "epoch": state["epoch"],
        "next_batch_cursor": state["next_batch_cursor"],
        "tokens_seen": model.tokens_seen,
        "stopped_by_wall_clock": state["stopped_by_wall_clock"],
        "chain_complete": state["step"] >= args.max_steps,
    }
    with open(os.path.join(args.ckpt_dir, "chain_state.json"), "w", encoding="utf-8") as f:
        json.dump(chain_status, f, indent=2)

    if args.kaggle_dataset_slug:
        push_kaggle_checkpoint_dataset(args.ckpt_dir, args.kaggle_dataset_slug, state["step"])

    print(json.dumps(chain_status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
