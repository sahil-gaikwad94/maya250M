"""Standalone end-to-end runner for Maya-250M + Vexx 3.5.

Executes:
1. SHA-256 deduplication and leak-free train/val/test split.
2. Byte-level BPE tokenizer training (with <think> and <experiment> protocol tokens).
3. MayaModel construction + Muon/AdamW hybrid training with Forward MTP Curriculum.
4. Schema-9 atomic checkpoint save, reload, and held-out evaluation.
5. Vexx 3.5 neuro-symbolic generation with CAS and <experiment> execution splices.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from maya250m import (
    PLAIN,
    PYTHON,
    SolveConfig,
    generate_adaptive,
    BPETokenizer,
    StreamPackedDataset,
    Vexx35Controller,
    build_maya_model,
    deduplicate_and_split,
    evaluate_maya,
    generate_maya,
    get_config,
    load_checkpoint_maya,
    train_maya,
)


SYNTHETIC_RESEARCH_CORPUS = [
    f"<think>Let us derive the sum of the first {k} integers: <<{k}*({k}+1)/2={(k*(k+1))//2}>>. "
    f"<experiment>RESULT = sum(range(1, {k + 1}))</experiment></think>\n"
    f"def sum_to_{k}(n: int) -> int:\n    return n * (n + 1) // 2\n"
    for k in range(2, 66)
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Maya-250M + Vexx 3.5 End-to-End Runner")
    parser.add_argument("--preset", type=str, default="MAYA_TINY_TEST")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--mtp-depth", type=int, default=3)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_maya")
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--adaptive-depth",
        action="store_true",
        help="choose recurrent-core loop count per prompt (AdaptiveLoopController)",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    train_texts, val_texts, test_texts, manifest = deduplicate_and_split(
        SYNTHETIC_RESEARCH_CORPUS, train_frac=0.75, val_frac=0.125, seed=42
    )
    tok = BPETokenizer.train(train_texts, vocab_size=384, min_pair_freq=2)
    tok.validate()

    cfg = get_config(args.preset)
    cfg["vocab"] = tok.vocab_size
    cfg["mtp_depth"] = args.mtp_depth
    cfg["recurrent_loops"] = args.loops

    model = build_maya_model(cfg, tok)
    train_ds = StreamPackedDataset(tok, train_texts, seq_len=args.seq_len, seed=42)
    val_ds = StreamPackedDataset(tok, val_texts, seq_len=args.seq_len, seed=43)
    test_ds = StreamPackedDataset(tok, test_texts, seq_len=args.seq_len, seed=44)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    last_ckpt = os.path.join(args.checkpoint_dir, "maya_last.pt")
    best_ckpt = os.path.join(args.checkpoint_dir, "maya_best.pt")
    tok_path = os.path.join(args.checkpoint_dir, "tokenizer.json")
    tok.save(tok_path)

    state = train_maya(
        model,
        train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_steps=args.steps,
        mtp_weight=0.2,
        intermediate_supervision_weight=0.05,
        use_mtp_curriculum=True,
        checkpoint_path=last_ckpt,
        best_checkpoint_path=best_ckpt,
        tokenizer=tok,
        seed=42,
        max_wall_hours=args.max_wall_hours,
    )

    reloaded = build_maya_model(cfg, tok)
    load_checkpoint_maya(last_ckpt, reloaded, tokenizer=tok)

    from torch.utils.data import DataLoader
    from maya250m.data import collate_stream_packed

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_stream_packed
    )
    test_metrics = evaluate_maya(reloaded, test_loader, mtp_weight=0.2)

    vexx = Vexx35Controller()
    prompt = "def solve(x: int) -> int:\n"
    depth_explain = None
    if args.adaptive_depth:
        gen, _loops, depth_explain = generate_adaptive(
            reloaded,
            tok,
            prompt=prompt,
            config=SolveConfig(
                max_new_tokens=args.max_new_tokens,
                max_loops=max(1, min(4, args.loops)),
                temperature=0.0,
                mode=PYTHON,
            ),
            vexx=vexx,
        )
    else:
        gen = generate_maya(
            reloaded,
            tok,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            vexx=vexx,
            mode=PYTHON,
        )

    summary = {
        "preset": args.preset,
        "parameters": sum(p.numel() for p in reloaded.parameters()),
        "effective_layers": reloaded.effective_layer_count(),
        "steps_completed": state["step"],
        "tokens_seen": reloaded.tokens_seen,
        "val_metrics": state["val_metrics"],
        "test_metrics": test_metrics,
        "sample_generation_provenance": gen.provenance,
        "generation_loops": gen.num_loops,
        "adaptive_depth": bool(args.adaptive_depth),
        "depth_explain": depth_explain,
        "manifest": manifest,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
