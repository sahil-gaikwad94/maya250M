"""Run the five domain batteries against a Maya checkpoint and report honestly.

Two columns, never merged: ``model_only`` is the network's own continuation, and
``system_symbolic`` adds Vexx's exact arithmetic derivation for tasks whose prompt
states an expression. The derive step sees the prompt, never the answer key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from maya250m import (
    BPETokenizer,
    PYTHON,
    StreamPackedDataset,
    build_maya_model,
    deduplicate_and_split,
    evaluate_maya,
    generate_maya,
    get_config,
    load_checkpoint_maya,
    train_maya,
)
from maya250m.data import collate_stream_packed
from maya250m.evaluation import (
    SymbolicSolver,
    completion_diversity,
    domain_tasks,
    expressions_by_prompt,
    run_tasks,
    symbolic_repair_generate,
)
from run_maya250m import resolve_corpus


def build_raw_generator(model, tokenizer, max_new_tokens: int):
    """Model-only generator: no Vexx controller, no repair."""

    def generate(prompt: str) -> str:
        result = generate_maya(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=0.0,
            mode=PYTHON,
        )
        return result.full_text

    return generate


def main() -> int:
    parser = argparse.ArgumentParser(description="Maya five-domain evaluation")
    parser.add_argument("--preset", type=str, default="MAYA_TINY_TEST")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/maya_eval_ckpt")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--allow-synthetic-at-scale", action="store_true")
    parser.add_argument("--domains", type=str, default="")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(PROJECT_ROOT, "artifacts", "eval_domains.json"),
    )
    args = parser.parse_args()

    torch.manual_seed(7)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    try:
        corpus_texts, provenance = resolve_corpus(
            args.corpus, args.preset, args.allow_synthetic_at_scale
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"data gate: {exc}")

    train_t, val_t, test_t, manifest = deduplicate_and_split(
        corpus_texts, train_frac=0.75, val_frac=0.125, seed=7
    )
    if not train_t or not val_t:
        raise SystemExit(
            f"data gate: corpus too small after split (train={len(train_t)}, val={len(val_t)})"
        )

    tokenizer = BPETokenizer.train(train_t, vocab_size=384, min_pair_freq=2)
    tokenizer.validate()

    cfg = get_config(args.preset)
    cfg["vocab"] = tokenizer.vocab_size
    cfg["recurrent_loops"] = max(1, int(args.loops))
    model = build_maya_model(cfg, tokenizer)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint = args.load or os.path.join(args.checkpoint_dir, "maya_eval_last.pt")
    training = {"steps": 0}

    if args.load and os.path.exists(args.load):
        load_checkpoint_maya(args.load, model, tokenizer=tokenizer)
        training["loaded_from"] = os.path.abspath(args.load)
    else:
        train_ds = StreamPackedDataset(tokenizer, train_t, seq_len=args.seq_len, seed=7)
        val_ds = StreamPackedDataset(tokenizer, val_t, seq_len=args.seq_len, seed=8)
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
            checkpoint_path=checkpoint,
            tokenizer=tokenizer,
            seed=7,
        )
        training = {
            "steps": int(state["step"]),
            "val_metrics": state["val_metrics"],
            "checkpoint": checkpoint,
        }

    test_ds = StreamPackedDataset(tokenizer, test_t, seq_len=args.seq_len, seed=9)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_stream_packed,
    )
    held_out = evaluate_maya(model, test_loader, mtp_weight=0.2)

    selected = [name.strip() for name in args.domains.split(",") if name.strip()]
    batteries = domain_tasks()
    if selected:
        unknown = [name for name in selected if name not in batteries]
        if unknown:
            raise SystemExit(f"unknown domain(s) {unknown}; valid: {sorted(batteries)}")
        batteries = {name: batteries[name] for name in selected}

    raw_generate = build_raw_generator(model, tokenizer, args.max_new_tokens)
    payload = {
        "preset": args.preset,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "training": training,
        "data_provenance": provenance,
        "manifest": manifest,
        "held_out": held_out,
        "claim_scope": (
            "model_only scores come from the neural model's own continuation; "
            "system_symbolic adds exact arithmetic derived from the prompt expression. "
            "The two are never merged into one number."
        ),
        "domains": {},
    }

    print(f"preset={args.preset} steps={training['steps']} held_out_ce={held_out}")
    for domain, tasks in batteries.items():
        samples: list = []

        def _record(task, sample: str) -> None:
            # Diversity must be judged on the continuation only: the prompt is
            # different for every task and would make any output look varied.
            suffix = sample[len(task.prompt) :] if sample.startswith(task.prompt) else sample
            samples.append(suffix)

        model_report = run_tasks(
            tasks,
            generate=raw_generate,
            title=f"{domain}/model_only",
            k=1,
            on_sample=_record,
        )
        entry = {
            "model_only": model_report.to_dict(),
            "diagnostics": {"completion_diversity": completion_diversity(samples)},
        }
        model_pass = model_report.require_measured("pass@1")

        expressions = expressions_by_prompt(tasks)
        line = f"{domain:10} model_only pass@1 = {model_pass:.3f}"
        if expressions:
            solver = SymbolicSolver()
            system_report = run_tasks(
                tasks,
                generate=symbolic_repair_generate(raw_generate, expressions, solver),
                title=f"{domain}/system_symbolic",
                k=1,
            )
            entry["system_symbolic"] = system_report.to_dict()
            entry["symbolic"] = solver.to_dict()
            line += f" | system_symbolic pass@1 = {system_report.require_measured('pass@1'):.3f}"
        payload["domains"][domain] = entry
        diversity = entry["diagnostics"]["completion_diversity"]
        line += (
            f" | unique completions {diversity['unique']}/{diversity['samples']}"
        )
        print(line)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
