"""Kaggle Free CPU Session Shard Packer + Vexx 3.5 CAS Annotator.

Runs entirely on free Kaggle CPU notebooks (0 GPU/TPU quota consumed):
1. Reads raw JSONL/text corpora (FineWeb-Edu, Stack-Edu, FineMath, Cosmopedia).
2. Evaluates inline arithmetic expressions via Vexx 3.5 Tier-2 ``safe_sympy`` and
   splices verified ``<<expr=val>>`` steps into ``<think>`` traces.
3. Deduplicates via SHA-256 and packs tokens into ``uint16`` binary memmap shards
   (``.tokens.bin`` + ``.docs.bin`` + ``.meta.json``) ready for zero-copy streaming.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maya250m import BPETokenizer, Vexx35Controller, deduplicate_and_split, write_memmap_shard


def annotate_cas_traces(texts: List[str], vexx: Vexx35Controller) -> List[str]:
    """Splice exact SymPy evaluations into ``<<expr=>>`` placeholders."""
    pattern = re.compile(r"<<([^<>=]+)=>>")
    annotated: List[str] = []
    for doc in texts:
        def _repl(match):
            expr = match.group(1).strip()
            val = vexx.crane_cas_evaluate(expr)
            if val.startswith("ERROR"):
                return match.group(0)
            return f"<<{expr}={val}>>"

        annotated.append(pattern.sub(_repl, doc))
    return annotated


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack JSONL/text into uint16 Kaggle memmap shards")
    parser.add_argument("--input-jsonl", type=str, default="")
    parser.add_argument("--output-prefix", type=str, default="/tmp/maya_shard")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=384)
    args = parser.parse_args()

    vexx = Vexx35Controller()
    if args.input_jsonl and os.path.exists(args.input_jsonl):
        raw_texts = []
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    raw_texts.append(str(row.get("text", "")))
    else:
        raw_texts = [
            f"<think>Compute exact rational value <<(5/6 - 1/4)/(7/12)=>> and sum {i}: <<{i}*12=>></think>\n"
            f"def f_{i}(x: int) -> int:\n    return x + {i}\n"
            for i in range(1, 80)
        ]

    enriched = annotate_cas_traces(raw_texts, vexx)
    train_t, val_t, _test_t, manifest = deduplicate_and_split(enriched, seed=42)
    tok = BPETokenizer.train(train_t, vocab_size=args.vocab_size)
    tok.validate()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)
    tok.save(f"{args.output_prefix}.tokenizer.json")
    train_meta = write_memmap_shard(tok, train_t, f"{args.output_prefix}_train", seq_len=args.seq_len)
    val_meta = write_memmap_shard(tok, val_t, f"{args.output_prefix}_val", seq_len=args.seq_len)

    print(
        json.dumps(
            {"manifest": manifest, "train_shard": train_meta, "val_shard": val_meta},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
