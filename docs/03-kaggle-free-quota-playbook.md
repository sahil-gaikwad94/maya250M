# 03 — Kaggle Free-Quota Training Playbook (`$0` Compute Budget)

## 1. Weekly Compute Budget on Kaggle Free Quotas

Kaggle provides two independent free weekly quotas ([Kaggle Notebooks Quota Guide](https://fastero.com/blog/kaggle-notebooks); [Kaggle TPU Docs](https://www.kaggle.com/docs/tpu)):

- **TPU Quota (~20 hours/week on `TPU v3-8` / `v5e-8`)**: 9-hour session limit (`8.25h` safe watchdog cutoff). Sustained throughput with Chunkwise Parallel GDN (`C=64`): `~106,000 tokens/sec` (`~7.6B tokens/week`).
- **Dual-GPU Quota (~30 hours/week on `Tesla T4 x2`, 32 GB VRAM)**: 12-hour session limit (`11.25h` safe watchdog cutoff). Sustained throughput with FP16 + DDP: `~36,400 tokens/sec` (`~3.9B tokens/week`).
- **Free CPU Notebooks (Unlimited Weekly Quota, 30 GB RAM)**: Used exclusively for downloading [FineWeb-Edu](https://huggingface.co/HuggingFaceTB), [Stack-Edu](https://huggingface.co/HuggingFaceTB), and [FineMath](https://huggingface.co/collections/HuggingFaceTB/finemath), running SHA-256 deduplication, annotating `<<expr=val>>` CAS traces, and packing `uint16` `.tokens.bin` memmap shards via `kaggle/pack_shards_cpu.py`.

## 2. Multi-Session Auto-Chaining Workflow

1. **Pack Shards on CPU**: Run `python3 kaggle/pack_shards_cpu.py --input-jsonl corpus.jsonl --output-prefix /kaggle/working/shard01 --seq-len 2048 --vocab-size 32000` and save the output as a Kaggle Dataset.
2. **Train with Auto-Watchdog**: Run `python3 kaggle/train_maya250m_kaggle.py --preset MAYA_250M --shard-prefix /kaggle/input/maya-shards/shard01_train --val-shard-prefix /kaggle/input/maya-shards/shard01_val --tokenizer-path /kaggle/input/maya-shards/shard01.tokenizer.json --ckpt-dir /kaggle/working/ckpt --kaggle-dataset-slug <username>/maya250m-ckpt`.
   The flagship path requires all three data arguments, a `vocab_size == 32000` tokenizer, and shard metadata whose `tokenizer_hash` matches it; it exits before model construction otherwise. Fixture runs must opt in explicitly with `--smoke-test-synthetic`, which writes `checkpoint_role: "smoke_test_only"` into the checkpoint metadata.
3. **Bitwise-Exact Resume**: At `8.25h` (TPU) or `11.25h` (Dual-T4), `train_maya` finishes the current gradient accumulation step, saves `maya250m_last.pt` (Schema 9 `weights_only=True` with the exact epoch permutation and `next_batch_cursor`), versions the Kaggle Dataset, and exits cleanly so the next session resumes with `max|Δw| = 0.0`.

## 3. Execution Scope & Single-Process Device Honesty

`kaggle/train_maya250m_kaggle.py` executes in **single-process device mode** (`xla_single` on a single XLA device via `xm.optimizer_step`, or `single_gpu` on `cuda:0` with `batch_size=4, grad_accum=16`). If `WORLD_SIZE > 1` is set in the environment without a multi-process distributed coordinator, `resolve_training_device()` fails closed with `NOT_RUN` rather than silently running on a single rank or faking multi-GPU DDP throughput.
