"""Deduplicated splitting, in-memory stream packing, and zero-copy Kaggle memmap shards.

Works with or without PyTorch installed so CPU shard-packing workers can pack
``uint16`` binary shards with zero deep-learning framework overhead.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import List, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _BaseDataset
except ImportError:  # pragma: no cover - lightweight CPU shard-packer fallback
    torch = None  # type: ignore[assignment]
    _BaseDataset = object  # type: ignore[misc,assignment]

from .tokenizer import BOS, PAD, BPETokenizer


def deduplicate_and_split(
    texts: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], dict]:
    """SHA-256 deduplicate stripped documents before deterministic splitting."""
    if not (0.0 < train_frac < 1.0) or not (0.0 <= val_frac < 1.0):
        raise ValueError("train_frac must be in (0,1) and val_frac in [0,1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave a non-empty test split")

    seen = set()
    unique_texts: List[str] = []
    for t in texts:
        h = hashlib.sha256(t.strip().encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_texts.append(t)

    rng = random.Random(seed)
    order = list(range(len(unique_texts)))
    rng.shuffle(order)
    n = len(order)
    n_train = max(1, int(n * train_frac))
    n_val = int(n * val_frac)
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    if n_train >= n:
        n_train = max(1, n - 1)

    train = [unique_texts[i] for i in order[:n_train]]
    val = [unique_texts[i] for i in order[n_train : n_train + n_val]]
    test = [unique_texts[i] for i in order[n_train + n_val :]]

    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise RuntimeError("Cross-split data leakage detected")

    manifest = {
        "raw_count": len(texts),
        "unique_count": len(unique_texts),
        "dropped_duplicates": len(texts) - len(unique_texts),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "seed": seed,
        "dedup": "sha256(strip(text))",
        "corpus_sha256": hashlib.sha256(
            "\n".join(sorted(text.strip() for text in unique_texts)).encode("utf-8")
        ).hexdigest(),
    }
    return train, val, test, manifest


class StreamPackedDataset(_BaseDataset):
    """In-memory stream-packed dataset with document-ID isolation and manifest."""

    def __init__(self, tokenizer: BPETokenizer, texts: List[str], seq_len: int, seed: int = 42):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.seq_len = seq_len
        self.examples: List[Tuple[List[int], List[int], List[int]]] = []

        rng = random.Random(seed)
        order = list(range(len(texts)))
        rng.shuffle(order)

        pad_id = tokenizer.token_to_id[PAD]
        stream_ids: List[int] = []
        stream_lbls: List[int] = []
        stream_docs: List[int] = []

        for doc_idx, di in enumerate(order):
            ids = tokenizer.encode(texts[di], add_bos=True, add_eos=True)
            if len(ids) < 2:
                continue
            lbls = list(ids)
            lbls[0] = -100
            stream_ids.extend(ids)
            stream_lbls.extend(lbls)
            stream_docs.extend([doc_idx] * len(ids))

        pos = 0
        while pos < len(stream_ids):
            chunk_ids = stream_ids[pos : pos + seq_len]
            chunk_lbls = stream_lbls[pos : pos + seq_len]
            chunk_docs = stream_docs[pos : pos + seq_len]
            pos += seq_len
            if len(chunk_ids) < 2:
                break
            pad_len = seq_len - len(chunk_ids)
            if pad_len > 0:
                chunk_ids = chunk_ids + [pad_id] * pad_len
                chunk_lbls = chunk_lbls + [-100] * pad_len
                chunk_docs = chunk_docs + [-1] * pad_len
            self.examples.append((chunk_ids, chunk_lbls, chunk_docs))

        digest = hashlib.sha256()
        for ids, lbls, docs in self.examples:
            digest.update(json.dumps([ids, lbls, docs]).encode("utf-8"))
        self.dataset_manifest = {
            "type": "StreamPackedDataset",
            "seq_len": self.seq_len,
            "num_examples": len(self.examples),
            "tokenizer_hash": tokenizer.tokenizer_hash(),
            "content_sha256": digest.hexdigest(),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ids, lbls, docs = self.examples[idx]
        if torch is not None:
            return (
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(lbls, dtype=torch.long),
                torch.tensor(docs, dtype=torch.long),
            )
        return (
            np.asarray(ids, dtype=np.int64),
            np.asarray(lbls, dtype=np.int64),
            np.asarray(docs, dtype=np.int64),
        )


def write_memmap_shard(
    tokenizer: BPETokenizer,
    texts: List[str],
    shard_prefix: str,
    seq_len: int,
    seed: int = 42,
) -> dict:
    """Pack ``texts`` into disk-backed uint16/int32 memmap files for Kaggle training."""
    packed = StreamPackedDataset(tokenizer, texts, seq_len=seq_len, seed=seed)
    n_ex = len(packed)
    if n_ex == 0:
        raise ValueError("Cannot write empty memmap shard")

    os.makedirs(os.path.dirname(os.path.abspath(shard_prefix)), exist_ok=True)
    tok_path = f"{shard_prefix}.tokens.bin"
    doc_path = f"{shard_prefix}.docs.bin"
    meta_path = f"{shard_prefix}.meta.json"

    tokens_arr = np.empty((n_ex, seq_len), dtype=np.uint16)
    docs_arr = np.empty((n_ex, seq_len), dtype=np.int32)
    for i, (ids, _lbls, docs) in enumerate(packed.examples):
        tokens_arr[i] = np.asarray(ids, dtype=np.uint16)
        docs_arr[i] = np.asarray(docs, dtype=np.int32)

    tokens_arr.tofile(tok_path)
    docs_arr.tofile(doc_path)

    bos_id = tokenizer.token_to_id[BOS]
    meta = {
        "type": "MemmapShardDataset",
        "num_examples": n_ex,
        "seq_len": seq_len,
        "bos_id": bos_id,
        "tokenizer_hash": tokenizer.tokenizer_hash(),
        "content_sha256": packed.dataset_manifest["content_sha256"],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


class MemmapShardDataset(_BaseDataset):
    """Zero-copy memory-mapped dataset for streaming 18 GB Kaggle token shards."""

    def __init__(self, shard_prefix: str):
        meta_path = f"{shard_prefix}.meta.json"
        tok_path = f"{shard_prefix}.tokens.bin"
        doc_path = f"{shard_prefix}.docs.bin"
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.num_examples = int(self.meta["num_examples"])
        self.seq_len = int(self.meta["seq_len"])
        self.bos_id = int(self.meta["bos_id"])
        self.tokens_mmap = np.memmap(
            tok_path, dtype=np.uint16, mode="r", shape=(self.num_examples, self.seq_len)
        )
        self.docs_mmap = np.memmap(
            doc_path, dtype=np.int32, mode="r", shape=(self.num_examples, self.seq_len)
        )
        self.dataset_manifest = dict(self.meta)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int):
        ids_np = np.array(self.tokens_mmap[idx], dtype=np.int64)
        docs_np = np.array(self.docs_mmap[idx], dtype=np.int64)
        lbls_np = ids_np.copy()
        lbls_np[docs_np < 0] = -100
        lbls_np[ids_np == self.bos_id] = -100
        if self.seq_len > 1:
            doc_change = np.zeros(self.seq_len, dtype=bool)
            doc_change[0] = True
            doc_change[1:] = docs_np[1:] != docs_np[:-1]
            lbls_np[doc_change] = -100
        if torch is not None:
            return (
                torch.from_numpy(ids_np),
                torch.from_numpy(lbls_np),
                torch.from_numpy(docs_np),
            )
        return ids_np, lbls_np, docs_np


def collate_stream_packed(batch):
    if torch is not None:
        ids = torch.stack([b[0] for b in batch], dim=0)
        lbls = torch.stack([b[1] for b in batch], dim=0)
        docs = torch.stack([b[2] for b in batch], dim=0)
        return ids, lbls, docs
    return (
        np.stack([b[0] for b in batch], axis=0),
        np.stack([b[1] for b in batch], axis=0),
        np.stack([b[2] for b in batch], axis=0),
    )
