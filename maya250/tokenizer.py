"""Versioned byte-level BPE tokenizer for Maya-250M & Vexx 3.5.

Upgrades over standard BPE:
1. Byte-level completeness (all 256 UTF-8 bytes pre-registered) with zero ``<|unk|>`` fallbacks.
2. Word-frequency-grouped merge counting for fast corpus training.
3. Explicit protocol markers for Vexx 3.5:
   - ``<think>`` / ``</think>`` for CRANE unconstrained latent/CoT reasoning.
   - ``<experiment>`` / ``</experiment>`` for ground-up scientific hypothesis
     testing and live sandbox execution splicing.
4. ``encode_to_uint16`` helper for zero-copy Kaggle CPU binary shard packing.
5. Full cryptographic identity hash (``tokenizer_hash``) verified on checkpoint load.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

BOS = "<|bos|>"
EOS = "<|eos|>"
PAD = "<|pad|>"
UNK = "<|unk|>"
THINK_START = "<think>"
THINK_END = "</think>"
EXP_START = "<experiment>"
EXP_END = "</experiment>"

SPECIAL_TOKENS = [BOS, EOS, PAD, UNK, THINK_START, THINK_END, EXP_START, EXP_END]
PROTOCOL_TOKENS_Tuple = (THINK_START, THINK_END, EXP_START, EXP_END)

NEVER_EMIT = "never_emit"
LEGAL_STOP = "legal_stop"
PROTOCOL_MARKERS = "protocol_markers"

_TOKEN_CATEGORY = {
    BOS: NEVER_EMIT,
    PAD: NEVER_EMIT,
    UNK: NEVER_EMIT,
    EOS: LEGAL_STOP,
    THINK_START: PROTOCOL_MARKERS,
    THINK_END: PROTOCOL_MARKERS,
    EXP_START: PROTOCOL_MARKERS,
    EXP_END: PROTOCOL_MARKERS,
}


def token_category(token_str: str) -> Optional[str]:
    """Classify a token string for the Vexx 3.5 generation protocol."""
    return _TOKEN_CATEGORY.get(token_str)


# Pre-tokenizer splits digits individually (`[0-9]`) so arithmetic and SymPy
# expressions have consistent digit granularity across scales.
_PAT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z_][A-Za-z0-9_]*| ?[0-9]| ?[^\sA-Za-z0-9_]+|\s+(?!\S)|\s+"
)


class BPETokenizer:
    """Byte-level BPE with Vexx 3.5 protocol markers and a deterministic SHA-256 hash."""

    VERSION = 9

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.special_tokens = list(special_tokens or SPECIAL_TOKENS)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_rank: Dict[Tuple[int, int], int] = {}

        for st in self.special_tokens:
            self._add_token(st)
        if vocab is not None:
            for token, tid in vocab.items():
                self.token_to_id[token] = int(tid)
                self.id_to_token[int(tid)] = token
        self._ensure_byte_tokens()
        if merges is not None:
            self.merges = [tuple(m) for m in merges]  # type: ignore[misc]
            for rank, (a, b) in enumerate(self.merges):
                la = self.token_to_id.get(a)
                rb = self.token_to_id.get(b)
                if la is not None and rb is not None:
                    self.merge_rank[(la, rb)] = rank
        self.vocab_size = len(self.token_to_id)

    def _add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        tid = len(self.token_to_id)
        self.token_to_id[token] = tid
        self.id_to_token[tid] = token
        return tid

    def _ensure_byte_tokens(self) -> None:
        for i in range(256):
            ch = chr(i)
            if ch not in self.token_to_id:
                self._add_token(ch)

    def tokenizer_hash(self) -> str:
        """Canonical SHA-256 hash of the complete tokenizer identity."""
        payload = {
            "version": self.VERSION,
            "special_tokens": self.special_tokens,
            "merges": [[a, b] for a, b in self.merges],
            "vocab": [
                [int(tid), tok]
                for tok, tid in sorted(self.token_to_id.items(), key=lambda kv: kv[1])
            ],
            "vocab_size": self.vocab_size,
            "pretokenizer": _PAT.pattern,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: Optional[List[str]] = None,
        min_pair_freq: int = 2,
    ) -> "BPETokenizer":
        """Train BPE merges on the training corpus only."""
        tok = cls(special_tokens=special_tokens)
        word_freqs: Counter = Counter()
        for text in texts:
            for pre in _PAT.findall(text):
                w = tuple(tok.token_to_id[chr(b)] for b in pre.encode("utf-8"))
                word_freqs[w] += 1

        specials = set(tok.special_tokens)
        while len(tok.token_to_id) < vocab_size:
            pair_counts: Counter = Counter()
            for w, freq in word_freqs.items():
                for i in range(len(w) - 1):
                    pair_counts[(w[i], w[i + 1])] += freq
            candidates = [(c, p) for p, c in pair_counts.items() if c >= min_pair_freq]
            if not candidates:
                break
            safe = [
                cp
                for cp in candidates
                if tok.id_to_token[cp[1][0]] + tok.id_to_token[cp[1][1]] not in specials
            ]
            if not safe:
                break
            _, (a, b) = max(safe, key=lambda cp: (cp[0], -cp[1][0], -cp[1][1]))
            left_tok, right_tok = tok.id_to_token[a], tok.id_to_token[b]
            merged_id = tok._add_token(left_tok + right_tok)
            tok.merges.append((left_tok, right_tok))
            tok.merge_rank[(a, b)] = len(tok.merges) - 1

            new_word_freqs: Counter = Counter()
            for w, freq in word_freqs.items():
                out: List[int] = []
                i = 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                        out.append(merged_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                new_word_freqs[tuple(out)] += freq
            word_freqs = new_word_freqs

        tok.vocab_size = len(tok.token_to_id)
        return tok

    def _encode_bytes(self, byte_ids: List[int]) -> List[int]:
        seq = list(byte_ids)
        while len(seq) > 1:
            best_rank = None
            best_i = -1
            for i in range(len(seq) - 1):
                r = self.merge_rank.get((seq[i], seq[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                break
            merged_tok = self.id_to_token[seq[best_i]] + self.id_to_token[seq[best_i + 1]]
            seq = seq[:best_i] + [self.token_to_id[merged_tok]] + seq[best_i + 2 :]
        return seq

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text into token IDs while preserving protocol markers as atomic tokens."""
        ids: List[int] = []
        if add_bos:
            ids.append(self.token_to_id[BOS])

        active_markers = [m for m in PROTOCOL_TOKENS_Tuple if m in self.token_to_id]
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for marker in active_markers:
                if text.startswith(marker, i):
                    ids.append(self.token_to_id[marker])
                    i += len(marker)
                    matched = True
                    break
            if matched:
                continue

            next_marker_pos = n
            for marker in active_markers:
                pos = text.find(marker, i + 1)
                if pos != -1 and pos < next_marker_pos:
                    next_marker_pos = pos
            plain = text[i:next_marker_pos]
            for pre in _PAT.findall(plain):
                byte_ids = [self.token_to_id[chr(b)] for b in pre.encode("utf-8")]
                ids.extend(self._encode_bytes(byte_ids))
            i = next_marker_pos

        if add_eos:
            ids.append(self.token_to_id[EOS])
        return ids

    def encode_to_uint16(self, text: str, add_bos: bool = True, add_eos: bool = True) -> np.ndarray:
        """Encode text directly to a compact uint16 NumPy array for Kaggle memmap shards."""
        if self.vocab_size > 65535:
            raise ValueError(f"uint16 packing requires vocab_size <= 65535, got {self.vocab_size}")
        return np.asarray(self.encode(text, add_bos=add_bos, add_eos=add_eos), dtype=np.uint16)

    def decode(self, ids: List[int], preserve_protocol: bool = False) -> str:
        unknown = [int(tid) for tid in ids if int(tid) not in self.id_to_token]
        if unknown:
            raise ValueError(f"Cannot decode unknown token ids {unknown[:5]}")
        chunks = [self.id_to_token[int(tid)] for tid in ids]
        if preserve_protocol:
            chunks = [
                c
                for c in chunks
                if c not in self.special_tokens or c in PROTOCOL_TOKENS_Tuple
            ]
        else:
            chunks = [c for c in chunks if c not in self.special_tokens]
        raw = "".join(chunks)
        try:
            return raw.encode("latin-1").decode("utf-8")
        except UnicodeDecodeError:
            return raw

    def validate(self) -> None:
        """Fail-closed validation of all vocabulary and merge invariants."""
        ids = list(self.id_to_token.keys())
        if ids != list(range(len(ids))):
            raise ValueError("token IDs are not dense/contiguous from zero")
        if self.vocab_size != len(self.token_to_id):
            raise ValueError("vocab_size mismatch with token_to_id")
        for a, b in self.merges:
            if a not in self.token_to_id or b not in self.token_to_id or (a + b) not in self.token_to_id:
                raise ValueError(f"invalid merge ({a!r}, {b!r})")
        if len(self.special_tokens) != len(set(self.special_tokens)):
            raise ValueError("duplicate special tokens")

    def save(self, path: str) -> None:
        payload = {
            "version": self.VERSION,
            "special_tokens": self.special_tokens,
            "merges": [[a, b] for a, b in self.merges],
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "tokenizer_hash": self.tokenizer_hash(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(
            vocab=payload.get("token_to_id"),
            merges=[tuple(m) for m in payload.get("merges", [])],
            special_tokens=payload.get("special_tokens") or SPECIAL_TOKENS,
        )
        tok.vocab_size = payload.get("vocab_size", len(tok.token_to_id))
        claimed = payload.get("tokenizer_hash")
        if claimed is not None and claimed != tok.tokenizer_hash():
            raise ValueError("Tokenizer hash mismatch on load (fail-closed)")
        return tok
