"""The single shared-trunk memory projection ``W_mem`` (``d -> 128``, bias-free).

Both memory queries and memory documents are projected by the *same* bias-free
linear map, so a document encoding is directly comparable to a query encoding with
one dot product. At ``d = 768`` this head costs exactly ``768 * 128 = 98,304``
parameters, which is the only trainable auxiliary added by the memory subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "MEMORY_DIM",
    "memory_projection_parameters",
    "MemoryProjection",
    "build_shared_trunk_memory_projection",
]

MEMORY_DIM = 128


def memory_projection_parameters(d: int, dim: int = MEMORY_DIM) -> int:
    """Exact parameter count of the bias-free ``d -> dim`` projection."""
    return int(d) * int(dim)


@dataclass
class MemoryProjection:
    """NumPy implementation of ``W_mem`` for CPU-only retrieval workers."""

    d: int = 768
    dim: int = MEMORY_DIM
    weight: Optional[np.ndarray] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.weight is None:
            rng = np.random.default_rng(self.seed)
            scale = 1.0 / float(np.sqrt(self.d))
            # Xavier-style scale keeps dot products O(1) at initialisation.
            self.weight = rng.standard_normal((self.dim, self.d)).astype(np.float32) * scale
        self.weight = np.asarray(self.weight, dtype=np.float32)
        if self.weight.shape != (self.dim, self.d):
            raise ValueError(
                f"weight shape {self.weight.shape} != expected {(self.dim, self.d)}"
            )

    @property
    def n_parameters(self) -> int:
        return memory_projection_parameters(self.d, self.dim)

    def project(self, x: np.ndarray, l2_normalize: bool = True) -> np.ndarray:
        """Project ``[..., d]`` activations onto the L2-normalised memory space."""
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != self.d:
            raise ValueError(f"expected last dim {self.d}, got {x.shape[-1]}")
        projected = x @ self.weight.T
        if l2_normalize:
            norm = np.linalg.norm(projected, axis=-1, keepdims=True)
            projected = projected / np.maximum(norm, 1e-8)
        return projected


def build_shared_trunk_memory_projection(d: int = 768, dim: int = MEMORY_DIM):
    """Build the torch ``nn.Linear(d, dim, bias=False)`` version of ``W_mem``.

    Imported lazily so CPU-only annotation workers never require PyTorch.
    """
    import torch.nn as nn

    return nn.Linear(int(d), int(dim), bias=False)
