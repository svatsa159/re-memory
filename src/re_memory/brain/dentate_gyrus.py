"""Dentate Gyrus (DG): Pattern Separation

The dentate gyrus orthogonalizes similar inputs — even highly overlapping
experiences get distinct representations. This prevents catastrophic
interference in episodic memory.

Wraps the Rust implementation for performance.
"""

from __future__ import annotations


def pattern_separate(
    embedding: list[float],
    sparse_dim: int = 2048,
    sparsity_k: int = 64,
    seed: int = 42,
) -> list[int]:
    """Generate a sparse binary code from a dense embedding.

    Uses random projection + winner-take-all activation (Rust core).
    Falls back to Python implementation if Rust extension unavailable.
    """
    try:
        from re_memory._core import pattern_separate as _rs_pattern_separate

        return _rs_pattern_separate(embedding, sparse_dim, sparsity_k, seed)
    except ImportError:
        return _py_pattern_separate(embedding, sparse_dim, sparsity_k, seed)


def batch_pattern_separate(
    embeddings: list[list[float]],
    sparse_dim: int = 2048,
    sparsity_k: int = 64,
    seed: int = 42,
) -> list[list[int]]:
    """Batch pattern separation."""
    try:
        from re_memory._core import batch_pattern_separate as _rs_batch

        return _rs_batch(embeddings, sparse_dim, sparsity_k, seed)
    except ImportError:
        return [_py_pattern_separate(e, sparse_dim, sparsity_k, seed) for e in embeddings]


def _py_pattern_separate(
    embedding: list[float],
    sparse_dim: int,
    sparsity_k: int,
    seed: int,
) -> list[int]:
    """Pure Python fallback for pattern separation."""
    import random

    rng = random.Random(seed)
    input_dim = len(embedding)
    activations = []

    for _j in range(sparse_dim):
        s = 0.0
        for i in range(input_dim):
            r = rng.randint(0, 5)
            if r == 0:
                s -= embedding[i]
            elif r == 5:
                s += embedding[i]
        activations.append(s)

    # Winner-take-all: top-k indices
    indexed = sorted(enumerate(activations), key=lambda x: -x[1])
    return [idx for idx, _ in indexed[:sparsity_k]]
