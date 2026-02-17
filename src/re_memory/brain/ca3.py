"""CA3: Pattern Completion (Attractor-based Retrieval)

CA3 implements Modern Hopfield Network dynamics — given a partial or noisy
cue, it converges to the closest stored memory pattern. Combined with
knowledge graph traversal for multi-hop retrieval.

Wraps the Rust implementation for performance.
"""

from __future__ import annotations


def pattern_complete(
    query: list[float],
    stored_patterns: list[list[float]],
    beta: float = 8.0,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Find best matching stored patterns using Modern Hopfield retrieval.

    Args:
        query: Query embedding vector
        stored_patterns: List of stored memory embeddings
        beta: Inverse temperature (higher = sharper retrieval)
        top_k: Number of top matches to return

    Returns:
        List of (index, probability) tuples
    """
    try:
        from re_memory._core import pattern_complete as _rs_pattern_complete

        return _rs_pattern_complete(query, stored_patterns, beta, top_k)
    except ImportError:
        return _py_pattern_complete(query, stored_patterns, beta, top_k)


def _py_pattern_complete(
    query: list[float],
    stored_patterns: list[list[float]],
    beta: float,
    top_k: int,
) -> list[tuple[int, float]]:
    """Pure Python fallback for pattern completion."""
    import math

    if not stored_patterns:
        return []

    def l2_norm(v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    query_norm = l2_norm(query)
    if query_norm < 1e-10:
        return []

    energies = []
    for pattern in stored_patterns:
        pnorm = l2_norm(pattern)
        if pnorm < 1e-10:
            energies.append(float("-inf"))
            continue
        dot = sum(a * b for a, b in zip(query, pattern))
        energies.append(beta * dot / (query_norm * pnorm))

    max_e = max(energies)
    exp_energies = [math.exp(e - max_e) for e in energies]
    sum_exp = sum(exp_energies)
    probs = [e / sum_exp for e in exp_energies] if sum_exp > 0 else [0.0] * len(energies)

    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    return indexed[:top_k]
