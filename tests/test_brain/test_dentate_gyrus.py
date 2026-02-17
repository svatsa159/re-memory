"""Tests for Dentate Gyrus pattern separation."""

from re_memory.brain.dentate_gyrus import pattern_separate, batch_pattern_separate


def test_output_size():
    embedding = [0.1] * 768
    result = pattern_separate(embedding, sparse_dim=2048, sparsity_k=64)
    assert len(result) == 64


def test_deterministic():
    embedding = [0.5] * 768
    r1 = pattern_separate(embedding, sparse_dim=2048, sparsity_k=64, seed=42)
    r2 = pattern_separate(embedding, sparse_dim=2048, sparsity_k=64, seed=42)
    assert r1 == r2


def test_different_inputs_different_codes():
    emb1 = [0.1] * 768
    emb2 = [0.9] * 768
    r1 = pattern_separate(emb1)
    r2 = pattern_separate(emb2)
    assert r1 != r2


def test_indices_in_bounds():
    embedding = [0.3] * 768
    result = pattern_separate(embedding, sparse_dim=2048)
    for idx in result:
        assert 0 <= idx < 2048


def test_batch():
    embeddings = [[0.1] * 768, [0.9] * 768]
    results = batch_pattern_separate(embeddings)
    assert len(results) == 2
    assert results[0] != results[1]
