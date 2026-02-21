"""Tests for the encoding loop.

Uses the shared `engine` fixture from conftest.py for Qdrant isolation.
"""

import time

from re_memory.loops.encoding import _DEDUP_CACHE


def _clear_dedup_cache():
    """Clear the module-level dedup cache between tests."""
    _DEDUP_CACHE.clear()


def test_observe_stores_event(engine):
    _clear_dedup_cache()
    result = engine.observe("User prefers Python for scripting")
    assert result["status"] == "encoded"
    assert result["verdict"] == "novel"
    assert result["sparse_code_bits"] == 64
    assert "id" in result


def test_observe_different_texts_different_ids(engine):
    _clear_dedup_cache()
    r1 = engine.observe("The database migration completed successfully last night")
    r2 = engine.observe("User's favorite recipe is homemade pasta with basil")
    assert r1["status"] == "encoded"
    assert r2["status"] == "encoded"
    assert r1["id"] != r2["id"]


def test_observe_stores_in_event_store(engine):
    _clear_dedup_cache()
    result = engine.observe("Test memory for retrieval")
    event = engine.event_store.get(result["id"])
    assert event is not None
    assert event.text == "Test memory for retrieval"
    assert event.source == "cli"
    assert len(event.sparse_code) == 64


def test_observe_with_source(engine):
    _clear_dedup_cache()
    result = engine.observe("Agent observation", source="agent-1")
    event = engine.event_store.get(result["id"])
    assert event.source == "agent-1"


def test_observe_with_metadata(engine):
    _clear_dedup_cache()
    result = engine.observe("Important fact", metadata={"priority": "high"})
    event = engine.event_store.get(result["id"])
    assert event.metadata.get("priority") == "high"


def test_status_after_observe(engine):
    _clear_dedup_cache()
    engine.observe("User prefers dark mode for all applications")
    engine.observe("The quarterly earnings report exceeded analyst expectations")
    status = engine.status()
    assert status["counts"]["episodic_events"] == 2


def test_search_finds_stored_memory(engine):
    _clear_dedup_cache()
    engine.observe("Python is the best language for data science")
    results = engine.search("Python")
    assert len(results) == 1
    assert "Python" in results[0]["text"]


# --- Fast tier tests ---


def test_observe_fast_tier(engine):
    """Fast tier should encode without LLM calls."""
    _clear_dedup_cache()
    result = engine.observe("User likes tea", tier="fast")
    assert result["status"] == "encoded"
    assert result["verdict"] == "novel"
    assert result.get("tier") == "fast"
    assert result["sparse_code_bits"] == 64
    # Verify stored
    event = engine.event_store.get(result["id"])
    assert event is not None
    assert event.text == "User likes tea"


def test_observe_fast_tier_no_features(engine):
    """Fast tier should not extract features (no LLM)."""
    _clear_dedup_cache()
    result = engine.observe("User works at ACME Corp", tier="fast")
    event = engine.event_store.get(result["id"])
    # Fast tier doesn't extract topics/entities
    assert event.tags == []


# --- Dedup cache tests ---


def test_dedup_cache_returns_redundant(engine):
    """Second observe of identical text should return redundant from cache."""
    _clear_dedup_cache()
    r1 = engine.observe("User prefers vim keybindings")
    assert r1["status"] == "encoded"

    r2 = engine.observe("User prefers vim keybindings")
    assert r2["status"] == "redundant"
    assert r2["closest_memory"] == r1["id"]
    assert "dedup cache" in r2["explanation"].lower()


def test_dedup_cache_whitespace_invariant(engine):
    """Cache should normalize whitespace (strip)."""
    _clear_dedup_cache()
    r1 = engine.observe("User likes coffee")
    r2 = engine.observe("  User likes coffee  ")
    assert r2["status"] == "redundant"
    assert r2["closest_memory"] == r1["id"]


def test_dedup_cache_different_text_not_cached(engine):
    """Different texts should not hit cache."""
    _clear_dedup_cache()
    r1 = engine.observe("The user enjoys mountain biking on weekends")
    r2 = engine.observe("PostgreSQL 16 introduced merge joins for JSON columns")
    assert r1["status"] == "encoded"
    assert r2["status"] == "encoded"
    assert r1["id"] != r2["id"]


def test_dedup_cache_reinforces_on_hit(engine):
    """Cache hit should bump access_count and confidence."""
    _clear_dedup_cache()
    r1 = engine.observe("User prefers dark themes in all editors")
    event_before = engine.event_store.get(r1["id"])
    original_confidence = event_before.confidence
    original_access = event_before.access_count

    # Hit the cache
    engine.observe("User prefers dark themes in all editors")
    event_after = engine.event_store.get(r1["id"])
    assert event_after.access_count == original_access + 1
    assert event_after.confidence == min(1.0, original_confidence + 0.05)


# --- Background mode tests ---


def test_observe_background_returns_immediately(engine):
    """Background observe should return accepted status immediately."""
    _clear_dedup_cache()
    result = engine.observe("Background test fact", background=True)
    assert result["status"] == "accepted"
    assert result["background"] is True
    # Give background thread time to finish
    time.sleep(1)
    engine.close()


# --- Tier parameter tests ---


def test_observe_full_tier(engine):
    """Full tier should work (falls back to fallback embedding without Ollama)."""
    _clear_dedup_cache()
    result = engine.observe("User enjoys hiking on weekends", tier="full")
    assert result["status"] == "encoded"
    assert result["verdict"] == "novel"


def test_observe_standard_tier_default(engine):
    """Standard tier is the default."""
    _clear_dedup_cache()
    result = engine.observe("Default tier observation")
    assert result["status"] == "encoded"


# --- Batch observe tests ---


def test_observe_batch_multiple(engine):
    """Batch observe should encode multiple texts."""
    _clear_dedup_cache()
    texts = [
        "The user prefers Python for backend development",
        "Tokyo has a population of 14 million people",
        "The recipe calls for three cups of flour and two eggs",
    ]
    results = engine.observe_batch(texts)
    assert len(results) == 3
    for r in results:
        assert r["status"] == "encoded"
        assert r["verdict"] == "novel"


def test_observe_batch_empty(engine):
    """Batch with empty list should return empty."""
    _clear_dedup_cache()
    results = engine.observe_batch([])
    assert results == []


def test_observe_batch_dedup(engine):
    """Batch should dedup against previously seen text."""
    _clear_dedup_cache()
    engine.observe("The cat sleeps on the windowsill every afternoon")
    results = engine.observe_batch([
        "The cat sleeps on the windowsill every afternoon",
        "Mars has two moons named Phobos and Deimos",
    ])
    assert len(results) == 2
    statuses = [r["status"] for r in results]
    assert "redundant" in statuses
    assert "encoded" in statuses


def test_observe_batch_background(engine):
    """Batch background should return accepted."""
    _clear_dedup_cache()
    result = engine.observe_batch(
        ["Bg batch 1", "Bg batch 2"], background=True
    )
    assert result["status"] == "accepted"
    assert result["background"] is True
    assert result["count"] == 2
    time.sleep(1)
    engine.close()


def test_observe_batch_fast_tier(engine):
    """Batch with fast tier should work."""
    _clear_dedup_cache()
    texts = [
        "Kubernetes orchestrates containerized applications at scale",
        "The Renaissance began in Italy during the 14th century",
    ]
    results = engine.observe_batch(texts, tier="fast")
    assert len(results) == 2
    for r in results:
        assert r["status"] == "encoded"
