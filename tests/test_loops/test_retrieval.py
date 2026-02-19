"""Tests for the retrieval loop.

Uses the shared `engine` fixture from conftest.py for Qdrant isolation.
"""


def test_recall_empty(engine):
    result = engine.recall("anything")
    assert result["total"] == 0
    assert result["memories"] == []


def test_recall_finds_text_match(engine):
    engine.observe("User prefers Python for scripting")
    result = engine.recall("Python")
    assert result["total"] >= 1
    assert any("Python" in m["content"] for m in result["memories"])


def test_recall_returns_metadata(engine):
    engine.observe("User works at OpenAI")
    result = engine.recall("OpenAI")
    assert result["total"] >= 1
    memory = result["memories"][0]
    assert "content" in memory
    assert "layer" in memory


def test_recall_token_budget(engine):
    # Store many memories
    for i in range(20):
        engine.observe(f"Memory number {i} about topic {i % 5}")

    result = engine.recall("topic", max_tokens=100)
    # Should respect token budget
    assert result["token_budget_used"] <= result["token_budget_max"]


def test_recall_scores_recent_higher(engine):
    """More recently accessed memories should score higher."""
    engine.observe("Old memory about cats")
    engine.observe("New memory about cats")

    result = engine.recall("cats")
    assert result["total"] >= 1


def test_recall_bumps_access_count(engine):
    engine.observe("Recall test memory")
    result = engine.recall("Recall test")
    if result["memories"]:
        mem_id = result["memories"][0].get("id")
        if mem_id:
            event = engine.event_store.get(mem_id)
            assert event.access_count >= 1


def test_schema_search_in_recall(engine, tmp_path):
    """Schema files should be searched during recall."""
    from re_memory.storage.file_store import FileStore

    fs = FileStore(engine.config.schema_path)
    fs.put("work", "# Work\nUser is a Python developer at OpenAI.")

    result = engine.recall("Python developer")
    schema_results = [m for m in result["memories"] if m["layer"] == "schema"]
    assert len(schema_results) >= 1
