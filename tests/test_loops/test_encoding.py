"""Tests for the encoding loop."""

import tempfile
from pathlib import Path

import pytest

from re_memory.config import Config
from re_memory.engine import MemoryEngine


@pytest.fixture
def engine(tmp_path):
    config = Config(
        data_dir=str(tmp_path / ".re-memory"),
        storage={"sqlite_path": str(tmp_path / "events.db"), "schema_dir": str(tmp_path / "schemas")},
    )
    eng = MemoryEngine(config=config)
    eng.init()
    yield eng
    eng.close()


def test_observe_stores_event(engine):
    result = engine.observe("User prefers Python for scripting")
    assert result["status"] == "encoded"
    assert result["verdict"] == "novel"
    assert result["sparse_code_bits"] == 64
    assert "id" in result


def test_observe_different_texts_different_ids(engine):
    r1 = engine.observe("I love Python")
    r2 = engine.observe("I love Rust")
    assert r1["id"] != r2["id"]


def test_observe_stores_in_event_store(engine):
    result = engine.observe("Test memory for retrieval")
    event = engine.event_store.get(result["id"])
    assert event is not None
    assert event.text == "Test memory for retrieval"
    assert event.source == "cli"
    assert len(event.sparse_code) == 64


def test_observe_with_source(engine):
    result = engine.observe("Agent observation", source="agent-1")
    event = engine.event_store.get(result["id"])
    assert event.source == "agent-1"


def test_observe_with_metadata(engine):
    result = engine.observe("Important fact", metadata={"priority": "high"})
    event = engine.event_store.get(result["id"])
    assert event.metadata.get("priority") == "high"


def test_status_after_observe(engine):
    engine.observe("Memory one")
    engine.observe("Memory two")
    status = engine.status()
    assert status["counts"]["episodic_events"] == 2


def test_search_finds_stored_memory(engine):
    engine.observe("Python is the best language for data science")
    results = engine.search("Python")
    assert len(results) == 1
    assert "Python" in results[0]["text"]
