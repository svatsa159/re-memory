"""Tests for the SQLite event store."""

import tempfile
from pathlib import Path

import pytest

from re_memory.storage.event_store import EventStore, EpisodicEvent


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_events.db"
    s = EventStore(db_path)
    s.init()
    yield s
    s.close()


def test_init_creates_tables(store):
    assert store.count() == 0


def test_upsert_and_get(store):
    event = EpisodicEvent(
        text="User prefers Python",
        confidence=0.8,
        importance=0.7,
        source="test",
    )
    store.upsert(event)
    assert store.count() == 1

    retrieved = store.get(event.id)
    assert retrieved is not None
    assert retrieved.text == "User prefers Python"
    assert retrieved.confidence == 0.8


def test_delete(store):
    event = EpisodicEvent(text="temporary")
    store.upsert(event)
    assert store.count() == 1
    store.delete(event.id)
    assert store.count() == 0


def test_search_text(store):
    store.upsert(EpisodicEvent(text="I love Python programming"))
    store.upsert(EpisodicEvent(text="Rust is fast"))
    store.upsert(EpisodicEvent(text="Python is great for scripting"))

    results = store.search_text("Python")
    assert len(results) == 2


def test_recent(store):
    for i in range(5):
        store.upsert(EpisodicEvent(text=f"event {i}"))
    results = store.recent(limit=3)
    assert len(results) == 3


def test_update_confidence(store):
    event = EpisodicEvent(text="test", confidence=0.5)
    store.upsert(event)
    store.update_confidence(event.id, 0.9)
    updated = store.get(event.id)
    assert updated.confidence == 0.9


def test_mark_accessed(store):
    event = EpisodicEvent(text="test", access_count=0)
    store.upsert(event)
    store.mark_accessed(event.id)
    updated = store.get(event.id)
    assert updated.access_count == 1


def test_prune_below_confidence(store):
    store.upsert(EpisodicEvent(text="low", confidence=0.1))
    store.upsert(EpisodicEvent(text="high", confidence=0.9))
    pruned = store.prune_below_confidence(0.5)
    assert pruned == 1
    assert store.count() == 1


def test_unconsolidated(store):
    store.upsert(EpisodicEvent(text="new", consolidated=False))
    e = EpisodicEvent(text="old", consolidated=True)
    store.upsert(e)
    store.mark_consolidated(e.id)
    results = store.unconsolidated()
    assert len(results) == 1
    assert results[0].text == "new"


def test_export_import(store, tmp_path):
    event = EpisodicEvent(text="exportable", confidence=0.7)
    store.upsert(event)

    d = event.to_full_dict()
    reimported = EpisodicEvent.from_dict(d)
    assert reimported.text == "exportable"
    assert reimported.confidence == 0.7
