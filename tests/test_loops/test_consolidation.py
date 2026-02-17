"""Tests for the consolidation loop."""

import pytest
from datetime import datetime, timezone, timedelta

from re_memory.config import Config
from re_memory.engine import MemoryEngine
from re_memory.storage.event_store import EpisodicEvent


@pytest.fixture
def engine(tmp_path):
    config = Config(
        data_dir=str(tmp_path / ".re-memory"),
        storage={"sqlite_path": str(tmp_path / "events.db"), "schema_dir": str(tmp_path / "schemas")},
        consolidation={"decay_rate": 0.1, "confidence_threshold": 0.2},
    )
    eng = MemoryEngine(config=config)
    eng.init()
    yield eng
    eng.close()


def test_consolidate_empty(engine):
    result = engine.consolidate()
    assert result["replayed"] == 0
    assert result["pruned"] == 0


def test_consolidate_decays_old_memories(engine):
    """Old, unaccessed memories should lose confidence."""
    old_time = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
    event = EpisodicEvent(
        text="Old memory",
        confidence=0.5,
        importance=0.3,
        access_count=0,
        last_accessed_at=old_time,
    )
    engine.event_store.upsert(event)

    result = engine.consolidate()
    assert result["decayed"] >= 1

    updated = engine.event_store.get(event.id)
    if updated:
        assert updated.confidence < 0.5  # should have decayed


def test_consolidate_prunes_low_confidence(engine):
    """Memories below threshold should be pruned."""
    event = EpisodicEvent(text="Weak memory", confidence=0.1)
    engine.event_store.upsert(event)

    result = engine.consolidate()
    assert result["pruned"] >= 1
    assert engine.event_store.get(event.id) is None


def test_high_importance_decays_slower(engine):
    """Important memories should decay slower than unimportant ones."""
    old_time = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()

    important = EpisodicEvent(
        text="Important fact", confidence=0.5, importance=0.9,
        access_count=0, last_accessed_at=old_time,
    )
    trivial = EpisodicEvent(
        text="Trivial fact", confidence=0.5, importance=0.1,
        access_count=0, last_accessed_at=old_time,
    )
    engine.event_store.upsert(important)
    engine.event_store.upsert(trivial)

    engine.consolidate()

    imp_updated = engine.event_store.get(important.id)
    triv_updated = engine.event_store.get(trivial.id)

    # Important memory should have higher remaining confidence
    if imp_updated and triv_updated:
        assert imp_updated.confidence > triv_updated.confidence


def test_frequently_accessed_decays_slower(engine):
    """Frequently accessed memories should decay slower."""
    old_time = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()

    frequent = EpisodicEvent(
        text="Frequent fact", confidence=0.5, importance=0.5,
        access_count=50, last_accessed_at=old_time,
    )
    rare = EpisodicEvent(
        text="Rare fact", confidence=0.5, importance=0.5,
        access_count=1, last_accessed_at=old_time,
    )
    engine.event_store.upsert(frequent)
    engine.event_store.upsert(rare)

    engine.consolidate()

    freq_updated = engine.event_store.get(frequent.id)
    rare_updated = engine.event_store.get(rare.id)

    if freq_updated and rare_updated:
        assert freq_updated.confidence > rare_updated.confidence


def test_merge_deduplicates(engine):
    """Exact duplicate texts should be merged."""
    engine.event_store.upsert(EpisodicEvent(text="duplicate text", confidence=0.5))
    engine.event_store.upsert(EpisodicEvent(text="duplicate text", confidence=0.6))
    engine.event_store.upsert(EpisodicEvent(text="unique text", confidence=0.5))

    assert engine.event_store.count() == 3

    result = engine.consolidate()
    assert result["merged"] >= 1
    assert engine.event_store.count() <= 2


def test_consolidation_daemon_status():
    from re_memory.loops.consolidation import daemon_status

    status = daemon_status()
    assert status["running"] is False
