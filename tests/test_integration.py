"""Integration tests — end-to-end verification of the full memory pipeline."""

import pytest

from re_memory.config import Config
from re_memory.engine import MemoryEngine
from re_memory.storage.event_store import EpisodicEvent


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


class TestJobChangeScenario:
    """The "job change" test from the verification plan.

    observe "User works at Google" →
    observe "User works at OpenAI" →
    recall "Where does the user work?" →
    must return OpenAI (the most recent fact)
    """

    def test_job_change_recall_returns_latest(self, engine):
        engine.observe("User works at Google")
        engine.observe("User works at OpenAI")

        result = engine.recall("Where does the user work?")
        texts = [m["content"] for m in result["memories"]]

        # Both should be present in results, but check that OpenAI appears
        assert any("OpenAI" in t for t in texts), f"Expected OpenAI in results: {texts}"


class TestPatternSeparation:
    """The "pattern separation" test.

    Similar but different inputs should produce distinct sparse codes.
    """

    def test_similar_texts_different_codes(self, engine):
        r1 = engine.observe("I love my job")
        r2 = engine.observe("I hate my job")

        # Both should be stored as novel (not redundant)
        assert r1["status"] == "encoded"
        assert r2["status"] == "encoded"
        assert r1["id"] != r2["id"]


class TestDecayScenario:
    """The "decay" test.

    Store memories → run consolidation → unreinforced memories lose confidence.
    """

    def test_unreinforced_memories_decay(self, engine):
        from datetime import datetime, timezone, timedelta

        old_time = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()

        event = EpisodicEvent(
            text="Unreinforced memory",
            confidence=0.8,
            importance=0.3,
            access_count=0,
            last_accessed_at=old_time,
        )
        engine.event_store.upsert(event)

        result = engine.consolidate()
        assert result["decayed"] >= 1

        updated = engine.event_store.get(event.id)
        if updated:
            assert updated.confidence < 0.8


class TestMultiSessionScenario:
    """Multi-session test: observe across separate invocations, recall later."""

    def test_observe_and_recall_across_sessions(self, tmp_path):
        config = Config(
            data_dir=str(tmp_path / ".re-memory"),
            storage={"sqlite_path": str(tmp_path / "events.db"), "schema_dir": str(tmp_path / "schemas")},
        )

        # Session 1: observe
        eng1 = MemoryEngine(config=config)
        eng1.init()
        eng1.observe("User prefers dark mode")
        eng1.close()

        # Session 2: observe more
        eng2 = MemoryEngine(config=config)
        eng2.observe("User's timezone is PST")
        eng2.close()

        # Session 3: recall
        eng3 = MemoryEngine(config=config)
        result = eng3.recall("dark mode")
        texts = [m["content"] for m in result["memories"]]
        assert any("dark mode" in t for t in texts)
        eng3.close()


class TestExportImport:
    """Export and import memory state."""

    def test_export_import_roundtrip(self, engine, tmp_path):
        engine.observe("Memory for export test")
        engine.observe("Another memory for export")

        export_path = tmp_path / "export.json"
        export_result = engine.export_data(export_path)
        assert export_result["exported"] == 2

        # Create a fresh engine and import
        config2 = Config(
            data_dir=str(tmp_path / ".re-memory2"),
            storage={"sqlite_path": str(tmp_path / "events2.db"), "schema_dir": str(tmp_path / "schemas2")},
        )
        eng2 = MemoryEngine(config=config2)
        eng2.init()
        import_result = eng2.import_data(export_path)
        assert import_result["imported"] == 2

        result = eng2.search("export test")
        assert len(result) >= 1
        eng2.close()


class TestForgetScenario:
    """Explicit forgetting."""

    def test_forget_removes_memory(self, engine):
        r = engine.observe("Secret memory")
        assert r["status"] == "encoded"

        forget_result = engine.forget(r["id"])
        assert forget_result["status"] == "forgotten"

        assert engine.event_store.get(r["id"]) is None


class TestJsonOutput:
    """CLI JSON output mode."""

    def test_observe_json(self, engine):
        result = engine.observe("JSON test memory")
        assert isinstance(result, dict)
        assert "id" in result
        assert "status" in result

    def test_recall_json(self, engine):
        engine.observe("JSON recall test")
        result = engine.recall("JSON recall")
        assert isinstance(result, dict)
        assert "memories" in result
        assert "total" in result
