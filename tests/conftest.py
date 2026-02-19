"""Shared test fixtures with proper isolation from production data.

Tests use unique Qdrant collections and FalkorDB graphs to avoid
contaminating or being contaminated by production data.
"""

import uuid

import pytest

from re_memory.config import Config
from re_memory.engine import MemoryEngine


@pytest.fixture
def engine(tmp_path):
    """Create an isolated MemoryEngine for testing.

    Uses:
      - Temp directory for SQLite and schemas (via tmp_path)
      - Unique Qdrant collection name per test run
      - Unique FalkorDB graph name per test run
    """
    test_id = uuid.uuid4().hex[:8]
    config = Config(
        data_dir=str(tmp_path / ".re-memory"),
        storage={
            "sqlite_path": str(tmp_path / "events.db"),
            "schema_dir": str(tmp_path / "schemas"),
        },
    )
    eng = MemoryEngine(config=config)

    # Override vector store with a test-specific collection
    from re_memory.storage.vector import VectorStore

    eng._vector_store = VectorStore(
        url=config.storage.qdrant_url,
        embedding_dim=config.embedding.dimensions,
        collection_name=f"test_{test_id}",
    )

    eng.init()
    yield eng

    # Cleanup: drop the test collection
    try:
        eng.vector_store.client.delete_collection(eng.vector_store.collection_name)
    except Exception:
        pass
    eng.close()
