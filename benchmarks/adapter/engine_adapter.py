"""Adapter that wraps MemoryEngine with benchmark-friendly operations.

Maps generic benchmark operations (store, query, update, delete) to re-memory's
brain-anatomical API without modifying any existing source code.

Each benchmark runner uses this adapter rather than touching the engine directly,
so all benchmark-specific logic (isolation, metrics hooks, timeouts) lives here.
"""

from __future__ import annotations

import time
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from re_memory.config import Config, LLMConfig, EmbeddingConfig, StorageConfig, MemoryConfig, ConsolidationConfig
from re_memory.engine import MemoryEngine


@dataclass
class OperationRecord:
    """Single logged operation for post-hoc analysis."""

    operation: str
    input_text: str
    result: dict
    latency_ms: float
    timestamp: float


@dataclass
class EngineAdapter:
    """Benchmark-safe wrapper around MemoryEngine.

    Features:
    - Isolated storage (temp directories, separate SQLite/Qdrant collection/FalkorDB graph)
    - Operation logging with latency tracking
    - Convenience methods that map 1:1 with benchmark task types
    - Automatic cleanup on close
    """

    engine: MemoryEngine | None = None
    _temp_dir: Path | None = None
    _operation_log: list[OperationRecord] = field(default_factory=list)
    _collection_suffix: str = ""

    @classmethod
    def create_isolated(
        cls,
        run_id: str = "bench",
        tier: str = "standard",
        base_config: Config | None = None,
        qdrant_url: str = "http://localhost:6333",
        falkordb_host: str = "localhost",
        falkordb_port: int = 6379,
    ) -> "EngineAdapter":
        """Create an adapter with fully isolated storage.

        Uses a temp directory for SQLite + schemas, and a unique Qdrant collection
        name + FalkorDB graph name to avoid colliding with user data.

        Args:
            run_id: Unique identifier for this benchmark run (used in collection names).
            tier: Default observe tier for this run.
            base_config: Optional base config to inherit LLM/embedding settings from.
                         If None, loads from the project's re_memory.toml.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=f"re_memory_bench_{run_id}_"))

        if base_config is None:
            from re_memory.config import load_config
            base_config = load_config()

        # Build isolated config — same LLM/embedding, isolated storage
        config = Config(
            llm=base_config.llm.model_copy(),
            embedding=base_config.embedding.model_copy(),
            storage=StorageConfig(
                qdrant_url=qdrant_url,
                falkordb_host=falkordb_host,
                falkordb_port=falkordb_port,
                sqlite_path=str(temp_dir / "bench_events.db"),
                schema_dir=str(temp_dir / "schemas"),
            ),
            consolidation=base_config.consolidation.model_copy(),
            memory=MemoryConfig(
                working_memory_slots=base_config.memory.working_memory_slots,
                max_retrieval_tokens=base_config.memory.max_retrieval_tokens,
                novelty_threshold=base_config.memory.novelty_threshold,
                importance_threshold=base_config.memory.importance_threshold,
                observe_tier=tier,
            ),
            data_dir=str(temp_dir),
        )

        engine = MemoryEngine(config=config)
        engine.init()

        # Qdrant: use a unique collection name to isolate from user data
        collection_suffix = f"_bench_{run_id}"
        engine.vector_store.collection_name = f"re_memory_embeddings{collection_suffix}"
        engine.vector_store.init()

        # FalkorDB: use a unique graph name
        try:
            engine.graph_store.graph_name = f"re_memory_bench_{run_id}"
            engine.graph_store.init()
        except Exception:
            pass  # FalkorDB may not be available

        adapter = cls(
            engine=engine,
            _temp_dir=temp_dir,
            _collection_suffix=collection_suffix,
        )
        return adapter

    # ── Core benchmark operations ────────────────────────────────────────

    def store(self, text: str, source: str = "benchmark", metadata: dict | None = None) -> dict:
        """Store a memory. Maps to engine.observe().

        This is the primary write operation used by all benchmarks.
        """
        t0 = time.perf_counter()
        result = self.engine.observe(text, source=source, metadata=metadata)
        latency = (time.perf_counter() - t0) * 1000

        self._operation_log.append(OperationRecord(
            operation="store",
            input_text=text,
            result=result,
            latency_ms=latency,
            timestamp=time.time(),
        ))
        return result

    def store_batch(self, texts: list[str], source: str = "benchmark") -> list[dict]:
        """Batch store. Maps to engine.observe_batch()."""
        t0 = time.perf_counter()
        results = self.engine.observe_batch(texts, source=source)
        latency = (time.perf_counter() - t0) * 1000

        if isinstance(results, list):
            for i, result in enumerate(results):
                self._operation_log.append(OperationRecord(
                    operation="store_batch",
                    input_text=texts[i] if i < len(texts) else "",
                    result=result,
                    latency_ms=latency / len(results),
                    timestamp=time.time(),
                ))
        return results if isinstance(results, list) else [results]

    def query(self, question: str, max_tokens: int | None = None) -> dict:
        """Query memories. Maps to engine.recall().

        Returns the recall result dict with `memories` list.
        """
        t0 = time.perf_counter()
        result = self.engine.recall(question, max_tokens=max_tokens)
        latency = (time.perf_counter() - t0) * 1000

        self._operation_log.append(OperationRecord(
            operation="query",
            input_text=question,
            result=result,
            latency_ms=latency,
            timestamp=time.time(),
        ))
        return result

    def query_text(self, question: str, max_tokens: int | None = None) -> str:
        """Query and return just the concatenated memory text (convenience)."""
        result = self.query(question, max_tokens=max_tokens)
        memories = result.get("memories", [])
        return "\n".join(m.get("content", "") for m in memories)

    def update(self, old_text: str, new_text: str, source: str = "benchmark") -> dict:
        """Update a memory by storing a contradicting/updated version.

        re-memory handles this through CA1 novelty detection — storing a new
        observation that contradicts an existing one will mark the old one as
        demoted and the new one as the current version.
        """
        return self.store(new_text, source=source, metadata={"updates": old_text})

    def delete(self, memory_id: str) -> dict:
        """Delete/forget a specific memory. Maps to engine.forget()."""
        t0 = time.perf_counter()
        result = self.engine.forget(memory_id)
        latency = (time.perf_counter() - t0) * 1000

        self._operation_log.append(OperationRecord(
            operation="delete",
            input_text=memory_id,
            result=result,
            latency_ms=latency,
            timestamp=time.time(),
        ))
        return result

    def consolidate(self) -> dict:
        """Trigger consolidation. Maps to engine.consolidate()."""
        t0 = time.perf_counter()
        result = self.engine.consolidate()
        latency = (time.perf_counter() - t0) * 1000

        self._operation_log.append(OperationRecord(
            operation="consolidate",
            input_text="",
            result=result,
            latency_ms=latency,
            timestamp=time.time(),
        ))
        return result

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Direct text search. Maps to engine.search()."""
        t0 = time.perf_counter()
        results = self.engine.search(query, limit=limit)
        latency = (time.perf_counter() - t0) * 1000

        self._operation_log.append(OperationRecord(
            operation="search",
            input_text=query,
            result={"count": len(results)},
            latency_ms=latency,
            timestamp=time.time(),
        ))
        return results

    def purge(self) -> dict:
        """Wipe all data for a fresh benchmark run."""
        return self.engine.purge()

    def get_memory_ids(self) -> list[str]:
        """Get all stored memory IDs (for forget/delete benchmarks)."""
        events = self.engine.event_store.all()
        return [e.id for e in events]

    def get_memory_count(self) -> int:
        """Get total number of stored memories."""
        return self.engine.event_store.count()

    # ── Metrics helpers ──────────────────────────────────────────────────

    @property
    def operation_log(self) -> list[OperationRecord]:
        return self._operation_log

    def get_latencies(self, operation: str | None = None) -> list[float]:
        """Get latencies in ms, optionally filtered by operation type."""
        ops = self._operation_log
        if operation:
            ops = [o for o in ops if o.operation == operation]
        return [o.latency_ms for o in ops]

    def get_verdicts(self) -> dict[str, int]:
        """Count verdict types from store operations."""
        verdicts: dict[str, int] = {}
        for op in self._operation_log:
            if op.operation in ("store", "store_batch"):
                v = op.result.get("verdict", "unknown")
                verdicts[v] = verdicts.get(v, 0) + 1
        return verdicts

    # ── Lifecycle ────────────────────────────────────────────────────────

    def reset(self):
        """Reset state for a new benchmark run (purge data + clear logs)."""
        self.engine.purge()
        self._operation_log.clear()

    def close(self):
        """Tear down: close engine, clean up Qdrant collection, remove temp dir."""
        if self.engine:
            # Clean up isolated Qdrant collection
            try:
                self.engine.vector_store.client.delete_collection(
                    self.engine.vector_store.collection_name
                )
            except Exception:
                pass

            # Clean up isolated FalkorDB graph
            try:
                self.engine.graph_store.graph.query("MATCH (n) DETACH DELETE n")
            except Exception:
                pass

            self.engine.close()

        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
