"""MemoryEngine: Main orchestrator for the re-memory system.

Coordinates all brain components, memory layers, and processing loops.
This is the central entry point that the CLI and programmatic APIs call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, load_config
from .storage.event_store import EventStore, EpisodicEvent
from .storage.vector import VectorStore
from .storage.graph import GraphStore
from .providers.base import get_providers


@dataclass
class MemoryEngine:
    """Main orchestrator for the brain-anatomical memory system."""

    config: Config = field(default_factory=load_config)
    _event_store: EventStore | None = None
    _vector_store: VectorStore | None = None
    _graph_store: GraphStore | None = None
    _initialized: bool = False

    @property
    def event_store(self) -> EventStore:
        if self._event_store is None:
            self._event_store = EventStore(self.config.sqlite_path)
        return self._event_store

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(
                url=self.config.storage.qdrant_url,
                embedding_dim=self.config.embedding.dimensions,
            )
        return self._vector_store

    @property
    def graph_store(self) -> GraphStore:
        if self._graph_store is None:
            self._graph_store = GraphStore(
                host=self.config.storage.falkordb_host,
                port=self.config.storage.falkordb_port,
            )
        return self._graph_store

    def init(self) -> dict:
        """Initialize the memory system: create directories, tables, indexes."""
        results = {}

        # Create data directories
        self.config.data_path.mkdir(parents=True, exist_ok=True)
        self.config.schema_path.mkdir(parents=True, exist_ok=True)
        results["data_dir"] = str(self.config.data_path)

        # Initialize SQLite event store
        self.event_store.init()
        results["event_store"] = "initialized"

        # Try connecting to vector store
        try:
            self.vector_store.init()
            results["vector_store"] = "connected"
        except Exception as e:
            results["vector_store"] = f"unavailable: {e}"

        # Try connecting to graph store
        try:
            self.graph_store.init()
            results["graph_store"] = "connected"
        except Exception as e:
            results["graph_store"] = f"unavailable: {e}"

        self._initialized = True
        return results

    def status(self) -> dict:
        """Get memory system health and statistics."""
        status = {
            "version": "0.1.0",
            "data_dir": str(self.config.data_path),
            "config_file": str(self.config.data_path / "config.toml"),
            "stores": {},
            "counts": {},
        }

        # Event store status
        try:
            counts = self.event_store.count()
            status["stores"]["event_store"] = "ok"
            status["counts"]["episodic_events"] = counts
        except Exception as e:
            status["stores"]["event_store"] = f"error: {e}"

        # Vector store status
        try:
            info = self.vector_store.status()
            if info.get("status") == "unavailable":
                status["stores"]["vector_store"] = "unavailable"
                status["counts"]["vectors"] = 0
            else:
                status["stores"]["vector_store"] = "ok"
                status["counts"]["vectors"] = info.get("vectors_count", 0)
        except Exception:
            status["stores"]["vector_store"] = "unavailable"
            status["counts"]["vectors"] = 0

        # Graph store status
        try:
            info = self.graph_store.status()
            if info.get("status") == "unavailable":
                status["stores"]["graph_store"] = "unavailable"
                status["counts"]["graph_nodes"] = 0
                status["counts"]["graph_edges"] = 0
            else:
                status["stores"]["graph_store"] = "ok"
                status["counts"].update({k: v for k, v in info.items() if k != "status"})
        except Exception:
            status["stores"]["graph_store"] = "unavailable"
            status["counts"]["graph_nodes"] = 0
            status["counts"]["graph_edges"] = 0

        # Schema files
        schema_dir = self.config.schema_path
        if schema_dir.exists():
            schemas = list(schema_dir.glob("*.md"))
            status["counts"]["schemas"] = len(schemas)
        else:
            status["counts"]["schemas"] = 0

        return status

    def observe(self, text: str, source: str = "cli", metadata: dict | None = None) -> dict:
        """Write path: parse → encode → store.

        This is the main entry point for the encoding loop.
        In Phase 1, we do basic storage. Full brain pipeline comes in Phase 2.
        """
        from .loops.encoding import encode

        return encode(self, text, source=source, metadata=metadata)

    def recall(self, query: str, max_tokens: int | None = None, **kwargs) -> dict:
        """Read path: goal-conditioned retrieval.

        Tiered retrieval: Schema → KG → Episodic events.
        In Phase 1, we do basic search. Full brain pipeline comes in Phase 3.
        """
        from .loops.retrieval import retrieve

        return retrieve(
            self,
            query,
            max_tokens=max_tokens or self.config.memory.max_retrieval_tokens,
            **kwargs,
        )

    def consolidate(self) -> dict:
        """Trigger consolidation loop manually."""
        from .loops.consolidation import consolidate

        return consolidate(self)

    def forget(self, memory_id: str) -> dict:
        """Explicitly forget a memory, cascading across all layers.

        Removes:
          - Episodic event from SQLite (Layer 2)
          - Embedding vector from Qdrant
          - Graph triples sourced from this event (Layer 3)
        """
        deleted_layers = []
        try:
            self.event_store.delete(memory_id)
            deleted_layers.append("episodic")
        except Exception:
            pass

        # Remove vector embedding
        try:
            self.vector_store.delete(memory_id)
            deleted_layers.append("vector")
        except Exception:
            pass

        # Remove graph triples sourced from this event
        try:
            count = self.graph_store.delete_by_source_event(memory_id)
            if count > 0:
                deleted_layers.append(f"graph({count} triples)")
        except Exception:
            pass

        if deleted_layers:
            return {
                "status": "forgotten",
                "id": memory_id,
                "deleted_from": deleted_layers,
            }
        return {"status": "error", "error": "Memory not found in any layer"}

    def inspect(self, memory_id: str) -> dict | None:
        """View a specific memory with full metadata."""
        event = self.event_store.get(memory_id)
        if event is None:
            return None
        return event.to_dict()

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Direct search across all layers."""
        results = self.event_store.search_text(query, limit=limit)
        return [e.to_dict() for e in results]

    def history(self, limit: int = 20) -> list[dict]:
        """Timeline of recent memory operations."""
        events = self.event_store.recent(limit=limit)
        return [e.to_dict() for e in events]

    def export_data(self, path: Path) -> dict:
        """Export all memory data to a JSON file."""
        events = self.event_store.all()
        data = {
            "version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "events": [e.to_dict() for e in events],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return {"exported": len(events), "path": str(path)}

    def import_data(self, path: Path) -> dict:
        """Import memory data from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        count = 0
        for event_data in data.get("events", []):
            event = EpisodicEvent.from_dict(event_data)
            self.event_store.upsert(event)
            count += 1

        return {"imported": count, "path": str(path)}

    def close(self):
        """Close all connections."""
        if self._event_store:
            self._event_store.close()
        if self._vector_store:
            self._vector_store.close()
        if self._graph_store:
            self._graph_store.close()
