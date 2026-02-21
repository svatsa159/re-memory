"""MemoryEngine: Main orchestrator for the re-memory system.

Coordinates all brain components, memory layers, and processing loops.
This is the central entry point that the CLI and programmatic APIs call.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
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
    _executor: ThreadPoolExecutor | None = None

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

    def observe(
        self,
        text: str,
        source: str = "cli",
        metadata: dict | None = None,
        tier: str | None = None,
        background: bool = False,
    ) -> dict:
        """Write path: parse → encode → store.

        Args:
            tier: Override observe pipeline tier (fast/standard/full).
            background: If True, submit to background thread and return immediately.
        """
        from .loops.encoding import encode

        if background:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1)
            self._executor.submit(self._observe_sync, text, source, metadata, tier)
            return {"status": "accepted", "background": True}

        return encode(self, text, source=source, metadata=metadata, tier=tier)

    def _observe_sync(
        self, text: str, source: str, metadata: dict | None, tier: str | None
    ) -> dict:
        """Synchronous observe for background execution."""
        from .loops.encoding import encode
        return encode(self, text, source=source, metadata=metadata, tier=tier)

    def observe_batch(
        self,
        texts: list[str],
        source: str = "cli",
        tier: str | None = None,
        background: bool = False,
    ) -> dict | list[dict]:
        """Batch observe: encode multiple texts efficiently.

        Args:
            texts: List of texts to observe.
            source: Source label.
            tier: Override observe tier.
            background: If True, process in background thread.
        """
        from .loops.encoding import encode_batch

        if background:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1)
            self._executor.submit(encode_batch, self, texts, source, tier)
            return {"status": "accepted", "background": True, "count": len(texts)}

        return encode_batch(self, texts, source=source, tier=tier)

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
          - Invalidates schemas referencing the forgotten content (Layer 4)
        """
        # Get event text before deleting (needed for schema invalidation)
        event = self.event_store.get(memory_id)
        event_text = event.text if event else ""

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

        # Invalidate schemas that reference the forgotten content
        if event_text:
            try:
                schemas_invalidated = self._invalidate_schemas_for(event_text)
                if schemas_invalidated:
                    deleted_layers.append(f"schemas({schemas_invalidated})")
            except Exception:
                pass

        if deleted_layers:
            return {
                "status": "forgotten",
                "id": memory_id,
                "deleted_from": deleted_layers,
            }
        return {"status": "error", "error": "Memory not found in any layer"}

    def _invalidate_schemas_for(self, text: str) -> int:
        """Surgically remove lines referencing the forgotten text from schemas.

        Instead of deleting entire schema files, removes only the lines/paragraphs
        that contain keywords from the forgotten text. If a schema becomes empty
        after removal, deletes the file entirely. Marks related events as
        unconsolidated so next consolidation rebuilds schemas from remaining data.
        """
        from .storage.file_store import FileStore

        file_store = FileStore(self.config.schema_path)
        matching = file_store.search(text)

        if not matching:
            return 0

        # Extract meaningful keywords from the forgotten text
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "do", "does", "did", "will", "would", "could", "should", "may",
            "might", "can", "shall", "has", "have", "had", "of", "in", "to",
            "for", "on", "at", "by", "with", "from", "about", "into", "and",
            "or", "but", "not", "no", "so", "if", "then", "than", "that",
            "this", "it", "its", "my", "your", "his", "her", "our", "their",
            "user", "prefers", "likes", "uses",
        }
        keywords = [
            w.lower() for w in text.split()
            if w.lower() not in stop_words and len(w) > 2
        ]

        updated = 0
        for match in matching:
            category = match["category"]
            content = file_store.get(category)
            if not content:
                continue

            # Remove lines containing any keyword from the forgotten text
            lines = content.split("\n")
            kept = []
            removed_any = False
            for line in lines:
                line_lower = line.lower()
                # Keep headings, timestamp comments, and lines without keyword matches
                is_heading = line.strip().startswith("#")
                is_comment = line.strip().startswith("<!--")
                has_keyword = any(kw in line_lower for kw in keywords)
                if is_heading or is_comment or not has_keyword:
                    kept.append(line)
                else:
                    removed_any = True

            if not removed_any:
                continue

            # Check if anything meaningful remains (beyond headings/comments/blank)
            meaningful = [
                l for l in kept
                if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("<!--")
            ]

            if meaningful:
                file_store.put(category, "\n".join(kept))
            else:
                file_store.delete(category)
            updated += 1

        # Mark related events as unconsolidated so next consolidation
        # rebuilds the affected schemas from remaining graph data
        if updated > 0:
            self.event_store.conn.execute(
                "UPDATE episodic_events SET consolidated = 0 "
                "WHERE consolidated = 1"
            )
            self.event_store.conn.commit()

        return updated

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
            "events": [e.to_full_dict() for e in events],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return {"exported": len(events), "path": str(path)}

    def import_data(self, path: Path) -> dict:
        """Import memory data from a JSON file.

        Restores events to SQLite and re-indexes embeddings into Qdrant.
        Knowledge graph is NOT rebuilt — run `consolidate` after import for that.
        """
        with open(path) as f:
            data = json.load(f)

        count = 0
        vectors_indexed = 0
        for event_data in data.get("events", []):
            event = EpisodicEvent.from_dict(event_data)
            self.event_store.upsert(event)
            count += 1

            # Re-index embedding into Qdrant if available
            if event.embedding:
                try:
                    self.vector_store.upsert(
                        point_id=event.id,
                        vector=event.embedding,
                        payload={
                            "text": event.text,
                            "source": event.source,
                            "importance": event.importance,
                            "created_at": event.created_at,
                        },
                    )
                    vectors_indexed += 1
                except Exception:
                    pass

        return {
            "imported": count,
            "vectors_indexed": vectors_indexed,
            "path": str(path),
        }

    def purge(self) -> dict:
        """Wipe all memory stores. Deletes every episodic event, vector,
        graph node/edge, and schema file. Re-initializes empty stores."""
        purged = {}

        # SQLite — delete all events and operation log
        try:
            events = self.event_store.all()
            count = len(events)
            for ev in events:
                self.event_store.delete(ev.id)
            # Clear operation log too
            self.event_store.conn.execute("DELETE FROM memory_operations")
            self.event_store.conn.commit()
            purged["episodic_events"] = count
        except Exception as e:
            purged["episodic_events"] = f"error: {e}"

        # Qdrant — drop and recreate collection
        try:
            self.vector_store.client.delete_collection(self.vector_store.collection_name)
            self.vector_store.init()
            purged["vectors"] = "purged"
        except Exception as e:
            purged["vectors"] = f"error: {e}"

        # FalkorDB — delete all nodes and edges
        try:
            self.graph_store.graph.query("MATCH (n) DETACH DELETE n")
            purged["graph"] = "purged"
        except Exception as e:
            purged["graph"] = f"error: {e}"

        # Schema files
        try:
            schema_dir = self.config.schema_path
            if schema_dir.exists():
                schemas = list(schema_dir.glob("*.md"))
                for f in schemas:
                    f.unlink()
                purged["schemas"] = len(schemas)
            else:
                purged["schemas"] = 0
        except Exception as e:
            purged["schemas"] = f"error: {e}"

        return {"status": "purged", **purged}

    def close(self):
        """Close all connections and shut down background executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._event_store:
            self._event_store.close()
        if self._vector_store:
            self._vector_store.close()
        if self._graph_store:
            self._graph_store.close()
