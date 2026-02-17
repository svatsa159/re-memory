"""SQLite-based episodic event store.

Stores immutable episodic events with full metadata.
Dev mode uses SQLite; production mode can use PostgreSQL.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class EpisodicEvent:
    """A single episodic memory event."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    embedding: list[float] = field(default_factory=list)
    sparse_code: list[int] = field(default_factory=list)
    confidence: float = 0.5
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    source: str = "unknown"
    metadata: dict = field(default_factory=dict)
    access_count: int = 0
    synaptic_tag_ttl: float = 72.0  # hours before tag expires
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_accessed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    consolidated: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't include embedding in display output (too large)
        d.pop("embedding", None)
        return d

    def to_full_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EpisodicEvent:
        # Handle fields that might be JSON strings
        for list_field in ("embedding", "sparse_code", "tags"):
            if isinstance(data.get(list_field), str):
                data[list_field] = json.loads(data[list_field])
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class EventStore:
    """SQLite-backed episodic event store."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def init(self):
        """Create tables and indexes."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodic_events (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT DEFAULT '[]',
                sparse_code TEXT DEFAULT '[]',
                confidence REAL DEFAULT 0.5,
                importance REAL DEFAULT 0.5,
                tags TEXT DEFAULT '[]',
                source TEXT DEFAULT 'unknown',
                metadata TEXT DEFAULT '{}',
                access_count INTEGER DEFAULT 0,
                synaptic_tag_ttl REAL DEFAULT 72.0,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL,
                consolidated INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_events_created
                ON episodic_events(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_events_confidence
                ON episodic_events(confidence);
            CREATE INDEX IF NOT EXISTS idx_events_importance
                ON episodic_events(importance);
            CREATE INDEX IF NOT EXISTS idx_events_consolidated
                ON episodic_events(consolidated);
            CREATE INDEX IF NOT EXISTS idx_events_source
                ON episodic_events(source);

            CREATE TABLE IF NOT EXISTS memory_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                memory_id TEXT,
                details TEXT DEFAULT '{}',
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_ops_timestamp
                ON memory_operations(timestamp DESC);
        """)
        self.conn.commit()

    def upsert(self, event: EpisodicEvent):
        """Insert or update an episodic event."""
        self.conn.execute(
            """INSERT OR REPLACE INTO episodic_events
               (id, text, embedding, sparse_code, confidence, importance,
                tags, source, metadata, access_count, synaptic_tag_ttl,
                created_at, last_accessed_at, consolidated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.id,
                event.text,
                json.dumps(event.embedding),
                json.dumps(event.sparse_code),
                event.confidence,
                event.importance,
                json.dumps(event.tags),
                event.source,
                json.dumps(event.metadata),
                event.access_count,
                event.synaptic_tag_ttl,
                event.created_at,
                event.last_accessed_at,
                int(event.consolidated),
            ),
        )
        self.conn.commit()

    def get(self, event_id: str) -> EpisodicEvent | None:
        """Get a single event by ID."""
        row = self.conn.execute(
            "SELECT * FROM episodic_events WHERE id = ?", (event_id,)
        ).fetchone()
        if row is None:
            return None
        return EpisodicEvent.from_dict(dict(row))

    def delete(self, event_id: str):
        """Delete an event by ID."""
        self.conn.execute("DELETE FROM episodic_events WHERE id = ?", (event_id,))
        self.conn.commit()
        self._log_operation("forget", event_id)

    def count(self) -> int:
        """Count total events."""
        row = self.conn.execute("SELECT COUNT(*) FROM episodic_events").fetchone()
        return row[0]

    def recent(self, limit: int = 20) -> list[EpisodicEvent]:
        """Get most recent events."""
        rows = self.conn.execute(
            "SELECT * FROM episodic_events ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [EpisodicEvent.from_dict(dict(r)) for r in rows]

    def all(self) -> list[EpisodicEvent]:
        """Get all events."""
        rows = self.conn.execute(
            "SELECT * FROM episodic_events ORDER BY created_at DESC"
        ).fetchall()
        return [EpisodicEvent.from_dict(dict(r)) for r in rows]

    def search_text(self, query: str, limit: int = 10) -> list[EpisodicEvent]:
        """Simple text search across event texts."""
        rows = self.conn.execute(
            """SELECT * FROM episodic_events
               WHERE text LIKE ?
               ORDER BY confidence DESC, created_at DESC
               LIMIT ?""",
            (f"%{query}%", limit),
        ).fetchall()
        return [EpisodicEvent.from_dict(dict(r)) for r in rows]

    def unconsolidated(self, limit: int = 100) -> list[EpisodicEvent]:
        """Get events that haven't been consolidated yet."""
        rows = self.conn.execute(
            """SELECT * FROM episodic_events
               WHERE consolidated = 0
               ORDER BY importance DESC, created_at ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [EpisodicEvent.from_dict(dict(r)) for r in rows]

    def update_confidence(self, event_id: str, confidence: float):
        """Update confidence score for an event."""
        self.conn.execute(
            "UPDATE episodic_events SET confidence = ? WHERE id = ?",
            (confidence, event_id),
        )
        self.conn.commit()

    def mark_accessed(self, event_id: str):
        """Record an access to an event (bump access_count, update timestamp)."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE episodic_events
               SET access_count = access_count + 1, last_accessed_at = ?
               WHERE id = ?""",
            (now, event_id),
        )
        self.conn.commit()

    def mark_consolidated(self, event_id: str):
        """Mark an event as consolidated."""
        self.conn.execute(
            "UPDATE episodic_events SET consolidated = 1 WHERE id = ?",
            (event_id,),
        )
        self.conn.commit()

    def prune_below_confidence(self, threshold: float) -> int:
        """Delete events below confidence threshold. Returns count deleted."""
        cursor = self.conn.execute(
            "DELETE FROM episodic_events WHERE confidence < ?",
            (threshold,),
        )
        self.conn.commit()
        return cursor.rowcount

    def _log_operation(self, operation: str, memory_id: str = "", details: dict | None = None):
        """Log a memory operation for history tracking."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO memory_operations (operation, memory_id, details, timestamp)
               VALUES (?, ?, ?, ?)""",
            (operation, memory_id, json.dumps(details or {}), now),
        )
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
