"""Prefrontal Cortex (PFC): Working Memory Buffer

The prefrontal cortex maintains active representations in working memory.
Capacity is limited (~7 items, theta-gamma coupling) with LRU eviction.

Working memory persists within a session/task and is cleared between sessions.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class WorkingMemoryItem:
    """A single item in working memory."""

    id: str
    content: str
    relevance: float = 1.0
    added_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class PrefrontalCortex:
    """Fixed-capacity working memory buffer with LRU eviction."""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self._buffer: OrderedDict[str, WorkingMemoryItem] = OrderedDict()

    def add(self, item: WorkingMemoryItem) -> WorkingMemoryItem | None:
        """Add item to working memory. Returns evicted item if at capacity."""
        evicted = None

        # If already present, move to end (most recent)
        if item.id in self._buffer:
            self._buffer.move_to_end(item.id)
            self._buffer[item.id] = item
            return None

        # Evict LRU if at capacity
        if len(self._buffer) >= self.capacity:
            _, evicted = self._buffer.popitem(last=False)

        self._buffer[item.id] = item
        return evicted

    def get(self, item_id: str) -> WorkingMemoryItem | None:
        """Retrieve and touch an item (moves to end)."""
        if item_id in self._buffer:
            self._buffer.move_to_end(item_id)
            return self._buffer[item_id]
        return None

    def remove(self, item_id: str) -> bool:
        """Explicitly remove an item."""
        if item_id in self._buffer:
            del self._buffer[item_id]
            return True
        return False

    def contents(self) -> list[WorkingMemoryItem]:
        """Get all items in order (oldest first)."""
        return list(self._buffer.values())

    def clear(self):
        """Clear working memory (session end)."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.capacity

    def to_context_string(self) -> str:
        """Format working memory contents as a context string."""
        if not self._buffer:
            return ""
        items = self.contents()
        lines = [f"- {item.content}" for item in items]
        return "Active context:\n" + "\n".join(lines)
