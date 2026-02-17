"""Layer 2: Episodic Memory

Event store for immutable episodic events with hippocampal processing.
Combines the event store with DG pattern separation and CA1 novelty detection.
"""

from __future__ import annotations

from ..storage.event_store import EventStore, EpisodicEvent

__all__ = ["EventStore", "EpisodicEvent"]
