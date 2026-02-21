"""BEAM benchmark runner.

Tests: 100K to 10M tokens, explicit contradiction resolution,
event ordering. Features a brain-inspired tri-memory architecture
(episodic, working, scratchpad) conceptually similar to re-memory.

Maps to re-memory operations:
    observe → memory encoding
    recall  → memory retrieval
    observe (contradicting) → contradiction resolution
    recall (temporal) → event ordering
"""
