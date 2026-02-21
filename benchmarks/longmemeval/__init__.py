"""LongMemEval (ICLR 2025) benchmark runner.

Tests: 500 curated questions across 5 dimensions —
knowledge updates, temporal reasoning, abstention,
information extraction, and multi-session reasoning.
Scale: 115K to 1.5M tokens.

Maps to re-memory operations:
    observe → conversation ingestion
    observe (contradicting) → knowledge updates
    recall  → information extraction + temporal reasoning
    recall (empty result) → abstention
"""
