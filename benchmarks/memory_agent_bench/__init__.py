"""MemoryAgentBench (ICLR 2026) benchmark runner.

Tests: test-time learning, accurate retrieval, long-range understanding,
selective forgetting, and conflict resolution (FactConsolidation).

Maps to re-memory operations:
    observe  → test-time learning
    recall   → accurate retrieval
    consolidate → long-range understanding
    forget   → selective forgetting
    observe (contradiction) → FactConsolidation
"""
