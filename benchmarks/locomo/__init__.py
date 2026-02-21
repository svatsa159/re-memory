"""LoCoMo (Snap Research) benchmark runner.

Tests: 300-turn multi-session conversations, multi-hop reasoning,
adversarial questions. Industry standard — Mem0, Zep, MemOS, Letta
all report on this.

Maps to re-memory operations:
    observe → per-turn conversation ingestion
    recall  → question answering
    recall  → multi-hop reasoning (requires KG traversal)
"""
