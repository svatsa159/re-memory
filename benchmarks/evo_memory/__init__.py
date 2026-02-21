"""Evo-Memory (Google DeepMind) benchmark runner.

Tests: Whether accumulated memory improves performance over time
in a streaming task format. Directly tests whether consolidation
provides real value vs. flat retrieval.

Maps to re-memory operations:
    observe → streaming memory intake
    recall  → periodic evaluation
    consolidate → accumulation effect measurement
"""
