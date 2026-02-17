"""Layer 3: Semantic Memory (Knowledge Graph)

Subject-predicate-object triples with temporal scoping stored in FalkorDB.
Derived from episodic events via consolidation.
"""

from __future__ import annotations

from ..storage.graph import GraphStore

__all__ = ["GraphStore"]
