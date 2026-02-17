"""Basal Ganglia (BG): Gating Controller

The basal ganglia controls what enters and leaves working memory.
It implements a gating policy — deciding which information is relevant
enough to maintain in the active buffer.

Current implementation: rule-based gating. Future: RL-trained policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GateDecision(str, Enum):
    ADMIT = "admit"  # Allow into working memory
    REJECT = "reject"  # Not relevant enough
    REPLACE = "replace"  # Replace a specific existing item


@dataclass
class GateResult:
    decision: GateDecision
    target_id: str | None = None  # ID of item to replace (if REPLACE)
    reason: str = ""


class BasalGanglia:
    """Rule-based gating policy for working memory."""

    def __init__(self, relevance_threshold: float = 0.3):
        self.relevance_threshold = relevance_threshold

    def gate(
        self,
        item_relevance: float,
        current_items: list[dict],
    ) -> GateResult:
        """Decide whether to admit a new item into working memory.

        Args:
            item_relevance: Relevance score of the candidate item (0..1)
            current_items: Current working memory contents with relevance scores
        """
        # Below threshold → reject
        if item_relevance < self.relevance_threshold:
            return GateResult(
                decision=GateDecision.REJECT,
                reason=f"Relevance {item_relevance:.2f} below threshold {self.relevance_threshold}",
            )

        # If buffer not full → admit
        if len(current_items) < 7:
            return GateResult(
                decision=GateDecision.ADMIT,
                reason="Buffer has capacity.",
            )

        # Buffer full → find least relevant item
        min_item = min(current_items, key=lambda x: x.get("relevance", 0))
        if item_relevance > min_item.get("relevance", 0):
            return GateResult(
                decision=GateDecision.REPLACE,
                target_id=min_item.get("id"),
                reason=f"Replacing least relevant item (rel={min_item.get('relevance', 0):.2f}).",
            )

        return GateResult(
            decision=GateDecision.REJECT,
            reason="All current items are more relevant.",
        )
