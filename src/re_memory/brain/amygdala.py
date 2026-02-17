"""Amygdala: Importance / Salience Scoring

The amygdala modulates memory encoding based on emotional significance
and behavioral relevance. High-importance memories get stronger encoding
and slower decay.

In our system: scores incoming observations for "is this worth remembering?"
"""

from __future__ import annotations

from ..providers.base import LLMProvider


class Amygdala:
    """Importance and salience scorer for incoming memories."""

    def __init__(self, llm: LLMProvider, threshold: float = 0.5):
        self.llm = llm
        self.threshold = threshold

    async def score_importance(self, text: str, context: str = "") -> float:
        """Score the importance/salience of new information (0..1).

        Higher scores for:
        - Personal preferences and decisions
        - Facts about the user
        - Project-critical information
        - Emotional or consequential events
        - Corrections or contradictions

        Lower scores for:
        - Trivial chitchat
        - Already well-known general facts
        - Transient/ephemeral observations
        """
        prompt = f"""Rate the importance of remembering this information on a scale of 0.0 to 1.0.
Consider: Is this personal? Actionable? A preference? A fact about the user?
Would forgetting this cause problems?

Return JSON: {{"importance": <float>, "reason": "<brief>"}}

{f'Context: {context}' if context else ''}
Information: {text}"""

        try:
            result = await self.llm.complete_json(prompt)
            score = float(result.get("importance", 0.5))
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5  # Default to medium importance on failure

    def should_encode(self, importance: float) -> bool:
        """Gate decision: is this important enough to store?"""
        return importance >= self.threshold
