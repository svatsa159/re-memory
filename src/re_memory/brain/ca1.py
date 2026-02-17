"""CA1: Novelty Detection (Prediction Error)

CA1 computes the mismatch between incoming input and existing memories.
This prediction error gates encoding:
  - High novelty → encode as new episodic memory
  - Low novelty (redundant) → reinforce existing, skip write
  - Contradiction → flag for reconsolidation

This is the core "should I remember this?" decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NoveltyVerdict(str, Enum):
    """What the CA1 novelty detector decides about new input."""

    NOVEL = "novel"  # New information → encode
    REDUNDANT = "redundant"  # Already known → reinforce existing
    CONTRADICTS = "contradicts"  # Conflicts with existing → flag reconsolidation
    UPDATE = "update"  # Partial update to existing knowledge


@dataclass
class NoveltyResult:
    verdict: NoveltyVerdict
    prediction_error: float  # 0 = perfect match, 1 = completely novel
    closest_memory_id: str | None = None
    closest_similarity: float = 0.0
    explanation: str = ""


class CA1NoveltyDetector:
    """Computes prediction error between new input and existing memories."""

    def __init__(self, novelty_threshold: float = 0.3):
        self.novelty_threshold = novelty_threshold

    def detect(
        self,
        query_embedding: list[float],
        nearest_matches: list[tuple[str, float]],  # (memory_id, similarity)
    ) -> NoveltyResult:
        """Determine if input is novel, redundant, or contradictory.

        Args:
            query_embedding: Embedding of the new input
            nearest_matches: Top matches from vector search as (id, similarity) pairs

        Returns:
            NoveltyResult with verdict and prediction error
        """
        if not nearest_matches:
            return NoveltyResult(
                verdict=NoveltyVerdict.NOVEL,
                prediction_error=1.0,
                explanation="No existing memories found — completely novel.",
            )

        best_id, best_sim = nearest_matches[0]
        prediction_error = 1.0 - best_sim

        if prediction_error < self.novelty_threshold:
            # Very similar to existing memory → redundant
            return NoveltyResult(
                verdict=NoveltyVerdict.REDUNDANT,
                prediction_error=prediction_error,
                closest_memory_id=best_id,
                closest_similarity=best_sim,
                explanation=f"Very similar to existing memory (sim={best_sim:.3f}).",
            )

        if prediction_error > (1.0 - self.novelty_threshold):
            # Very different → novel
            return NoveltyResult(
                verdict=NoveltyVerdict.NOVEL,
                prediction_error=prediction_error,
                closest_memory_id=best_id,
                closest_similarity=best_sim,
                explanation=f"Significantly different from all memories (PE={prediction_error:.3f}).",
            )

        # Moderate prediction error → could be an update
        return NoveltyResult(
            verdict=NoveltyVerdict.UPDATE,
            prediction_error=prediction_error,
            closest_memory_id=best_id,
            closest_similarity=best_sim,
            explanation=f"Moderate similarity — possible update (PE={prediction_error:.3f}).",
        )

    async def detect_with_contradiction(
        self,
        new_text: str,
        existing_text: str,
        llm,
    ) -> bool:
        """Use LLM to determine if new information contradicts existing memory."""
        prompt = f"""Do these two statements contradict each other? Answer with JSON:
{{"contradicts": true/false, "reason": "brief explanation"}}

Statement A (existing): {existing_text}
Statement B (new): {new_text}"""

        try:
            result = await llm.complete_json(prompt)
            return result.get("contradicts", False)
        except Exception:
            return False
