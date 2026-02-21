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

        Thresholds (with default novelty_threshold=0.3):
          - REDUNDANT: PE < 0.3  (sim > 0.70) — near-duplicate
          - UPDATE:    PE < 0.45 (sim > 0.55) — same topic, modified info
          - NOVEL:     PE >= 0.45 (sim <= 0.55) — genuinely new
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

        # UPDATE band: slightly above redundancy threshold
        # Only memories close enough to be about the same subject
        update_ceiling = self.novelty_threshold + 0.15
        if prediction_error < update_ceiling:
            return NoveltyResult(
                verdict=NoveltyVerdict.UPDATE,
                prediction_error=prediction_error,
                closest_memory_id=best_id,
                closest_similarity=best_sim,
                explanation=f"Similar to existing memory — possible update (PE={prediction_error:.3f}).",
            )

        # Everything else is novel
        return NoveltyResult(
            verdict=NoveltyVerdict.NOVEL,
            prediction_error=prediction_error,
            closest_memory_id=best_id,
            closest_similarity=best_sim,
            explanation=f"Sufficiently different from existing memories (PE={prediction_error:.3f}).",
        )

    async def detect_with_contradiction(
        self,
        new_text: str,
        existing_text: str,
        llm,
    ) -> bool:
        """Use LLM to determine if new information contradicts existing memory."""
        prompt = f"""You are comparing two memory statements about the same entity.

Statement A (old): {existing_text}
Statement B (new): {new_text}

Does Statement B REPLACE or CONTRADICT Statement A? Think step by step:
1. Do both statements describe the same entity or person?
2. Do they describe the same attribute (job, location, preference, status, role)?
3. Do they give DIFFERENT values for that attribute?

If YES to all three → contradicts.

Examples of contradictions:
- "works at Google" vs "works at OpenAI" → YES (same person, same attribute: employer, different values)
- "lives in NYC" vs "lives in London" → YES (same attribute: location)
- "prefers Python" vs "prefers Rust" → YES (same attribute: language preference)
- "works at Google" vs "works at Google DeepMind" → YES (employer changed/updated)

Examples of NOT contradictions:
- "likes Python" vs "works at Google" → NO (different attributes)
- "has a dog" vs "has a cat" → NO (can have both)

Return JSON: {{"contradicts": true/false, "reason": "brief"}}"""

        try:
            result = await llm.complete_json(prompt)
            return result.get("contradicts", False)
        except Exception:
            return False
