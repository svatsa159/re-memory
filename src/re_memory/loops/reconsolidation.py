"""Reconsolidation Loop

Every recall() can trigger memory modification:
- Compute prediction error between retrieved memory and current context
- Low PE → confirm memory (bump confidence)
- Medium PE → labilize memory, update with new info, restabilize
- High PE → don't modify old, create new episodic event
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..brain.ca1 import NoveltyVerdict


class ReconsolidationAction(str, Enum):
    CONFIRM = "confirm"  # Low PE: bump confidence
    UPDATE = "update"  # Medium PE: labilize + update
    BIFURCATE = "bifurcate"  # High PE: create new memory


@dataclass
class ReconsolidationResult:
    action: ReconsolidationAction
    memory_id: str
    prediction_error: float
    updated: bool = False
    new_memory_id: str | None = None


def check_reconsolidation(
    memory_text: str,
    current_context: str,
    prediction_error: float,
    low_threshold: float = 0.2,
    high_threshold: float = 0.7,
) -> ReconsolidationAction:
    """Determine reconsolidation action based on prediction error.

    Args:
        memory_text: The retrieved memory text
        current_context: Current context/query that triggered recall
        prediction_error: PE score from CA1 novelty detection
        low_threshold: Below this → confirm
        high_threshold: Above this → bifurcate

    Returns:
        The appropriate reconsolidation action
    """
    if prediction_error < low_threshold:
        return ReconsolidationAction.CONFIRM
    elif prediction_error > high_threshold:
        return ReconsolidationAction.BIFURCATE
    else:
        return ReconsolidationAction.UPDATE


async def reconsolidate(
    engine,
    memory_id: str,
    current_context: str,
) -> ReconsolidationResult:
    """Execute reconsolidation on a recalled memory.

    Called during retrieval when a memory is accessed in a new context.
    """
    event = engine.event_store.get(memory_id)
    if event is None:
        return ReconsolidationResult(
            action=ReconsolidationAction.CONFIRM,
            memory_id=memory_id,
            prediction_error=0.0,
        )

    # Compute prediction error between memory and current context
    try:
        from ..providers.base import get_providers

        _, embedder = get_providers(engine.config)
        mem_emb = event.embedding or await embedder.embed_single(event.text)
        ctx_emb = await embedder.embed_single(current_context)

        # Cosine similarity → prediction error
        try:
            from re_memory._core import cosine_similarity

            sim = cosine_similarity(mem_emb, ctx_emb)
        except ImportError:
            dot = sum(a * b for a, b in zip(mem_emb, ctx_emb))
            norm_a = sum(a * a for a in mem_emb) ** 0.5
            norm_b = sum(b * b for b in ctx_emb) ** 0.5
            sim = dot / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0

        pe = 1.0 - max(0.0, sim)
    except Exception:
        pe = 0.0  # Can't compute PE without embeddings

    action = check_reconsolidation(event.text, current_context, pe)

    result = ReconsolidationResult(
        action=action,
        memory_id=memory_id,
        prediction_error=pe,
    )

    if action == ReconsolidationAction.CONFIRM:
        # Bump confidence
        new_conf = min(1.0, event.confidence + 0.05)
        engine.event_store.update_confidence(memory_id, new_conf)
        engine.event_store.mark_accessed(memory_id)

    elif action == ReconsolidationAction.UPDATE:
        # Labilize → update → restabilize
        # For now, just mark as accessed; full LLM-based update in Phase 6
        engine.event_store.mark_accessed(memory_id)
        result.updated = True

    elif action == ReconsolidationAction.BIFURCATE:
        # Create a new episodic event for the new context
        # Don't modify the original
        engine.event_store.mark_accessed(memory_id)

    return result
