"""Encoding Loop (Write Path)

Input → EC (parse + embed)
  → DG (pattern separate)
  → CA1 (novelty detection)
  → Amygdala (importance scoring)
  → If novel: create Episodic Event with synaptic tag
  → If redundant: reinforce existing memory
  → If contradicts: flag for reconsolidation
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from datetime import datetime, timezone

from ..storage.event_store import EpisodicEvent


def encode(engine, text: str, source: str = "cli", metadata: dict | None = None) -> dict:
    """Synchronous wrapper for the encoding pipeline."""
    return asyncio.run(_encode_async(engine, text, source, metadata))


async def _encode_async(
    engine, text: str, source: str, metadata: dict | None
) -> dict:
    """Full encoding pipeline: EC → DG → CA1 → Amygdala → Store.

    Parallelizes independent LLM calls for performance:
      - Embedding, LLM feature extraction, and importance scoring run concurrently
      - Contradiction check runs after novelty detection (needs its result)
    """
    from ..brain.dentate_gyrus import pattern_separate
    from ..brain.ca1 import CA1NoveltyDetector, NoveltyVerdict

    # --- Phase 1: Parallel LLM + embedding calls ---
    embedding = None
    parsed_features = {}
    importance = 0.5
    llm = None
    llm_available = False

    try:
        from ..providers.base import get_providers

        llm, embedder = get_providers(engine.config)
        llm_available = True

        from ..brain.entorhinal import EntorhinalCortex
        from ..brain.amygdala import Amygdala

        ec = EntorhinalCortex(llm, embedder)
        amygdala = Amygdala(llm, threshold=engine.config.memory.importance_threshold)

        # Run embedding+features and importance scoring concurrently
        parse_task = asyncio.create_task(ec.parse(text))
        importance_task = asyncio.create_task(amygdala.score_importance(text))

        parsed, importance = await asyncio.gather(parse_task, importance_task)

        embedding = parsed.embedding
        parsed_features = {
            "entities": parsed.entities,
            "topics": parsed.topics,
            "sentiment": parsed.sentiment,
            "summary": parsed.summary,
        }
    except Exception:
        pass

    # Fallback embedding if provider unavailable
    if embedding is None:
        embedding = _fallback_embedding(text, engine.config.embedding.dimensions)

    # --- Phase 2: DG + CA1 (needs embedding) ---
    sparse_code = pattern_separate(embedding)

    novelty_detector = CA1NoveltyDetector(
        novelty_threshold=engine.config.memory.novelty_threshold
    )

    # Search for similar memories via vector store
    nearest = []
    try:
        results = engine.vector_store.search(embedding, limit=5)
        nearest = [(r["id"], r["score"]) for r in results]
    except Exception:
        pass

    # Text similarity fallback
    if not nearest:
        similar = engine.event_store.search_text(text, limit=3)
        for s in similar:
            nearest.append((s.id, 0.5))

    novelty = novelty_detector.detect(embedding, nearest)

    # --- Phase 3: Contradiction check (needs novelty result) ---
    # Check both UPDATE and REDUNDANT verdicts — semantically similar statements
    # (e.g., "works at Google" vs "works at OpenAI") can have near-identical
    # embeddings but still contradict each other. Only skip if the text is an
    # exact duplicate of the existing memory.
    if (
        novelty.verdict in (NoveltyVerdict.UPDATE, NoveltyVerdict.REDUNDANT)
        and novelty.closest_memory_id
        and llm_available
    ):
        existing = engine.event_store.get(novelty.closest_memory_id)
        if existing and existing.text.strip() != text.strip():
            is_contradiction = await novelty_detector.detect_with_contradiction(
                text, existing.text, llm
            )
            if is_contradiction:
                novelty.verdict = NoveltyVerdict.CONTRADICTS

    # Redundant → reinforce existing, don't create new event
    if novelty.verdict == NoveltyVerdict.REDUNDANT:
        if novelty.closest_memory_id:
            engine.event_store.mark_accessed(novelty.closest_memory_id)
            existing = engine.event_store.get(novelty.closest_memory_id)
            if existing:
                new_conf = min(1.0, existing.confidence + 0.05)
                engine.event_store.update_confidence(existing.id, new_conf)
        return {
            "status": "redundant",
            "verdict": novelty.verdict.value,
            "prediction_error": novelty.prediction_error,
            "closest_memory": novelty.closest_memory_id,
            "explanation": novelty.explanation,
        }

    # Contradiction handling: archive old fact, store new
    if novelty.verdict == NoveltyVerdict.CONTRADICTS and novelty.closest_memory_id:
        existing = engine.event_store.get(novelty.closest_memory_id)
        if existing:
            # Lower confidence on old memory rather than delete — let consolidation decide
            new_conf = max(0.1, existing.confidence * 0.5)
            engine.event_store.update_confidence(existing.id, new_conf)

    # Determine initial confidence based on novelty verdict
    if novelty.verdict == NoveltyVerdict.CONTRADICTS:
        confidence = 0.6  # Slightly higher — contradictions are usually intentional corrections
    elif novelty.verdict == NoveltyVerdict.UPDATE:
        confidence = 0.55
    else:
        confidence = 0.5

    # --- Create episodic event ---
    tags = parsed_features.get("topics", [])
    event = EpisodicEvent(
        text=text,
        embedding=embedding,
        sparse_code=sparse_code,
        confidence=confidence,
        importance=importance,
        tags=tags,
        source=source,
        metadata={
            **(metadata or {}),
            **{k: v for k, v in parsed_features.items() if k != "topics"},
        },
    )

    # Store in event store
    engine.event_store.upsert(event)
    engine.event_store._log_operation("observe", event.id, {
        "source": source,
        "verdict": novelty.verdict.value,
        "importance": importance,
    })

    # Store embedding in vector store
    try:
        engine.vector_store.upsert(
            point_id=event.id,
            vector=embedding,
            payload={
                "text": text,
                "source": source,
                "importance": importance,
                "created_at": event.created_at,
            },
        )
    except Exception:
        pass

    return {
        "status": "encoded",
        "id": event.id,
        "verdict": novelty.verdict.value,
        "prediction_error": novelty.prediction_error,
        "confidence": event.confidence,
        "importance": importance,
        "sparse_code_bits": len(sparse_code),
        "tags": tags,
        "timestamp": event.created_at,
    }


def _fallback_embedding(text: str, dimensions: int) -> list[float]:
    """Generate a deterministic pseudo-embedding when no provider is available."""
    h = hashlib.sha256(text.encode()).digest()
    seed = int.from_bytes(h[:8], "little")
    rng = random.Random(seed)
    return [rng.gauss(0, 0.1) for _ in range(dimensions)]
