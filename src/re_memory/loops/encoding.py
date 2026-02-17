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
    """Full encoding pipeline: EC → DG → CA1 → Amygdala → Store."""
    from ..brain.dentate_gyrus import pattern_separate
    from ..brain.ca1 import CA1NoveltyDetector, NoveltyVerdict

    # --- EC: Entorhinal Cortex — parse + embed ---
    embedding = None
    parsed_features = {}
    llm_available = False

    try:
        from ..providers.base import get_providers

        llm, embedder = get_providers(engine.config)
        llm_available = True

        # Full EC parse: embedding + LLM feature extraction
        from ..brain.entorhinal import EntorhinalCortex

        ec = EntorhinalCortex(llm, embedder)
        parsed = await ec.parse(text)
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

    # --- DG: Dentate Gyrus — pattern separation ---
    sparse_code = pattern_separate(embedding)

    # --- CA1: Novelty Detection ---
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

    # Handle contradictions via LLM if available
    if novelty.verdict == NoveltyVerdict.UPDATE and novelty.closest_memory_id and llm_available:
        existing = engine.event_store.get(novelty.closest_memory_id)
        if existing:
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

    # --- Amygdala: Importance scoring ---
    importance = 0.5
    if llm_available:
        try:
            from ..brain.amygdala import Amygdala

            amygdala = Amygdala(llm, threshold=engine.config.memory.importance_threshold)
            importance = await amygdala.score_importance(text)
        except Exception:
            pass

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
