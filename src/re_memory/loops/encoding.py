"""Encoding Loop (Write Path)

Input → EC (parse + embed)
  → DG (pattern separate)
  → CA1 (novelty detection)
  → Amygdala (importance scoring)
  → If novel: create Episodic Event with synaptic tag
  → If redundant: reinforce existing memory
  → If contradicts: flag for reconsolidation

Supports three tiers:
  - fast: embed + DG + CA1 + store (no LLM calls, ~150ms)
  - standard: embed + combined LLM call + DG + CA1 + contradiction + store (1 LLM call)
  - full: embed + separate LLM calls + DG + CA1 + contradiction + store (2-3 LLM calls)
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections import OrderedDict
from datetime import datetime, timezone

from ..storage.event_store import EpisodicEvent

# --- Exact-text dedup cache ---
# Keyed on SHA256 of stripped text → memory_id
_DEDUP_CACHE: OrderedDict[str, str] = OrderedDict()
_DEDUP_CACHE_MAX = 256


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def _cache_get(text: str) -> str | None:
    key = _cache_key(text)
    if key in _DEDUP_CACHE:
        _DEDUP_CACHE.move_to_end(key)
        return _DEDUP_CACHE[key]
    return None


def _cache_put(text: str, memory_id: str) -> None:
    key = _cache_key(text)
    _DEDUP_CACHE[key] = memory_id
    if len(_DEDUP_CACHE) > _DEDUP_CACHE_MAX:
        _DEDUP_CACHE.popitem(last=False)


def encode(
    engine,
    text: str,
    source: str = "cli",
    metadata: dict | None = None,
    tier: str | None = None,
) -> dict:
    """Synchronous wrapper for the encoding pipeline."""
    return asyncio.run(_encode_async(engine, text, source, metadata, tier))


async def _encode_async(
    engine, text: str, source: str, metadata: dict | None, tier: str | None = None,
) -> dict:
    """Dispatch to the appropriate tier."""
    # --- Dedup cache check ---
    cached_id = _cache_get(text)
    if cached_id is not None:
        existing = engine.event_store.get(cached_id)
        if existing is not None:
            engine.event_store.mark_accessed(cached_id)
            new_conf = min(1.0, existing.confidence + 0.05)
            engine.event_store.update_confidence(cached_id, new_conf)
            return {
                "status": "redundant",
                "verdict": "redundant",
                "prediction_error": 0.0,
                "closest_memory": cached_id,
                "explanation": "Exact text match (dedup cache hit).",
            }

    # Resolve tier
    effective_tier = tier or getattr(engine.config.memory, "observe_tier", "standard")

    if effective_tier == "fast":
        return await _encode_fast(engine, text, source, metadata)
    elif effective_tier == "full":
        return await _encode_full(engine, text, source, metadata)
    else:
        return await _encode_standard(engine, text, source, metadata)


async def _encode_fast(
    engine, text: str, source: str, metadata: dict | None,
) -> dict:
    """Fast tier: embed + DG + CA1 + store. No LLM calls. ~150ms."""
    from ..brain.dentate_gyrus import pattern_separate
    from ..brain.ca1 import CA1NoveltyDetector, NoveltyVerdict

    # Embedding only (no LLM)
    embedding = None
    try:
        from ..providers.base import get_providers
        _, embedder = get_providers(engine.config)
        embedding = await embedder.embed_single(text)
    except Exception:
        pass

    if embedding is None:
        embedding = _fallback_embedding(text, engine.config.embedding.dimensions)

    # DG + CA1
    sparse_code = pattern_separate(embedding)
    novelty_detector = CA1NoveltyDetector(
        novelty_threshold=engine.config.memory.novelty_threshold
    )

    nearest = []
    try:
        results = engine.vector_store.search(embedding, limit=5)
        nearest = [(r["id"], r["score"]) for r in results]
    except Exception:
        pass

    if not nearest:
        similar = engine.event_store.search_text(text, limit=3)
        for s in similar:
            nearest.append((s.id, 0.5))

    novelty = novelty_detector.detect(embedding, nearest)

    # Redundant → reinforce
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

    # Store (no feature extraction in fast mode)
    confidence = 0.5
    importance = 0.5

    event = EpisodicEvent(
        text=text,
        embedding=embedding,
        sparse_code=sparse_code,
        confidence=confidence,
        importance=importance,
        tags=[],
        source=source,
        metadata=metadata or {},
    )

    engine.event_store.upsert(event)
    engine.event_store._log_operation("observe", event.id, {
        "source": source,
        "verdict": novelty.verdict.value,
        "importance": importance,
        "tier": "fast",
    })

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

    _cache_put(text, event.id)

    return {
        "status": "encoded",
        "id": event.id,
        "verdict": novelty.verdict.value,
        "prediction_error": novelty.prediction_error,
        "confidence": event.confidence,
        "importance": importance,
        "sparse_code_bits": len(sparse_code),
        "tags": [],
        "timestamp": event.created_at,
        "tier": "fast",
    }


async def _encode_standard(
    engine, text: str, source: str, metadata: dict | None,
) -> dict:
    """Standard tier: embed + single combined LLM call + DG + CA1 + contradiction + store.

    Uses parse_with_importance() for 1 LLM call instead of 2.
    """
    from ..brain.dentate_gyrus import pattern_separate
    from ..brain.ca1 import CA1NoveltyDetector, NoveltyVerdict

    embedding = None
    parsed_features = {}
    importance = 0.5
    llm = None
    llm_available = False

    try:
        from ..providers.base import get_providers
        from ..brain.entorhinal import EntorhinalCortex

        llm, embedder = get_providers(engine.config)
        llm_available = True
        ec = EntorhinalCortex(llm, embedder)

        # Single combined LLM call for features + importance
        parsed, importance = await ec.parse_with_importance(text)

        embedding = parsed.embedding
        parsed_features = {
            "entities": parsed.entities,
            "topics": parsed.topics,
            "sentiment": parsed.sentiment,
            "summary": parsed.summary,
        }
    except Exception:
        pass

    if embedding is None:
        embedding = _fallback_embedding(text, engine.config.embedding.dimensions)

    # DG + CA1
    sparse_code = pattern_separate(embedding)
    novelty_detector = CA1NoveltyDetector(
        novelty_threshold=engine.config.memory.novelty_threshold
    )

    nearest = []
    try:
        results = engine.vector_store.search(embedding, limit=5)
        nearest = [(r["id"], r["score"]) for r in results]
    except Exception:
        pass

    if not nearest:
        similar = engine.event_store.search_text(text, limit=3)
        for s in similar:
            nearest.append((s.id, 0.5))

    novelty = novelty_detector.detect(embedding, nearest)

    # Contradiction check
    contradiction_pe_ceiling = engine.config.memory.novelty_threshold + 0.30
    if (
        novelty.closest_memory_id
        and novelty.prediction_error < contradiction_pe_ceiling
        and llm_available
    ):
        existing = engine.event_store.get(novelty.closest_memory_id)
        if existing and existing.text.strip() != text.strip():
            is_contradiction = await novelty_detector.detect_with_contradiction(
                text, existing.text, llm
            )
            if is_contradiction:
                novelty.verdict = NoveltyVerdict.CONTRADICTS

    # Redundant → reinforce
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

    # Contradiction handling — aggressively demote old memory so new fact ranks higher
    if novelty.verdict == NoveltyVerdict.CONTRADICTS and novelty.closest_memory_id:
        existing = engine.event_store.get(novelty.closest_memory_id)
        if existing:
            engine.event_store.update_confidence(existing.id, 0.15)

    # Update handling — demote old memory so new version ranks higher
    if novelty.verdict == NoveltyVerdict.UPDATE and novelty.closest_memory_id:
        existing = engine.event_store.get(novelty.closest_memory_id)
        if existing:
            new_conf = max(0.2, existing.confidence * 0.6)
            engine.event_store.update_confidence(existing.id, new_conf)

    # Confidence based on verdict
    if novelty.verdict == NoveltyVerdict.CONTRADICTS:
        confidence = 0.75  # High confidence — contradictions are intentional corrections
    elif novelty.verdict == NoveltyVerdict.UPDATE:
        confidence = 0.65  # Updates should outrank the old version
    else:
        confidence = 0.5

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

    engine.event_store.upsert(event)
    engine.event_store._log_operation("observe", event.id, {
        "source": source,
        "verdict": novelty.verdict.value,
        "importance": importance,
        "tier": "standard",
    })

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

    _cache_put(text, event.id)

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


async def _encode_full(
    engine, text: str, source: str, metadata: dict | None,
) -> dict:
    """Full tier: embed + separate LLM calls + DG + CA1 + contradiction + store.

    Original behavior with parallel parse() + score_importance() (2 LLM calls).
    """
    from ..brain.dentate_gyrus import pattern_separate
    from ..brain.ca1 import CA1NoveltyDetector, NoveltyVerdict

    embedding = None
    parsed_features = {}
    importance = 0.5
    llm = None
    llm_available = False

    try:
        from ..providers.base import get_providers
        from ..brain.entorhinal import EntorhinalCortex
        from ..brain.amygdala import Amygdala

        llm, embedder = get_providers(engine.config)
        llm_available = True
        ec = EntorhinalCortex(llm, embedder)
        amygdala = Amygdala(llm, threshold=engine.config.memory.importance_threshold)

        # Parallel: parse (1 LLM call) + importance (1 LLM call)
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

    if embedding is None:
        embedding = _fallback_embedding(text, engine.config.embedding.dimensions)

    # DG + CA1
    sparse_code = pattern_separate(embedding)
    novelty_detector = CA1NoveltyDetector(
        novelty_threshold=engine.config.memory.novelty_threshold
    )

    nearest = []
    try:
        results = engine.vector_store.search(embedding, limit=5)
        nearest = [(r["id"], r["score"]) for r in results]
    except Exception:
        pass

    if not nearest:
        similar = engine.event_store.search_text(text, limit=3)
        for s in similar:
            nearest.append((s.id, 0.5))

    novelty = novelty_detector.detect(embedding, nearest)

    # Contradiction check
    contradiction_pe_ceiling = engine.config.memory.novelty_threshold + 0.30
    if (
        novelty.closest_memory_id
        and novelty.prediction_error < contradiction_pe_ceiling
        and llm_available
    ):
        existing = engine.event_store.get(novelty.closest_memory_id)
        if existing and existing.text.strip() != text.strip():
            is_contradiction = await novelty_detector.detect_with_contradiction(
                text, existing.text, llm
            )
            if is_contradiction:
                novelty.verdict = NoveltyVerdict.CONTRADICTS

    # Redundant → reinforce
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

    # Contradiction handling
    if novelty.verdict == NoveltyVerdict.CONTRADICTS and novelty.closest_memory_id:
        existing = engine.event_store.get(novelty.closest_memory_id)
        if existing:
            new_conf = max(0.1, existing.confidence * 0.5)
            engine.event_store.update_confidence(existing.id, new_conf)

    if novelty.verdict == NoveltyVerdict.CONTRADICTS:
        confidence = 0.6
    elif novelty.verdict == NoveltyVerdict.UPDATE:
        confidence = 0.55
    else:
        confidence = 0.5

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

    engine.event_store.upsert(event)
    engine.event_store._log_operation("observe", event.id, {
        "source": source,
        "verdict": novelty.verdict.value,
        "importance": importance,
        "tier": "full",
    })

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

    _cache_put(text, event.id)

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


def encode_batch(
    engine,
    texts: list[str],
    source: str = "cli",
    tier: str | None = None,
) -> list[dict]:
    """Synchronous wrapper for batch encoding."""
    return asyncio.run(_encode_batch_async(engine, texts, source, tier))


async def _encode_batch_async(
    engine, texts: list[str], source: str, tier: str | None,
) -> list[dict]:
    """Batch encoding: shared LLM call + batch embedding, individual CA1 per item."""
    from ..brain.dentate_gyrus import pattern_separate, batch_pattern_separate
    from ..brain.ca1 import CA1NoveltyDetector, NoveltyVerdict

    if not texts:
        return []

    effective_tier = tier or getattr(engine.config.memory, "observe_tier", "standard")

    # --- Dedup pass: split into cached vs new ---
    results: list[dict | None] = [None] * len(texts)
    new_indices: list[int] = []
    new_texts: list[str] = []

    for i, text in enumerate(texts):
        cached_id = _cache_get(text)
        if cached_id is not None:
            existing = engine.event_store.get(cached_id)
            if existing is not None:
                engine.event_store.mark_accessed(cached_id)
                new_conf = min(1.0, existing.confidence + 0.05)
                engine.event_store.update_confidence(cached_id, new_conf)
                results[i] = {
                    "status": "redundant",
                    "verdict": "redundant",
                    "prediction_error": 0.0,
                    "closest_memory": cached_id,
                    "explanation": "Exact text match (dedup cache hit).",
                }
                continue
        new_indices.append(i)
        new_texts.append(text)

    if not new_texts:
        return [r for r in results if r is not None]

    # --- Batch embedding + optional LLM ---
    embeddings: list[list[float]] = []
    all_features: list[dict] = []
    all_importances: list[float] = []
    llm = None
    llm_available = False

    if effective_tier == "fast":
        # Embed only, no LLM
        try:
            from ..providers.base import get_providers
            _, embedder = get_providers(engine.config)
            embeddings = await embedder.embed(new_texts)
        except Exception:
            pass
        all_features = [{}] * len(new_texts)
        all_importances = [0.5] * len(new_texts)
    elif effective_tier == "standard":
        # Combined LLM call
        try:
            from ..providers.base import get_providers
            from ..brain.entorhinal import EntorhinalCortex

            llm, embedder = get_providers(engine.config)
            llm_available = True
            ec = EntorhinalCortex(llm, embedder)
            parsed_results = await ec.parse_batch_with_importance(new_texts)
            for parsed, imp in parsed_results:
                embeddings.append(parsed.embedding)
                all_features.append({
                    "entities": parsed.entities,
                    "topics": parsed.topics,
                    "sentiment": parsed.sentiment,
                    "summary": parsed.summary,
                })
                all_importances.append(imp)
        except Exception:
            pass
    else:
        # Full tier: individual LLM calls (parallel)
        try:
            from ..providers.base import get_providers
            from ..brain.entorhinal import EntorhinalCortex
            from ..brain.amygdala import Amygdala

            llm, embedder = get_providers(engine.config)
            llm_available = True
            ec = EntorhinalCortex(llm, embedder)
            amygdala = Amygdala(llm, threshold=engine.config.memory.importance_threshold)

            tasks = []
            for t in new_texts:
                tasks.append(asyncio.create_task(ec.parse(t)))
                tasks.append(asyncio.create_task(amygdala.score_importance(t)))

            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            for j in range(0, len(gathered), 2):
                parsed = gathered[j] if not isinstance(gathered[j], Exception) else None
                imp = gathered[j + 1] if not isinstance(gathered[j + 1], Exception) else 0.5
                if parsed is not None:
                    embeddings.append(parsed.embedding)
                    all_features.append({
                        "entities": parsed.entities,
                        "topics": parsed.topics,
                        "sentiment": parsed.sentiment,
                        "summary": parsed.summary,
                    })
                else:
                    embeddings.append([])
                    all_features.append({})
                all_importances.append(float(imp) if not isinstance(imp, Exception) else 0.5)
        except Exception:
            pass

    # Fill missing embeddings with fallbacks
    while len(embeddings) < len(new_texts):
        embeddings.append([])
    while len(all_features) < len(new_texts):
        all_features.append({})
    while len(all_importances) < len(new_texts):
        all_importances.append(0.5)

    for j in range(len(new_texts)):
        if not embeddings[j]:
            embeddings[j] = _fallback_embedding(new_texts[j], engine.config.embedding.dimensions)

    # --- Batch DG ---
    sparse_codes = batch_pattern_separate(embeddings)

    # --- Individual CA1 per item ---
    novelty_detector = CA1NoveltyDetector(
        novelty_threshold=engine.config.memory.novelty_threshold
    )

    for j, text in enumerate(new_texts):
        idx = new_indices[j]
        embedding = embeddings[j]
        sparse_code = sparse_codes[j]
        importance = all_importances[j]
        features = all_features[j]

        nearest = []
        try:
            search_results = engine.vector_store.search(embedding, limit=5)
            nearest = [(r["id"], r["score"]) for r in search_results]
        except Exception:
            pass
        if not nearest:
            similar = engine.event_store.search_text(text, limit=3)
            for s in similar:
                nearest.append((s.id, 0.5))

        novelty = novelty_detector.detect(embedding, nearest)

        # Contradiction check (skip for fast tier)
        if effective_tier != "fast":
            contradiction_pe_ceiling = engine.config.memory.novelty_threshold + 0.30
            if (
                novelty.closest_memory_id
                and novelty.prediction_error < contradiction_pe_ceiling
                and llm_available
            ):
                existing = engine.event_store.get(novelty.closest_memory_id)
                if existing and existing.text.strip() != text.strip():
                    is_contradiction = await novelty_detector.detect_with_contradiction(
                        text, existing.text, llm
                    )
                    if is_contradiction:
                        novelty.verdict = NoveltyVerdict.CONTRADICTS

        if novelty.verdict == NoveltyVerdict.REDUNDANT:
            if novelty.closest_memory_id:
                engine.event_store.mark_accessed(novelty.closest_memory_id)
                existing = engine.event_store.get(novelty.closest_memory_id)
                if existing:
                    new_conf = min(1.0, existing.confidence + 0.05)
                    engine.event_store.update_confidence(existing.id, new_conf)
            results[idx] = {
                "status": "redundant",
                "verdict": novelty.verdict.value,
                "prediction_error": novelty.prediction_error,
                "closest_memory": novelty.closest_memory_id,
                "explanation": novelty.explanation,
            }
            continue

        if novelty.verdict == NoveltyVerdict.CONTRADICTS and novelty.closest_memory_id:
            existing = engine.event_store.get(novelty.closest_memory_id)
            if existing:
                new_conf = max(0.1, existing.confidence * 0.5)
                engine.event_store.update_confidence(existing.id, new_conf)

        if novelty.verdict == NoveltyVerdict.CONTRADICTS:
            confidence = 0.6
        elif novelty.verdict == NoveltyVerdict.UPDATE:
            confidence = 0.55
        else:
            confidence = 0.5

        tags = features.get("topics", [])
        event = EpisodicEvent(
            text=text,
            embedding=embedding,
            sparse_code=sparse_code,
            confidence=confidence,
            importance=importance,
            tags=tags,
            source=source,
            metadata={
                **{k: v for k, v in features.items() if k != "topics"},
            },
        )

        engine.event_store.upsert(event)
        engine.event_store._log_operation("observe", event.id, {
            "source": source,
            "verdict": novelty.verdict.value,
            "importance": importance,
            "tier": effective_tier,
        })

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

        _cache_put(text, event.id)

        results[idx] = {
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

    return [r for r in results if r is not None]


def _fallback_embedding(text: str, dimensions: int) -> list[float]:
    """Generate a deterministic pseudo-embedding when no provider is available."""
    h = hashlib.sha256(text.encode()).digest()
    seed = int.from_bytes(h[:8], "little")
    rng = random.Random(seed)
    return [rng.gauss(0, 0.1) for _ in range(dimensions)]
