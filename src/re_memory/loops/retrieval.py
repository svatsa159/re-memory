"""Retrieval Loop (Read Path)

Query → EC (parse + embed query)
  → First: check Schema summaries (Layer 4) — token cheap
  → If insufficient: query Knowledge Graph (Layer 3) — structured precision
  → If ambiguous: pull supporting Episodic events (Layer 2) — ground truth
  → CA3 pattern completion for attractor-based recall
  → Time-decay ranking via Rust core
  → Token budget management: inject only top-N within budget
  → Reconsolidation check on retrieved memories
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ..loops.encoding import _fallback_embedding


def retrieve(engine, query: str, max_tokens: int = 2000, limit: int = 10, **kwargs) -> dict:
    """Synchronous wrapper for the retrieval pipeline."""
    return asyncio.run(_retrieve_async(engine, query, max_tokens, limit, **kwargs))


async def _retrieve_async(
    engine, query: str, max_tokens: int, limit: int, **kwargs
) -> dict:
    """Tiered retrieval: Schema → KG → Episodic with CA3 pattern completion."""
    results = {
        "query": query,
        "memories": [],
        "sources": [],
        "token_budget_used": 0,
    }

    now = datetime.now(timezone.utc)

    # --- Generate query embedding ---
    query_embedding = None
    try:
        from ..providers.base import get_providers

        _, embedder = get_providers(engine.config)
        query_embedding = await embedder.embed_single(query)
    except Exception:
        query_embedding = _fallback_embedding(query, engine.config.embedding.dimensions)

    # --- Layer 4: Schema summaries (cheapest, most summarized) ---
    try:
        from ..storage.file_store import FileStore

        file_store = FileStore(engine.config.schema_path)
        schema_results = file_store.search(query)
        for sr in schema_results[:3]:
            content = sr["snippet"]
            results["memories"].append({
                "layer": "schema",
                "category": sr["category"],
                "content": content,
                "score": 1.0,  # Schema matches are highly relevant
            })
            results["sources"].append("schema")
            results["token_budget_used"] += _estimate_tokens(content)
    except Exception:
        pass

    # --- Layer 3: Knowledge graph search ---
    try:
        kg_results = engine.graph_store.search_triples(query, limit=5)
        for triple in kg_results:
            content = f"{triple['subject']} {triple['predicate']} {triple['object']}"
            results["memories"].append({
                "layer": "semantic",
                "content": content,
                "confidence": triple.get("confidence", 1.0),
                "score": triple.get("confidence", 1.0),
            })
            results["sources"].append("semantic")
            results["token_budget_used"] += _estimate_tokens(content)
    except Exception:
        pass

    # --- Layer 2: Episodic events with CA3 pattern completion ---
    episodic_memories = []

    # Vector search (primary retrieval method)
    try:
        vector_results = engine.vector_store.search(
            query_embedding, limit=limit, score_threshold=0.3
        )
        for vr in vector_results:
            event = engine.event_store.get(vr["id"])
            if event:
                episodic_memories.append((event, vr["score"]))
    except Exception:
        pass

    # CA3 pattern completion on stored sparse codes (if we have matches but want more)
    if len(episodic_memories) < limit:
        try:
            all_events = engine.event_store.recent(limit=100)
            stored_embeddings = [e.embedding for e in all_events if e.embedding]
            seen_ids = {em[0].id for em in episodic_memories}

            if stored_embeddings and query_embedding:
                from ..brain.ca3 import pattern_complete

                matches = pattern_complete(
                    query_embedding,
                    stored_embeddings,
                    beta=8.0,
                    top_k=limit,
                )
                for idx, prob in matches:
                    if idx < len(all_events):
                        event = all_events[idx]
                        if event.id not in seen_ids and prob > 0.01:
                            episodic_memories.append((event, prob))
                            seen_ids.add(event.id)
        except Exception:
            pass

    # Text search fallback
    if not episodic_memories:
        text_results = engine.event_store.search_text(query, limit=limit)
        for event in text_results:
            episodic_memories.append((event, 0.5))

    # --- Time-decay ranking ---
    scored_memories = []
    for event, raw_score in episodic_memories:
        last_accessed = datetime.fromisoformat(event.last_accessed_at)
        elapsed_hours = (now - last_accessed).total_seconds() / 3600

        try:
            from re_memory._core import time_decay_score

            final_score = time_decay_score(
                raw_score, elapsed_hours, recency_weight=0.3, half_life_hours=168.0
            )
        except ImportError:
            import math

            decay = math.exp(-(0.693 * elapsed_hours) / 168.0) if elapsed_hours > 0 else 1.0
            final_score = 0.7 * raw_score + 0.3 * decay

        scored_memories.append((event, final_score, raw_score))

    # Sort by final score descending
    scored_memories.sort(key=lambda x: -x[1])

    # Apply token budget
    for event, score, raw_score in scored_memories:
        content = event.text
        tokens = _estimate_tokens(content)

        if results["token_budget_used"] + tokens > max_tokens:
            continue

        # Record access (reconsolidation trigger in Phase 6)
        engine.event_store.mark_accessed(event.id)

        results["memories"].append({
            "layer": "episodic",
            "id": event.id,
            "content": content,
            "confidence": event.confidence,
            "importance": event.importance,
            "score": round(score, 4),
            "raw_similarity": round(raw_score, 4),
            "access_count": event.access_count + 1,
            "created_at": event.created_at,
        })
        results["sources"].append("episodic")
        results["token_budget_used"] += tokens

    results["total"] = len(results["memories"])
    results["token_budget_max"] = max_tokens
    return results


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 chars per token)."""
    return max(1, len(text) // 4)
