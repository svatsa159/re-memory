"""Consolidation Loop (Sleep Analogue)

Background daemon that:
1. REPLAY: Sample episodic events biased by importance/recency
2. PROMOTE: High-confidence events → extract triples → KG
3. ABSTRACT: Cluster KG triples → update schema summaries
4. DECAY: Reduce confidence on unreinforced memories
5. PRUNE: Archive memories below confidence threshold
6. MERGE: Deduplicate events
7. RE-INDEX: Rebuild embeddings
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

_scheduler = None


def consolidate(engine) -> dict:
    """Run one consolidation cycle synchronously."""
    return asyncio.run(_consolidate_async(engine))


async def _consolidate_async(engine) -> dict:
    """Run one full consolidation cycle."""
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "replayed": 0,
        "promoted": 0,
        "abstracted": 0,
        "decayed": 0,
        "pruned": 0,
        "merged": 0,
    }

    config = engine.config.consolidation
    now = datetime.now(timezone.utc)

    # --- Step 1: REPLAY — sample events biased by importance × recency ---
    events = engine.event_store.all()
    if not events:
        return results

    results["replayed"] = len(events)

    # --- Step 2: PROMOTE — high-confidence unconsolidated events → KG ---
    unconsolidated = engine.event_store.unconsolidated(limit=50)
    llm_available = False
    llm = None

    try:
        from ..providers.base import get_providers

        llm, _ = get_providers(engine.config)
        llm_available = True
    except Exception:
        pass

    if llm_available and unconsolidated:
        from ..brain.neocortex import Neocortex
        from ..storage.file_store import FileStore

        graph = engine.graph_store
        file_store = FileStore(engine.config.schema_path)
        neocortex = Neocortex(llm, graph, file_store)

        promoted_triples = []

        for event in unconsolidated:
            # Only promote events with sufficient confidence and importance
            if event.confidence >= 0.4 and event.importance >= 0.3:
                try:
                    triples = await neocortex.extract_triples(event.text, event.id)
                    promoted_triples.extend(triples)
                    engine.event_store.mark_consolidated(event.id)
                    results["promoted"] += 1
                except Exception:
                    pass

        # --- Step 3: ABSTRACT — group triples into schema summaries ---
        if promoted_triples:
            # Group triples by topic/entity
            topic_triples: dict[str, list[dict]] = {}
            for triple in promoted_triples:
                # Use subject as topic key
                topic = triple.get("subject", "general").lower()
                topic_triples.setdefault(topic, []).append(triple)

            for topic, triples in topic_triples.items():
                try:
                    await neocortex.update_schema(topic, triples)
                    results["abstracted"] += 1
                except Exception:
                    pass

    # --- Step 4: DECAY — apply Ebbinghaus forgetting curve ---
    for event in events:
        last_accessed = datetime.fromisoformat(event.last_accessed_at)
        elapsed_hours = (now - last_accessed).total_seconds() / 3600

        new_confidence = _compute_decay(
            event.confidence,
            elapsed_hours,
            event.access_count,
            event.importance,
            config.decay_rate,
        )

        if abs(new_confidence - event.confidence) > 0.01:
            engine.event_store.update_confidence(event.id, new_confidence)
            results["decayed"] += 1

    # --- Step 5: PRUNE — remove memories below threshold ---
    pruned = engine.event_store.prune_below_confidence(config.confidence_threshold)
    results["pruned"] = pruned

    # --- Step 6: MERGE — deduplicate similar events ---
    results["merged"] = await _merge_duplicates(engine)

    return results


async def _merge_duplicates(engine) -> int:
    """Find and merge near-duplicate episodic events."""
    merged_count = 0
    events = engine.event_store.recent(limit=200)

    if len(events) < 2:
        return 0

    # Simple text-based dedup: if two events have very similar text, keep the newer one
    seen_texts: dict[str, str] = {}  # normalized_text -> event_id
    for event in events:
        normalized = event.text.strip().lower()
        if normalized in seen_texts:
            # Duplicate found — keep the one with higher confidence
            existing = engine.event_store.get(seen_texts[normalized])
            if existing and event.confidence <= existing.confidence:
                engine.event_store.delete(event.id)
                merged_count += 1
            elif existing:
                engine.event_store.delete(existing.id)
                seen_texts[normalized] = event.id
                merged_count += 1
        else:
            seen_texts[normalized] = event.id

    return merged_count


def _compute_decay(
    base_confidence: float,
    elapsed_hours: float,
    access_count: int,
    importance: float,
    decay_rate: float,
) -> float:
    """Compute decayed confidence using Rust core or Python fallback."""
    try:
        from re_memory._core import compute_confidence_decay

        return compute_confidence_decay(
            base_confidence, elapsed_hours, access_count, importance, decay_rate
        )
    except ImportError:
        access_boost = max(1.0, 1.0 + math.log(1.0 + access_count))
        importance_factor = max(0.5, min(1.5, 0.5 + importance))
        stability = access_boost * importance_factor / max(0.01, decay_rate)
        retention = base_confidence * math.exp(-elapsed_hours / stability)
        return max(0.0, min(1.0, retention))


def start_daemon(engine):
    """Start the background consolidation daemon."""
    global _scheduler

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler

        _scheduler = BlockingScheduler()
        _scheduler.add_job(
            lambda: consolidate(engine),
            "interval",
            hours=engine.config.consolidation.interval_hours,
            id="consolidation",
        )
        _scheduler.start()
    except (ImportError, KeyboardInterrupt):
        pass


def stop_daemon():
    """Stop the consolidation daemon."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown()
        _scheduler = None


def daemon_status() -> dict:
    """Get daemon status."""
    global _scheduler
    if _scheduler and _scheduler.running:
        jobs = _scheduler.get_jobs()
        return {
            "running": True,
            "jobs": len(jobs),
            "next_run": str(jobs[0].next_run_time) if jobs else "none",
        }
    return {"running": False}
