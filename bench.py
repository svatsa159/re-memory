#!/usr/bin/env python3
"""Performance benchmark for re-memory — profiles each command with realistic statements.

Measures wall-clock time and breaks down into sub-operations where possible.
"""

import time
import json
import sys
import statistics
from pathlib import Path
from contextlib import contextmanager

# ── Realistic test statements ────────────────────────────────────────────────
STATEMENTS = {
    "personal_facts": [
        "Lucian graduated from IIT Bombay with a B.Tech in Computer Science in 2019",
        "His preferred stack is Python + FastAPI + PostgreSQL for backend services",
        "He runs Arch Linux on his ThinkPad X1 Carbon with i3wm as the window manager",
        "Lucian's SSH key fingerprint for GitHub is SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8",
    ],
    "project_context": [
        "The re-memory project uses a hippocampal architecture: DG for pattern separation, CA3 for autoassociative recall, CA1 for novelty detection",
        "FalkorDB stores knowledge graph triples extracted by the LLM during consolidation",
        "Qdrant collection 're_memory_embeddings' uses cosine distance with 1024-dim mxbai-embed-large vectors",
        "The consolidation daemon runs every 24 hours with Ebbinghaus decay rate 0.1",
    ],
    "contradictions": [
        ("The deployment target is AWS us-east-1 on EKS", "The deployment target is GCP us-central1 on GKE"),
        ("The team uses Jira for project management", "The team switched to Linear for project management"),
    ],
    "edge_cases": [
        "SELECT * FROM users WHERE id = 1; DROP TABLE users; --",  # SQL injection lookalike
        "🧠 The émojis and üñîcödé characters should encode properly into the embedding space",
        "A" * 500,  # long repetitive input
        "",  # empty string
    ],
}


class Timer:
    def __init__(self):
        self.laps = {}

    @contextmanager
    def lap(self, name):
        t0 = time.perf_counter()
        yield
        self.laps[name] = time.perf_counter() - t0

    def total(self):
        return sum(self.laps.values())


def fmt_ms(seconds):
    if seconds < 0.001:
        return f"{seconds*1_000_000:.0f}μs"
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    return f"{seconds:.2f}s"


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_row(label, duration, detail=""):
    bar_len = min(int(duration * 20), 50)  # 1 char per 50ms
    bar = "█" * bar_len + "░" * max(0, 10 - bar_len)
    detail_str = f"  ({detail})" if detail else ""
    print(f"  {label:<40} {fmt_ms(duration):>8}  {bar}{detail_str}")


def clean_stores(engine):
    """Wipe all stores for a clean benchmark."""
    # SQLite
    try:
        for ev in engine.event_store.all():
            engine.event_store.delete(ev.id)
    except Exception:
        pass

    # Qdrant — delete and recreate collection
    try:
        engine.vector_store.client.delete_collection(engine.vector_store.collection_name)
        engine.vector_store.init()
    except Exception:
        pass

    # FalkorDB
    try:
        engine.graph_store.client.execute_command("GRAPH.DELETE", engine.graph_store.graph_name)
    except Exception:
        pass


def bench_init(engine):
    """Benchmark: init command."""
    print_header("1. INIT")
    t = Timer()

    with t.lap("full_init"):
        result = engine.init()

    print_row("engine.init()", t.laps["full_init"])
    for k, v in result.items():
        print(f"    {k}: {v}")
    return t


def bench_observe(engine):
    """Benchmark: observe command with various statement types."""
    print_header("2. OBSERVE (encoding pipeline)")
    timings = []

    # Personal facts
    print(f"\n  ── Personal facts ──")
    for stmt in STATEMENTS["personal_facts"]:
        t = Timer()
        with t.lap("observe"):
            result = engine.observe(stmt, source="benchmark")
        timings.append(t.laps["observe"])
        verdict = result.get("verdict", result.get("status", "?"))
        importance = result.get("importance", "?")
        print_row(f"  [{verdict}] {stmt[:50]}...", t.laps["observe"], f"imp={importance}")

    # Project context
    print(f"\n  ── Project context ──")
    for stmt in STATEMENTS["project_context"]:
        t = Timer()
        with t.lap("observe"):
            result = engine.observe(stmt, source="benchmark")
        timings.append(t.laps["observe"])
        verdict = result.get("verdict", result.get("status", "?"))
        importance = result.get("importance", "?")
        print_row(f"  [{verdict}] {stmt[:50]}...", t.laps["observe"], f"imp={importance}")

    # Edge cases
    print(f"\n  ── Edge cases ──")
    for stmt in STATEMENTS["edge_cases"]:
        label = repr(stmt[:50]) + ("..." if len(stmt) > 50 else "")
        t = Timer()
        try:
            with t.lap("observe"):
                result = engine.observe(stmt, source="benchmark")
            timings.append(t.laps["observe"])
            verdict = result.get("verdict", result.get("status", "?"))
            print_row(f"  [{verdict}] {label}", t.laps["observe"])
        except Exception as e:
            print_row(f"  [ERROR] {label}", 0, str(e)[:60])

    print(f"\n  ── Observe Summary ──")
    if timings:
        print(f"    Mean:   {fmt_ms(statistics.mean(timings))}")
        print(f"    Median: {fmt_ms(statistics.median(timings))}")
        print(f"    Min:    {fmt_ms(min(timings))}")
        print(f"    Max:    {fmt_ms(max(timings))}")
        print(f"    Total:  {fmt_ms(sum(timings))} for {len(timings)} observations")
    return timings


def bench_recall(engine):
    """Benchmark: recall (retrieval pipeline)."""
    print_header("3. RECALL (retrieval pipeline)")
    queries = [
        "What programming languages does Lucian use?",
        "What is the deployment infrastructure?",
        "How does the memory consolidation work?",
        "What hardware does Lucian use for development?",
        "Tell me about the knowledge graph storage",
    ]
    timings = []
    for q in queries:
        t = Timer()
        with t.lap("recall"):
            result = engine.recall(q)
        timings.append(t.laps["recall"])
        n_results = result.get("total", len(result.get("memories", [])))
        top_match = ""
        if result.get("memories"):
            top_match = result["memories"][0].get("content", "")[:40]
        print_row(f"  Q: {q[:45]}...", t.laps["recall"], f"{n_results} results")
        if top_match:
            print(f"      → top: \"{top_match}...\"")

    print(f"\n  ── Recall Summary ──")
    if timings:
        print(f"    Mean:   {fmt_ms(statistics.mean(timings))}")
        print(f"    Median: {fmt_ms(statistics.median(timings))}")
        print(f"    Min:    {fmt_ms(min(timings))}")
        print(f"    Max:    {fmt_ms(max(timings))}")
    return timings


def bench_contradiction(engine):
    """Benchmark: contradiction detection (observe conflicting facts)."""
    print_header("4. CONTRADICTION DETECTION")
    timings = []
    for original, update in STATEMENTS["contradictions"]:
        # Observe original
        t1 = Timer()
        with t1.lap("original"):
            r1 = engine.observe(original, source="benchmark")

        # Observe contradicting fact
        t2 = Timer()
        with t2.lap("contradiction"):
            r2 = engine.observe(update, source="benchmark")

        timings.append(t1.laps["original"] + t2.laps["contradiction"])
        v1 = r1.get("verdict", "?")
        v2 = r2.get("verdict", "?")
        print_row(f"  Original: {original[:40]}...", t1.laps["original"], f"[{v1}]")
        print_row(f"  Update:   {update[:40]}...", t2.laps["contradiction"], f"[{v2}]")
        print()
    return timings


def bench_search(engine):
    """Benchmark: text search."""
    print_header("5. SEARCH (text search)")
    queries = ["Python", "deployment", "memory", "Lucian", "GKE"]
    timings = []
    for q in queries:
        t = Timer()
        with t.lap("search"):
            results = engine.search(q)
        timings.append(t.laps["search"])
        print_row(f"  search(\"{q}\")", t.laps["search"], f"{len(results)} results")

    print(f"\n  ── Search Summary ──")
    if timings:
        print(f"    Mean:   {fmt_ms(statistics.mean(timings))}")
        print(f"    Median: {fmt_ms(statistics.median(timings))}")
    return timings


def bench_consolidate(engine):
    """Benchmark: consolidation loop."""
    print_header("6. CONSOLIDATE")
    t = Timer()
    with t.lap("consolidate"):
        result = engine.consolidate()
    print_row("  consolidate()", t.laps["consolidate"])
    for k, v in result.items():
        print(f"    {k}: {v}")
    return [t.laps["consolidate"]]


def bench_status(engine):
    """Benchmark: status command."""
    print_header("7. STATUS")
    t = Timer()
    with t.lap("status"):
        result = engine.status()
    print_row("  status()", t.laps["status"])
    print(f"    stores: {result.get('stores', {})}")
    print(f"    counts: {result.get('counts', {})}")
    return [t.laps["status"]]


def bench_inspect_history(engine):
    """Benchmark: inspect + history."""
    print_header("8. INSPECT & HISTORY")
    timings = []

    # History
    t = Timer()
    with t.lap("history"):
        entries = engine.history(limit=20)
    timings.append(t.laps["history"])
    print_row(f"  history(limit=20)", t.laps["history"], f"{len(entries)} entries")

    # Inspect first available memory
    if entries:
        mid = entries[0].get("id", "")
        t2 = Timer()
        with t2.lap("inspect"):
            detail = engine.inspect(mid)
        timings.append(t2.laps["inspect"])
        print_row(f"  inspect({mid[:12]}...)", t2.laps["inspect"])

    return timings


def bench_forget(engine):
    """Benchmark: forget (cascade delete)."""
    print_header("9. FORGET (cascade delete)")
    # Create a memory to forget
    r = engine.observe("Temporary benchmark memory to be forgotten", source="benchmark")
    mid = r.get("id", "")

    t = Timer()
    with t.lap("forget"):
        result = engine.forget(mid)
    print_row(f"  forget({mid[:12]}...)", t.laps["forget"])
    deleted = result.get("deleted_from", [])
    print(f"    deleted_from: {deleted}")
    return [t.laps["forget"]]


def bench_export_import(engine, tmp_dir="/tmp/re_memory_bench"):
    """Benchmark: export + import."""
    print_header("10. EXPORT & IMPORT")
    timings = []

    export_path = Path(tmp_dir) / "bench_export.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    t = Timer()
    with t.lap("export"):
        result = engine.export_data(export_path)
    timings.append(t.laps["export"])
    n = result.get("exported", 0)
    size = export_path.stat().st_size if export_path.exists() else 0
    print_row(f"  export({n} events)", t.laps["export"], f"{size/1024:.1f}KB")

    t2 = Timer()
    with t2.lap("import"):
        result = engine.import_data(export_path)
    timings.append(t2.laps["import"])
    print_row(f"  import({result.get('imported', 0)} events)", t2.laps["import"])

    return timings


def bench_embedding_breakdown():
    """Benchmark: raw embedding generation vs LLM parsing (isolate bottlenecks)."""
    print_header("11. PIPELINE BREAKDOWN (isolate bottlenecks)")

    from re_memory.config import load_config
    config = load_config()

    # Raw embedding
    try:
        import httpx
        test_text = "Lucian graduated from IIT Bombay with a B.Tech in Computer Science"

        t = Timer()
        with t.lap("embedding_api"):
            resp = httpx.post(
                f"{config.embedding.base_url}/api/embed",
                json={"model": config.embedding.model, "input": test_text},
                timeout=30,
            )
            resp.raise_for_status()
        print_row("  Ollama embed (single)", t.laps["embedding_api"])

        # Batch embedding
        batch = [s for s in STATEMENTS["personal_facts"]]
        t2 = Timer()
        with t2.lap("embedding_batch"):
            resp = httpx.post(
                f"{config.embedding.base_url}/api/embed",
                json={"model": config.embedding.model, "input": batch},
                timeout=30,
            )
            resp.raise_for_status()
        print_row(f"  Ollama embed (batch of {len(batch)})", t2.laps["embedding_batch"])
        per_item = t2.laps["embedding_batch"] / len(batch)
        print(f"    → per item in batch: {fmt_ms(per_item)}")
    except Exception as e:
        print(f"  Embedding benchmark failed: {e}")

    # LLM call (DeepSeek)
    try:
        from re_memory.providers.base import get_providers
        llm, _ = get_providers(config)
        import asyncio

        t3 = Timer()
        with t3.lap("llm_parse"):
            async def _test_llm():
                return await llm.complete(
                    "Extract entities and topics from: 'Lucian uses Python and FastAPI on Arch Linux'. Return JSON with keys: entities, topics, sentiment, summary",
                    max_tokens=200,
                )
            result = asyncio.run(_test_llm())
        print_row("  LLM parse (DeepSeek)", t3.laps["llm_parse"])

        t4 = Timer()
        with t4.lap("llm_importance"):
            async def _test_importance():
                return await llm.complete(
                    "Rate the importance of this information on a scale 0.0-1.0: 'Lucian graduated from IIT Bombay'. Return only the number.",
                    max_tokens=10,
                )
            result = asyncio.run(_test_importance())
        print_row("  LLM importance score", t4.laps["llm_importance"])

        t5 = Timer()
        with t5.lap("llm_contradiction"):
            async def _test_contradiction():
                return await llm.complete(
                    "Do these two statements contradict each other? Answer YES or NO.\n1: 'Team uses Jira'\n2: 'Team switched to Linear'",
                    max_tokens=10,
                )
            result = asyncio.run(_test_contradiction())
        print_row("  LLM contradiction check", t5.laps["llm_contradiction"])
    except Exception as e:
        print(f"  LLM benchmark failed: {e}")

    # Qdrant search
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=config.storage.qdrant_url, timeout=5)

        # Generate a query vector
        resp = httpx.post(
            f"{config.embedding.base_url}/api/embed",
            json={"model": config.embedding.model, "input": "test query"},
            timeout=30,
        )
        qvec = resp.json()["embeddings"][0]

        t6 = Timer()
        with t6.lap("qdrant_search"):
            client.query_points(
                collection_name="re_memory_embeddings",
                query=qvec,
                limit=10,
            )
        print_row("  Qdrant vector search", t6.laps["qdrant_search"])
        client.close()
    except Exception as e:
        print(f"  Qdrant benchmark failed: {e}")

    # SQLite text search
    try:
        from re_memory.storage.event_store import EventStore
        es = EventStore(config.sqlite_path)

        t7 = Timer()
        with t7.lap("sqlite_search"):
            es.search_text("Python", limit=10)
        print_row("  SQLite text search", t7.laps["sqlite_search"])
        es.close()
    except Exception as e:
        print(f"  SQLite benchmark failed: {e}")


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           re-memory Performance Benchmark                          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    from re_memory.engine import MemoryEngine
    engine = MemoryEngine()

    # Clean slate
    print("\nCleaning stores for fresh benchmark...")
    clean_stores(engine)
    print("Done.\n")

    all_timings = {}

    # Run all benchmarks
    bench_init(engine)
    all_timings["observe"] = bench_observe(engine)
    all_timings["recall"] = bench_recall(engine)
    all_timings["contradiction"] = bench_contradiction(engine)
    all_timings["search"] = bench_search(engine)
    all_timings["consolidate"] = bench_consolidate(engine)
    all_timings["status"] = bench_status(engine)
    all_timings["inspect_history"] = bench_inspect_history(engine)
    all_timings["forget"] = bench_forget(engine)
    all_timings["export_import"] = bench_export_import(engine)

    # Pipeline breakdown
    bench_embedding_breakdown()

    # Final summary
    print_header("SUMMARY")
    print(f"\n  {'Command':<30} {'Total':>10} {'Avg':>10} {'Calls':>6}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*6}")
    for name, times in sorted(all_timings.items(), key=lambda x: sum(x[1]), reverse=True):
        total = sum(times)
        avg = statistics.mean(times) if times else 0
        print(f"  {name:<30} {fmt_ms(total):>10} {fmt_ms(avg):>10} {len(times):>6}")

    grand_total = sum(sum(t) for t in all_timings.values())
    print(f"\n  {'GRAND TOTAL':<30} {fmt_ms(grand_total):>10}")

    engine.close()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
