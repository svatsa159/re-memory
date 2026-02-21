"""Evo-Memory benchmark runner.

Evo-Memory (arxiv.org/abs/2511.20857) tests whether accumulated memory
actually improves performance over time using a streaming task format.

Since this is a DeepMind paper-based benchmark, this runner implements
the streaming evaluation protocol synthetically:

1. Stream observations over time
2. Periodically evaluate retrieval quality
3. Measure whether consolidation improves retrieval over flat storage
4. Compare "with consolidation" vs "without consolidation" trajectories

The key insight: if consolidation works, retrieval quality should IMPROVE
or at least NOT DEGRADE as more memories accumulate.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from benchmarks.adapter import EngineAdapter, BenchmarkMetrics
from benchmarks.adapter.metrics import BenchmarkResult
from benchmarks.datasets.download import CACHE_DIR


BENCHMARK_NAME = "Evo-Memory"


@dataclass
class StreamingFact:
    """A fact in the streaming sequence with its evaluation query."""

    text: str
    query: str
    expected_answer: str
    category: str
    sequence_index: int


def _generate_streaming_facts(
    n_facts: int = 200,
    n_categories: int = 10,
    update_rate: float = 0.15,
) -> list[StreamingFact]:
    """Generate a stream of facts across categories, with periodic updates.

    Args:
        n_facts: Total number of facts to generate.
        n_categories: Number of distinct topic categories.
        update_rate: Fraction of facts that update a previous fact.
    """
    categories = [f"topic_{i}" for i in range(n_categories)]
    category_state: dict[str, dict[str, str]] = {c: {} for c in categories}

    # Templates for fact generation
    property_templates = {
        "leader": ("{cat} is led by {val}", "Who leads {cat}?"),
        "location": ("{cat} is based in {val}", "Where is {cat} based?"),
        "status": ("{cat} status is {val}", "What is the status of {cat}?"),
        "priority": ("{cat} priority is {val}", "What is the priority of {cat}?"),
        "tool": ("{cat} uses {val} as primary tool", "What tool does {cat} use?"),
    }
    properties = list(property_templates.keys())
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]
    locations = ["NYC", "London", "Tokyo", "Berlin", "Sydney", "Toronto", "Mumbai", "Seoul", "Paris", "Dublin"]
    statuses = ["active", "paused", "completed", "reviewing", "planning"]
    priorities = ["P0-critical", "P1-high", "P2-medium", "P3-low"]
    tools = ["PostgreSQL", "Redis", "Kafka", "Elasticsearch", "Neo4j", "MongoDB", "ClickHouse", "DynamoDB"]

    value_pools = {
        "leader": names,
        "location": locations,
        "status": statuses,
        "priority": priorities,
        "tool": tools,
    }

    facts = []
    for i in range(n_facts):
        cat = categories[i % n_categories]
        prop = properties[i % len(properties)]

        # Decide if this is an update
        is_update = random.random() < update_rate and prop in category_state[cat]

        pool = value_pools[prop]
        if is_update:
            # Pick a different value than the current one
            current = category_state[cat][prop]
            new_val = random.choice([v for v in pool if v != current])
        else:
            new_val = random.choice(pool)

        category_state[cat][prop] = new_val

        fact_template, query_template = property_templates[prop]
        text = fact_template.format(cat=cat, val=new_val)
        query = query_template.format(cat=cat)

        facts.append(StreamingFact(
            text=text,
            query=query,
            expected_answer=new_val,
            category=cat,
            sequence_index=i,
        ))

    return facts


def run_evolution_test(
    adapter: EngineAdapter,
    facts: list[StreamingFact],
    eval_interval: int = 20,
    consolidate_interval: int | None = 50,
    limit: int | None = None,
) -> BenchmarkResult:
    """Run the streaming evolution test.

    Streams facts and periodically evaluates retrieval quality to see
    if it improves, degrades, or stays stable over time.

    Args:
        adapter: Initialized EngineAdapter.
        facts: Stream of facts to ingest.
        eval_interval: Evaluate every N facts.
        consolidate_interval: Consolidate every N facts (None = never).
        limit: Max facts to process.
    """
    mode = "with_consolidation" if consolidate_interval else "no_consolidation"
    result = BenchmarkResult(benchmark_name=BENCHMARK_NAME, task_name=f"evolution_{mode}")
    adapter.reset()

    evaluation_points: list[dict] = []
    all_facts = facts[:limit] if limit else facts

    for i, fact in enumerate(tqdm(all_facts, desc=f"evo-memory-{mode}")):
        # Store the fact
        adapter.store(fact.text, source="evo_memory")

        # Consolidate periodically
        if consolidate_interval and (i + 1) % consolidate_interval == 0:
            adapter.consolidate()

        # Evaluate periodically
        if (i + 1) % eval_interval == 0:
            # Sample recent facts for evaluation
            eval_facts = all_facts[max(0, i - eval_interval):i + 1]
            predictions, references = [], []

            for ef in eval_facts[-10:]:  # Evaluate last 10 facts in window
                retrieved = adapter.query_text(ef.query)
                predictions.append(retrieved)
                references.append(ef.expected_answer)

            if predictions:
                metrics = BenchmarkMetrics.evaluate_qa_batch(predictions, references)
                contains_rate = sum(
                    BenchmarkMetrics.contains_match(p, r) for p, r in zip(predictions, references)
                ) / len(predictions)

                evaluation_points.append({
                    "step": i + 1,
                    "memories_stored": adapter.get_memory_count(),
                    "contains_rate": contains_rate,
                    "token_f1": metrics["token_f1"],
                    "exact_match": metrics["exact_match"],
                })

    result.samples = evaluation_points

    if evaluation_points:
        # Compute trajectory metrics
        contains_rates = [ep["contains_rate"] for ep in evaluation_points]
        f1_scores = [ep["token_f1"] for ep in evaluation_points]

        # Final accuracy
        result.add_metric("final_contains_rate", contains_rates[-1], count=len(evaluation_points))
        result.add_metric("final_token_f1", f1_scores[-1], count=len(evaluation_points))

        # Average across all checkpoints
        result.add_metric("avg_contains_rate", sum(contains_rates) / len(contains_rates))
        result.add_metric("avg_token_f1", sum(f1_scores) / len(f1_scores))

        # Trend: positive = improving, negative = degrading
        if len(contains_rates) >= 2:
            first_half = sum(contains_rates[:len(contains_rates)//2]) / (len(contains_rates)//2)
            second_half = sum(contains_rates[len(contains_rates)//2:]) / (len(contains_rates) - len(contains_rates)//2)
            trend = second_half - first_half
            result.add_metric("performance_trend", trend)

        # Peak and trough
        result.add_metric("peak_contains_rate", max(contains_rates))
        result.add_metric("trough_contains_rate", min(contains_rates))

    # Latency
    store_latencies = adapter.get_latencies("store")
    if store_latencies:
        for k, v in BenchmarkMetrics.latency_stats(store_latencies).items():
            result.add_metric(f"store_{k}", v)

    query_latencies = adapter.get_latencies("query")
    if query_latencies:
        for k, v in BenchmarkMetrics.latency_stats(query_latencies).items():
            result.add_metric(f"query_{k}", v)

    return result


def run(
    adapter: EngineAdapter,
    data_dir: Path | None = None,
    limit: int | None = None,
    tasks: list[str] | None = None,
    n_facts: int = 200,
    eval_interval: int = 20,
) -> list[BenchmarkResult]:
    """Run Evo-Memory benchmark.

    Runs the streaming test twice:
    1. WITH consolidation — to measure if consolidation helps
    2. WITHOUT consolidation — as a baseline

    The delta between the two trajectories shows the value of consolidation.
    """
    task_types = tasks or ["with_consolidation", "no_consolidation"]
    results = []

    # Generate streaming facts
    print(f"  Generating {n_facts} streaming facts...")
    facts = _generate_streaming_facts(n_facts=n_facts)
    print(f"  Generated {len(facts)} facts across {len(set(f.category for f in facts))} categories")

    if "with_consolidation" in task_types:
        print(f"\n  Running WITH consolidation (consolidate every 50 facts)...")
        r = run_evolution_test(
            adapter, facts,
            eval_interval=eval_interval,
            consolidate_interval=50,
            limit=limit,
        )
        results.append(r)

    if "no_consolidation" in task_types:
        print(f"\n  Running WITHOUT consolidation (baseline)...")
        r = run_evolution_test(
            adapter, facts,
            eval_interval=eval_interval,
            consolidate_interval=None,
            limit=limit,
        )
        results.append(r)

    # Compare if both were run
    if len(results) == 2:
        with_consol = results[0].get_metric("final_contains_rate") or 0
        without_consol = results[1].get_metric("final_contains_rate") or 0
        delta = with_consol - without_consol
        print(f"\n  Consolidation effect: {delta:+.3f} "
              f"(with={with_consol:.3f}, without={without_consol:.3f})")

    return results
