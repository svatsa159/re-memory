"""BEAM benchmark runner.

BEAM stresses memory systems at extreme scales (100K-10M tokens) with
tasks focused on contradiction resolution and event ordering.

Since BEAM is paper-based (arxiv.org/abs/2510.27246) and may not have
a standard dataset release, this runner supports two modes:

1. Synthetic mode: Generate BEAM-style tasks using configurable parameters
2. Dataset mode: Load from a downloaded dataset if available

Synthetic tasks match the BEAM paper's evaluation protocol:
- Contradiction pairs at varying temporal distances
- Event ordering across long sequences
- Memory capacity under extreme scale
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

from tqdm import tqdm

from benchmarks.adapter import EngineAdapter, BenchmarkMetrics
from benchmarks.adapter.metrics import BenchmarkResult
from benchmarks.datasets.download import CACHE_DIR


BENCHMARK_NAME = "BEAM"


# ── Synthetic task generators ────────────────────────────────────────────

def _generate_contradiction_pairs(n_pairs: int = 50, n_distractors: int = 100) -> list[dict]:
    """Generate contradiction resolution tasks.

    Each task: store N distractor facts, then an original fact, then more
    distractors, then a contradicting fact. Query to verify latest is returned.
    """
    templates = [
        ("The capital of {entity} is {old_value}", "The capital of {entity} is {new_value}"),
        ("{person} works at {old_value}", "{person} works at {new_value}"),
        ("The project uses {old_value} for deployment", "The project switched to {new_value} for deployment"),
        ("{person}'s favorite language is {old_value}", "{person}'s favorite language is now {new_value}"),
        ("The meeting is scheduled for {old_value}", "The meeting has been moved to {new_value}"),
    ]

    entities = ["Zarkon", "Voltara", "Nexion", "Primex", "Qualdor"]
    persons = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    values_old = ["AlphaCity", "BetaCorp", "Kubernetes", "Python", "Monday 3pm"]
    values_new = ["GammaVille", "DeltaInc", "Nomad", "Rust", "Wednesday 10am"]
    distractor_facts = [
        f"Fact #{i}: The {random.choice(['temperature', 'humidity', 'pressure'])} "
        f"in zone {random.randint(1, 100)} is {random.randint(0, 100)}."
        for i in range(n_distractors)
    ]

    tasks = []
    for i in range(n_pairs):
        template_old, template_new = templates[i % len(templates)]
        entity = entities[i % len(entities)]
        person = persons[i % len(persons)]
        old_val = values_old[i % len(values_old)]
        new_val = values_new[i % len(values_new)]

        kwargs = {"entity": entity, "person": person, "old_value": old_val, "new_value": new_val}
        original = template_old.format(**kwargs)
        updated = template_new.format(**kwargs)

        # Query that should return the new value
        query_templates = [
            f"What is the capital of {entity}?",
            f"Where does {person} work?",
            f"What does the project use for deployment?",
            f"What is {person}'s favorite language?",
            f"When is the meeting?",
        ]
        query = query_templates[i % len(query_templates)]

        tasks.append({
            "type": "contradiction",
            "distractors_before": distractor_facts[:n_distractors // 2],
            "original_fact": original,
            "distractors_after": distractor_facts[n_distractors // 2:],
            "updated_fact": updated,
            "query": query,
            "expected_answer": new_val,
        })

    return tasks


def _generate_event_ordering_tasks(n_tasks: int = 30, n_events: int = 20) -> list[dict]:
    """Generate temporal event ordering tasks.

    Store a sequence of events with timestamps, then ask about ordering.
    """
    event_templates = [
        "On day {day}, {person} completed task #{task_id}: {action}",
        "At timestamp T{day}, system logged event: {action} by {person}",
    ]
    actions = ["code review", "deployment", "testing", "design doc", "standup", "retrospective"]
    persons = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

    tasks = []
    for t in range(n_tasks):
        events = []
        for i in range(n_events):
            template = event_templates[i % len(event_templates)]
            event_text = template.format(
                day=i + 1,
                person=persons[i % len(persons)],
                task_id=i + 1,
                action=actions[i % len(actions)],
            )
            events.append({"day": i + 1, "text": event_text})

        # Ask about ordering of two random events
        idx_a, idx_b = sorted(random.sample(range(n_events), 2))
        query = f"What happened first: the event on day {idx_a + 1} or the event on day {idx_b + 1}?"
        expected = events[idx_a]["text"]

        tasks.append({
            "type": "event_ordering",
            "events": events,
            "query": query,
            "expected_answer": expected,
            "correct_day": idx_a + 1,
        })

    return tasks


def _load_dataset(data_dir: Path | None = None) -> list[dict] | None:
    """Try to load BEAM dataset from cache. Returns None if not found."""
    if data_dir is None:
        data_dir = CACHE_DIR / "beam"

    if not data_dir.exists():
        return None

    samples = []
    for json_file in sorted(data_dir.rglob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                samples.extend(data)
            elif isinstance(data, dict):
                samples.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    return samples if samples else None


# ── Task runners ─────────────────────────────────────────────────────────

def run_contradiction_resolution(
    adapter: EngineAdapter,
    tasks: list[dict],
    limit: int | None = None,
) -> BenchmarkResult:
    """Run contradiction resolution benchmark."""
    result = BenchmarkResult(benchmark_name=BENCHMARK_NAME, task_name="contradiction_resolution")
    predictions, references = [], []
    contradiction_results = []

    for task in tqdm(tasks[:limit], desc="beam-contradiction"):
        adapter.reset()

        # Store distractor facts before
        for fact in task.get("distractors_before", []):
            adapter.store(fact)

        # Store original fact
        adapter.store(task["original_fact"])

        # Store distractor facts after
        for fact in task.get("distractors_after", []):
            adapter.store(fact)

        # Store contradicting/updated fact
        update_result = adapter.store(task["updated_fact"])
        verdict = update_result.get("verdict", "")
        contradiction_results.append({"verdict": verdict, "expected": "contradicts"})

        # Query
        retrieved = adapter.query_text(task["query"])
        predictions.append(retrieved)
        references.append(task["expected_answer"])

        result.samples.append({
            "query": task["query"],
            "expected": task["expected_answer"],
            "retrieved": retrieved[:200],
            "verdict": verdict,
            "contains_expected": BenchmarkMetrics.contains_match(retrieved, task["expected_answer"]),
        })

    if predictions:
        metrics = BenchmarkMetrics.evaluate_qa_batch(predictions, references)
        for name, value in metrics.items():
            result.add_metric(name, value, count=len(predictions))

    if contradiction_results:
        rate = BenchmarkMetrics.contradiction_resolution_rate(contradiction_results)
        result.add_metric("contradiction_detection_rate", rate, count=len(contradiction_results))

    if result.samples:
        acc = sum(s["contains_expected"] for s in result.samples) / len(result.samples)
        result.add_metric("latest_fact_accuracy", acc, count=len(result.samples))

    return result


def run_event_ordering(
    adapter: EngineAdapter,
    tasks: list[dict],
    limit: int | None = None,
) -> BenchmarkResult:
    """Run event ordering benchmark."""
    result = BenchmarkResult(benchmark_name=BENCHMARK_NAME, task_name="event_ordering")
    predictions, references = [], []

    for task in tqdm(tasks[:limit], desc="beam-event-ordering"):
        adapter.reset()

        # Store events in order
        for event in task["events"]:
            adapter.store(event["text"])

        # Query about ordering
        retrieved = adapter.query_text(task["query"])
        predictions.append(retrieved)
        references.append(task["expected_answer"])

        result.samples.append({
            "query": task["query"],
            "expected": task["expected_answer"],
            "retrieved": retrieved[:200],
            "contains_expected": BenchmarkMetrics.contains_match(retrieved, task["expected_answer"]),
        })

    if predictions:
        metrics = BenchmarkMetrics.evaluate_qa_batch(predictions, references)
        for name, value in metrics.items():
            result.add_metric(name, value, count=len(predictions))

    if result.samples:
        acc = sum(s["contains_expected"] for s in result.samples) / len(result.samples)
        result.add_metric("ordering_accuracy", acc, count=len(result.samples))

    return result


def run(
    adapter: EngineAdapter,
    data_dir: Path | None = None,
    limit: int | None = None,
    tasks: list[str] | None = None,
    synthetic: bool = True,
    n_contradiction_pairs: int = 50,
    n_ordering_tasks: int = 30,
    n_distractors: int = 100,
) -> list[BenchmarkResult]:
    """Run BEAM benchmark.

    Args:
        adapter: Initialized EngineAdapter.
        data_dir: Path to dataset (optional).
        limit: Max tasks per category.
        tasks: Subset to run ("contradiction", "event_ordering").
        synthetic: Generate synthetic tasks (True) or load from dataset.
        n_contradiction_pairs: Number of contradiction pairs to generate.
        n_ordering_tasks: Number of event ordering tasks to generate.
        n_distractors: Number of distractor facts per contradiction task.
    """
    # Try loading real dataset first
    dataset = _load_dataset(data_dir) if not synthetic else None

    task_types = tasks or ["contradiction", "event_ordering"]
    results = []

    if dataset:
        print(f"  Loaded {len(dataset)} samples from BEAM dataset")
        # Route to task runners based on loaded data
        for task_type in task_types:
            type_tasks = [s for s in dataset if s.get("type") == task_type]
            if task_type == "contradiction" and type_tasks:
                results.append(run_contradiction_resolution(adapter, type_tasks, limit=limit))
            elif task_type == "event_ordering" and type_tasks:
                results.append(run_event_ordering(adapter, type_tasks, limit=limit))
    else:
        print(f"  Using synthetic BEAM tasks (no dataset found)")

        if "contradiction" in task_types:
            print(f"\n  Generating {n_contradiction_pairs} contradiction pairs with {n_distractors} distractors each...")
            contradiction_tasks = _generate_contradiction_pairs(n_contradiction_pairs, n_distractors)
            results.append(run_contradiction_resolution(adapter, contradiction_tasks, limit=limit))

        if "event_ordering" in task_types:
            print(f"\n  Generating {n_ordering_tasks} event ordering tasks...")
            ordering_tasks = _generate_event_ordering_tasks(n_ordering_tasks)
            results.append(run_event_ordering(adapter, ordering_tasks, limit=limit))

    return results
