"""MemoryAgentBench runner.

Loads data from HuggingFace (ai-hyz/MemoryAgentBench), feeds it through
the engine adapter, and computes per-task metrics.

Actual dataset schema (each JSONL line = one scenario):
    - "context": str — large text (documents, dialogues, facts) to store
    - "questions": list[str] — evaluation questions
    - "answers": list[list[str]] — acceptable answers per question
    - "metadata": dict with question_types, source, etc.

Split files ARE the task types:
    Accurate_Retrieval.json (22 samples, ~1M chars each)
    Test_Time_Learning.json (6 samples, ~5.6M chars each)
    Long_Range_Understanding.json (110 samples)
    Conflict_Resolution.json (8 samples, ~26K chars each)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tqdm import tqdm

from benchmarks.adapter import EngineAdapter, BenchmarkMetrics
from benchmarks.adapter.metrics import BenchmarkResult
from benchmarks.datasets.download import CACHE_DIR


BENCHMARK_NAME = "MemoryAgentBench"

# Map split filenames to internal task names
SPLIT_MAP = {
    "Accurate_Retrieval": "accurate_retrieval",
    "Test_Time_Learning": "test_time_learning",
    "Long_Range_Understanding": "long_range_understanding",
    "Conflict_Resolution": "conflict_resolution",
}


def _load_split(data_dir: Path, split_name: str) -> list[dict]:
    """Load a specific split from the dataset cache."""
    path = data_dir / f"{split_name}.json"
    if not path.exists():
        return []

    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _load_all_splits(data_dir: Path | None = None) -> dict[str, list[dict]]:
    """Load all splits, keyed by task name."""
    if data_dir is None:
        data_dir = CACHE_DIR / "memory_agent_bench"

    splits = {}
    for split_file, task_name in SPLIT_MAP.items():
        samples = _load_split(data_dir, split_file)
        if samples:
            splits[task_name] = samples
    return splits


def _chunk_context(context: str, max_chunk_chars: int = 2000) -> list[str]:
    """Split a large context into storable chunks.

    Splits on document boundaries ("Document N:"), paragraph breaks,
    or numbered fact lines. Falls back to sentence splitting.
    """
    chunks = []

    # Try splitting on "Document N:" boundaries first
    doc_pattern = re.split(r"\n(?=Document \d+:)", context)
    if len(doc_pattern) > 1:
        for doc in doc_pattern:
            doc = doc.strip()
            if doc:
                # If a single document is still too long, split by paragraphs
                if len(doc) > max_chunk_chars:
                    chunks.extend(_split_by_paragraphs(doc, max_chunk_chars))
                else:
                    chunks.append(doc)
        return chunks

    # Try splitting on numbered facts ("0. Fact text\n1. Fact text")
    fact_lines = re.split(r"\n(?=\d+\.\s)", context)
    if len(fact_lines) > 3:
        # Group facts into chunks
        current = []
        current_len = 0
        for line in fact_lines:
            line = line.strip()
            if not line:
                continue
            if current_len + len(line) > max_chunk_chars and current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += len(line) + 1
        if current:
            chunks.append("\n".join(current))
        return chunks

    # Try splitting on dialogue boundaries ("Dialogue N:" or "System:/User:")
    dialogue_pattern = re.split(r"\n(?=Dialogue \d+:)", context)
    if len(dialogue_pattern) > 1:
        for dial in dialogue_pattern:
            dial = dial.strip()
            if dial:
                if len(dial) > max_chunk_chars:
                    chunks.extend(_split_by_paragraphs(dial, max_chunk_chars))
                else:
                    chunks.append(dial)
        return chunks

    # Fallback: paragraph splitting
    return _split_by_paragraphs(context, max_chunk_chars)


def _split_by_paragraphs(text: str, max_chunk_chars: int) -> list[str]:
    """Split text by double newlines, grouping into chunks up to max_chunk_chars."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chunk_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _get_best_answer(answers: list) -> str:
    """Extract the best single answer from the answer list.

    MAB answers are list[str] with multiple annotator answers.
    We take the first (or most common) as the reference.
    """
    if not answers:
        return ""
    if isinstance(answers[0], list):
        # Nested list — take first element of first list
        return str(answers[0][0]) if answers[0] else ""
    return str(answers[0])


def _get_all_answers(answers: list) -> list[str]:
    """Get all acceptable answer variants."""
    if not answers:
        return []
    if isinstance(answers, str):
        return [answers]
    return [str(a) for a in answers if a]


# ── Task runners ─────────────────────────────────────────────────────────

def run_accurate_retrieval(
    adapter: EngineAdapter,
    samples: list[dict],
    limit: int | None = None,
    question_limit: int | None = None,
) -> BenchmarkResult:
    """Accurate retrieval: store context documents, query to find specific answers."""
    result = BenchmarkResult(benchmark_name=BENCHMARK_NAME, task_name="accurate_retrieval")
    predictions, references, all_acceptable = [], [], []

    for sample in tqdm(samples[:limit], desc="accurate-retrieval"):
        adapter.reset()

        # Chunk and store context
        chunks = _chunk_context(sample["context"])
        print(f"    Storing {len(chunks)} chunks ({len(sample['context'])} chars)...")
        for chunk in chunks:
            adapter.store(chunk, source="mab_retrieval")

        # Consolidate for KG + schema promotion
        adapter.consolidate()

        # Evaluate questions
        questions = sample["questions"]
        answers = sample["answers"]
        q_limit = question_limit or len(questions)

        for q, ans_list in zip(questions[:q_limit], answers[:q_limit]):
            acceptable = _get_all_answers(ans_list)
            best_answer = acceptable[0] if acceptable else ""

            recall_result = adapter.query(q)
            memories = recall_result.get("memories", [])
            retrieved_texts = [m.get("content", "") for m in memories]
            prediction = "\n".join(retrieved_texts[:5])

            predictions.append(prediction)
            references.append(best_answer)
            all_acceptable.append(acceptable)

            # Hit@K with any acceptable answer
            hit1 = max(BenchmarkMetrics.hit_at_k(retrieved_texts, a, k=1) for a in acceptable) if acceptable else 0
            hit5 = max(BenchmarkMetrics.hit_at_k(retrieved_texts, a, k=5) for a in acceptable) if acceptable else 0
            contains = max(BenchmarkMetrics.contains_match(prediction, a) for a in acceptable) if acceptable else 0

            result.samples.append({
                "question": q,
                "expected": best_answer,
                "acceptable": acceptable,
                "retrieved_preview": prediction[:200],
                "hit@1": hit1,
                "hit@5": hit5,
                "contains_any": contains,
                "num_retrieved": len(memories),
            })

    if predictions:
        # Standard metrics against best answer
        metrics = BenchmarkMetrics.evaluate_qa_batch(predictions, references)
        for name, value in metrics.items():
            result.add_metric(name, value, count=len(predictions))

        # Aggregate flexible metrics (any acceptable answer)
        result.add_metric("hit@1", _avg(result.samples, "hit@1"), count=len(result.samples))
        result.add_metric("hit@5", _avg(result.samples, "hit@5"), count=len(result.samples))
        result.add_metric("contains_any_answer", _avg(result.samples, "contains_any"), count=len(result.samples))

    _add_latency_metrics(result, adapter)
    return result


def run_test_time_learning(
    adapter: EngineAdapter,
    samples: list[dict],
    limit: int | None = None,
    question_limit: int | None = None,
) -> BenchmarkResult:
    """Test-time learning: observe dialogues, answer questions about learned patterns."""
    result = BenchmarkResult(benchmark_name=BENCHMARK_NAME, task_name="test_time_learning")
    predictions, references = [], []

    for sample in tqdm(samples[:limit], desc="test-time-learning"):
        adapter.reset()

        # Store dialogue context
        chunks = _chunk_context(sample["context"])
        print(f"    Storing {len(chunks)} chunks ({len(sample['context'])} chars)...")
        for chunk in chunks:
            adapter.store(chunk, source="mab_learning")

        questions = sample["questions"]
        answers = sample["answers"]
        q_limit = question_limit or len(questions)

        for q, ans_list in zip(questions[:q_limit], answers[:q_limit]):
            acceptable = _get_all_answers(ans_list)
            best_answer = acceptable[0] if acceptable else ""

            retrieved = adapter.query_text(q)
            predictions.append(retrieved)
            references.append(best_answer)

            contains = max(BenchmarkMetrics.contains_match(retrieved, a) for a in acceptable) if acceptable else 0

            result.samples.append({
                "question": q[:100],
                "expected": best_answer,
                "retrieved_preview": retrieved[:200],
                "contains_any": contains,
            })

    if predictions:
        metrics = BenchmarkMetrics.evaluate_qa_batch(predictions, references)
        for name, value in metrics.items():
            result.add_metric(name, value, count=len(predictions))

        result.add_metric("contains_any_answer", _avg(result.samples, "contains_any"), count=len(result.samples))

    _add_latency_metrics(result, adapter)
    return result


def run_conflict_resolution(
    adapter: EngineAdapter,
    samples: list[dict],
    limit: int | None = None,
    question_limit: int | None = None,
) -> BenchmarkResult:
    """Conflict resolution: store facts (with contradictions), query for latest truth."""
    result = BenchmarkResult(benchmark_name=BENCHMARK_NAME, task_name="conflict_resolution")
    predictions, references = [], []
    contradiction_verdicts = []

    for sample in tqdm(samples[:limit], desc="conflict-resolution"):
        adapter.reset()

        # Store facts — context contains numbered facts, some may contradict
        chunks = _chunk_context(sample["context"])
        print(f"    Storing {len(chunks)} chunks ({len(sample['context'])} chars)...")
        for chunk in chunks:
            store_result = adapter.store(chunk, source="mab_conflict")
            verdict = store_result.get("verdict", "")
            if verdict:
                contradiction_verdicts.append(verdict)

        questions = sample["questions"]
        answers = sample["answers"]
        q_limit = question_limit or len(questions)

        for q, ans_list in zip(questions[:q_limit], answers[:q_limit]):
            acceptable = _get_all_answers(ans_list)
            best_answer = acceptable[0] if acceptable else ""

            retrieved = adapter.query_text(q)
            predictions.append(retrieved)
            references.append(best_answer)

            contains = max(BenchmarkMetrics.contains_match(retrieved, a) for a in acceptable) if acceptable else 0

            result.samples.append({
                "question": q[:100],
                "expected": best_answer,
                "retrieved_preview": retrieved[:200],
                "contains_any": contains,
            })

    if predictions:
        metrics = BenchmarkMetrics.evaluate_qa_batch(predictions, references)
        for name, value in metrics.items():
            result.add_metric(name, value, count=len(predictions))

        result.add_metric("contains_any_answer", _avg(result.samples, "contains_any"), count=len(result.samples))

    # Verdict distribution
    if contradiction_verdicts:
        from collections import Counter
        verdict_counts = Counter(contradiction_verdicts)
        total_v = len(contradiction_verdicts)
        for v, count in verdict_counts.most_common():
            result.add_metric(f"verdict_{v}_rate", count / total_v, count=count)

        contradict_rate = sum(1 for v in contradiction_verdicts if v in ("contradicts", "update")) / total_v
        result.add_metric("contradiction_detection_rate", contradict_rate, count=total_v)

    _add_latency_metrics(result, adapter)
    return result


def run_long_range_understanding(
    adapter: EngineAdapter,
    samples: list[dict],
    limit: int | None = None,
    question_limit: int | None = None,
) -> BenchmarkResult:
    """Long-range understanding: store large texts, answer comprehension questions."""
    result = BenchmarkResult(benchmark_name=BENCHMARK_NAME, task_name="long_range_understanding")
    predictions, references = [], []

    for sample in tqdm(samples[:limit], desc="long-range"):
        adapter.reset()

        chunks = _chunk_context(sample["context"])
        print(f"    Storing {len(chunks)} chunks ({len(sample['context'])} chars)...")
        for chunk in chunks:
            adapter.store(chunk, source="mab_long_range")

        # Consolidate — long-range needs schema/KG for abstraction
        adapter.consolidate()

        questions = sample["questions"]
        answers = sample["answers"]
        q_limit = question_limit or len(questions)

        for q, ans_list in zip(questions[:q_limit], answers[:q_limit]):
            acceptable = _get_all_answers(ans_list)
            best_answer = acceptable[0] if acceptable else ""

            retrieved = adapter.query_text(q, max_tokens=4000)
            predictions.append(retrieved)
            references.append(best_answer)

            contains = max(BenchmarkMetrics.contains_match(retrieved, a) for a in acceptable) if acceptable else 0

            result.samples.append({
                "question": q[:100],
                "expected": best_answer[:100],
                "retrieved_preview": retrieved[:200],
                "contains_any": contains,
            })

    if predictions:
        metrics = BenchmarkMetrics.evaluate_qa_batch(predictions, references)
        for name, value in metrics.items():
            result.add_metric(name, value, count=len(predictions))

        result.add_metric("contains_any_answer", _avg(result.samples, "contains_any"), count=len(result.samples))

    _add_latency_metrics(result, adapter)
    return result


# ── Helpers ──────────────────────────────────────────────────────────────

def _avg(samples: list[dict], key: str) -> float:
    vals = [s[key] for s in samples if key in s]
    return sum(vals) / len(vals) if vals else 0.0


def _add_latency_metrics(result: BenchmarkResult, adapter: EngineAdapter):
    """Add store and query latency stats to a result."""
    store_latencies = adapter.get_latencies("store")
    if store_latencies:
        for k, v in BenchmarkMetrics.latency_stats(store_latencies).items():
            result.add_metric(f"store_{k}", v)

    query_latencies = adapter.get_latencies("query")
    if query_latencies:
        for k, v in BenchmarkMetrics.latency_stats(query_latencies).items():
            result.add_metric(f"query_{k}", v)


# ── Main runner ──────────────────────────────────────────────────────────

TASK_RUNNERS = {
    "accurate_retrieval": run_accurate_retrieval,
    "test_time_learning": run_test_time_learning,
    "conflict_resolution": run_conflict_resolution,
    "long_range_understanding": run_long_range_understanding,
}


def run(
    adapter: EngineAdapter,
    data_dir: Path | None = None,
    limit: int | None = None,
    tasks: list[str] | None = None,
    question_limit: int | None = 20,
) -> list[BenchmarkResult]:
    """Run MemoryAgentBench tasks.

    Args:
        adapter: Initialized EngineAdapter.
        data_dir: Path to dataset. If None, uses cached download.
        limit: Max samples per task (for quick testing).
        tasks: Subset of tasks to run. None = all.
        question_limit: Max questions per sample (the dataset has 100-200 per sample).

    Returns:
        List of BenchmarkResult objects.
    """
    splits = _load_all_splits(data_dir)
    if not splits:
        print(f"  WARNING: No data found. Run `re-memory-bench download memory_agent_bench` first.")
        return []

    print(f"  Loaded MemoryAgentBench splits:")
    for task_name, samples in splits.items():
        total_q = sum(len(s.get("questions", [])) for s in samples)
        print(f"    {task_name}: {len(samples)} samples, {total_q} total questions")

    run_tasks = tasks or list(splits.keys())
    results = []

    for task_name in run_tasks:
        task_samples = splits.get(task_name, [])
        if not task_samples:
            print(f"  No samples for task '{task_name}', skipping...")
            continue

        runner_fn = TASK_RUNNERS.get(task_name)
        if runner_fn is None:
            print(f"  Unknown task '{task_name}', running as accurate_retrieval...")
            runner_fn = run_accurate_retrieval

        print(f"\n  Running {task_name} ({len(task_samples)} samples)...")
        result = runner_fn(adapter, task_samples, limit=limit, question_limit=question_limit)
        results.append(result)

    return results
