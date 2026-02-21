"""LoCoMo benchmark runner.

LoCoMo contains long conversations (300+ turns across 35 sessions) with
QA tasks requiring multi-hop reasoning and adversarial robustness.

Dataset structure (from snap-research/locomo):
    data/
        locomo10.json (or similar)
    Each file contains sessions with:
        - "conversation": list of utterances
        - "qa": list of {"question", "answer", "type", "evidence"}
        - QA types: single-hop, multi-hop, temporal, adversarial, open-domain
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from benchmarks.adapter import EngineAdapter, BenchmarkMetrics
from benchmarks.adapter.metrics import BenchmarkResult
from benchmarks.datasets.download import CACHE_DIR


BENCHMARK_NAME = "LoCoMo"

QA_TYPES = ["single-hop", "multi-hop", "temporal", "adversarial", "open-domain"]


def _load_dataset(data_dir: Path | None = None) -> list[dict]:
    """Load LoCoMo conversation + QA data."""
    if data_dir is None:
        data_dir = CACHE_DIR / "locomo"

    samples = []

    # Try data/ subdirectory first (git clone structure)
    search_dirs = [data_dir / "data", data_dir]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for json_file in sorted(search_dir.rglob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    samples.extend(data)
                elif isinstance(data, dict):
                    # Could be a single session or a dict of sessions
                    if "conversation" in data or "qa" in data:
                        samples.append(data)
                    else:
                        # Dict of session_id -> session_data
                        for key, val in data.items():
                            if isinstance(val, dict):
                                val["session_id"] = key
                                samples.append(val)
            except (json.JSONDecodeError, KeyError):
                continue

    return samples


def _extract_utterances(sample: dict) -> list[str]:
    """Extract conversation utterances from a LoCoMo sample."""
    # Variant 1: "conversation" list of dicts with "text" or "content"
    if "conversation" in sample:
        conv = sample["conversation"]
        if isinstance(conv, list):
            texts = []
            for turn in conv:
                if isinstance(turn, dict):
                    text = turn.get("text", turn.get("content", turn.get("utterance", "")))
                    speaker = turn.get("speaker", turn.get("role", ""))
                    if text:
                        prefix = f"{speaker}: " if speaker else ""
                        texts.append(f"{prefix}{text}")
                elif isinstance(turn, str):
                    texts.append(turn)
            return texts

    # Variant 2: "sessions" list of conversation sessions
    if "sessions" in sample:
        texts = []
        for session in sample["sessions"]:
            if isinstance(session, dict) and "conversation" in session:
                texts.extend(_extract_utterances(session))
            elif isinstance(session, list):
                texts.extend(str(u) for u in session)
        return texts

    return []


def _extract_qa(sample: dict) -> list[dict]:
    """Extract QA pairs with metadata from a LoCoMo sample."""
    qa_pairs = []

    # "qa" list
    if "qa" in sample:
        for qa in sample["qa"]:
            if isinstance(qa, dict):
                qa_pairs.append({
                    "question": qa.get("question", qa.get("query", "")),
                    "answer": qa.get("answer", qa.get("expected", "")),
                    "type": qa.get("type", qa.get("category", "unknown")),
                    "evidence": qa.get("evidence", qa.get("supporting_facts", [])),
                })

    # "questions" list
    elif "questions" in sample:
        for q in sample["questions"]:
            if isinstance(q, dict):
                qa_pairs.append({
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "type": q.get("type", "unknown"),
                    "evidence": q.get("evidence", []),
                })

    return qa_pairs


def run_by_qa_type(
    adapter: EngineAdapter,
    samples: list[dict],
    qa_type: str,
    limit: int | None = None,
) -> BenchmarkResult:
    """Run evaluation for a specific QA type."""
    result = BenchmarkResult(benchmark_name=BENCHMARK_NAME, task_name=f"qa_{qa_type}")
    predictions, references = [], []

    sample_count = 0
    for sample in tqdm(samples, desc=f"locomo-{qa_type}"):
        if limit and sample_count >= limit:
            break

        qa_pairs = [qa for qa in _extract_qa(sample) if qa["type"] == qa_type or qa_type == "all"]
        if not qa_pairs:
            continue

        adapter.reset()

        # Ingest conversation
        utterances = _extract_utterances(sample)
        for utt in utterances:
            if utt.strip():
                adapter.store(utt)

        # Trigger consolidation for multi-hop (needs KG)
        if qa_type in ("multi-hop", "temporal"):
            adapter.consolidate()

        # Evaluate
        for qa in qa_pairs:
            question, answer = qa["question"], qa["answer"]
            if not question or not answer:
                continue

            recall_result = adapter.query(question)
            memories = recall_result.get("memories", [])
            retrieved_texts = [m.get("content", "") for m in memories]
            prediction = "\n".join(retrieved_texts[:5])

            predictions.append(prediction)
            references.append(answer)
            result.samples.append({
                "question": question,
                "expected": answer,
                "retrieved": prediction[:200],
                "qa_type": qa["type"],
                "hit@1": BenchmarkMetrics.hit_at_k(retrieved_texts, answer, k=1),
                "hit@5": BenchmarkMetrics.hit_at_k(retrieved_texts, answer, k=5),
                "mrr": BenchmarkMetrics.mrr(retrieved_texts, answer),
            })

        sample_count += 1

    if predictions:
        metrics = BenchmarkMetrics.evaluate_qa_batch(predictions, references)
        for name, value in metrics.items():
            result.add_metric(name, value, count=len(predictions))

        # Hit@K and MRR aggregates
        result.add_metric("hit@1", _avg(result.samples, "hit@1"), count=len(result.samples))
        result.add_metric("hit@5", _avg(result.samples, "hit@5"), count=len(result.samples))
        result.add_metric("mrr", _avg(result.samples, "mrr"), count=len(result.samples))

    return result


def _avg(samples: list[dict], key: str) -> float:
    vals = [s[key] for s in samples if key in s]
    return sum(vals) / len(vals) if vals else 0.0


def run(
    adapter: EngineAdapter,
    data_dir: Path | None = None,
    limit: int | None = None,
    tasks: list[str] | None = None,
) -> list[BenchmarkResult]:
    """Run all LoCoMo QA tasks.

    Args:
        adapter: Initialized EngineAdapter.
        data_dir: Path to dataset. If None, uses cached download.
        limit: Max samples per QA type.
        tasks: Subset of QA types to run. None = all.

    Returns:
        List of BenchmarkResult objects per QA type.
    """
    samples = _load_dataset(data_dir)
    if not samples:
        print(f"  WARNING: No data found. Run `re-memory-bench download locomo` first.")
        return []

    print(f"  Loaded {len(samples)} conversations from LoCoMo")

    # Count QA types
    type_counts: dict[str, int] = {}
    for sample in samples:
        for qa in _extract_qa(sample):
            t = qa["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  QA types: {', '.join(f'{k}({v})' for k, v in type_counts.items())}")

    qa_types = tasks or list(type_counts.keys()) or ["all"]
    results = []

    for qa_type in qa_types:
        print(f"\n  Running QA type: {qa_type}...")
        result = run_by_qa_type(adapter, samples, qa_type, limit=limit)
        results.append(result)

    return results
