"""LongMemEval benchmark runner.

LongMemEval tests 5 memory dimensions with curated questions:
1. Information Extraction — basic retrieval of stored facts
2. Multi-session Reasoning — connecting info across sessions
3. Temporal Reasoning — understanding time-ordered events
4. Knowledge Update — handling contradictions/updates
5. Abstention — knowing when you DON'T have the answer

Dataset structure (from xiaowu0162/LongMemEval):
    data/
        longmemeval_s.json (short context ~115K tokens)
        longmemeval_m.json (medium ~500K tokens)
        longmemeval_l.json (long ~1.5M tokens)
    Each contains:
        - "sessions": list of conversation sessions
        - "questions": list of evaluation questions with types
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from benchmarks.adapter import EngineAdapter, BenchmarkMetrics
from benchmarks.adapter.metrics import BenchmarkResult
from benchmarks.datasets.download import CACHE_DIR


BENCHMARK_NAME = "LongMemEval"

DIMENSIONS = [
    "information_extraction",
    "multi_session_reasoning",
    "temporal_reasoning",
    "knowledge_update",
    "abstention",
]


def _load_dataset(data_dir: Path | None = None, scale: str = "s") -> dict:
    """Load LongMemEval at the specified scale (s/m/l)."""
    if data_dir is None:
        data_dir = CACHE_DIR / "longmemeval"

    # Try multiple naming patterns
    candidates = [
        data_dir / f"longmemeval_{scale}.json",
        data_dir / "data" / f"longmemeval_{scale}.json",
        data_dir / f"LongMemEval_{scale}.json",
        data_dir / "data" / f"LongMemEval_{scale}.json",
    ]

    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)

    # Fallback: try any JSON file
    search_dirs = [data_dir / "data", data_dir]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for json_file in sorted(search_dir.rglob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if isinstance(data, dict) and ("sessions" in data or "questions" in data):
                    return data
            except (json.JSONDecodeError, KeyError):
                continue

    return {}


def _ingest_sessions(adapter: EngineAdapter, sessions: list) -> int:
    """Ingest all sessions into the engine. Returns count of stored observations."""
    count = 0
    for session in sessions:
        if isinstance(session, dict):
            turns = session.get("conversation", session.get("turns", session.get("messages", [])))
            for turn in turns:
                if isinstance(turn, dict):
                    text = turn.get("content", turn.get("text", turn.get("utterance", "")))
                    role = turn.get("role", turn.get("speaker", ""))
                    if text.strip():
                        full_text = f"{role}: {text}" if role else text
                        adapter.store(full_text, source="longmemeval")
                        count += 1
                elif isinstance(turn, str) and turn.strip():
                    adapter.store(turn, source="longmemeval")
                    count += 1
        elif isinstance(session, str) and session.strip():
            adapter.store(session, source="longmemeval")
            count += 1
    return count


def _classify_question(question: dict) -> str:
    """Determine which LongMemEval dimension a question belongs to."""
    for key in ("type", "dimension", "category", "task_type"):
        if key in question:
            val = str(question[key]).lower().replace("-", "_").replace(" ", "_")
            # Normalize to our dimension names
            if "extract" in val or "retrieval" in val:
                return "information_extraction"
            if "multi" in val or "cross" in val or "reasoning" in val:
                return "multi_session_reasoning"
            if "temporal" in val or "time" in val or "when" in val:
                return "temporal_reasoning"
            if "update" in val or "contradict" in val or "knowledge" in val:
                return "knowledge_update"
            if "abstain" in val or "unanswerable" in val or "no_answer" in val:
                return "abstention"
            return val

    return "unknown"


def run_dimension(
    adapter: EngineAdapter,
    data: dict,
    dimension: str,
    limit: int | None = None,
) -> BenchmarkResult:
    """Run evaluation for a specific LongMemEval dimension."""
    result = BenchmarkResult(benchmark_name=BENCHMARK_NAME, task_name=dimension)

    questions = data.get("questions", data.get("qa", []))
    dim_questions = [q for q in questions if _classify_question(q) == dimension]

    if not dim_questions and dimension == "all":
        dim_questions = questions

    if not dim_questions:
        print(f"    No questions for dimension '{dimension}'")
        return result

    predictions, references = [], []
    has_answer_flags = []

    for q in tqdm(dim_questions[:limit], desc=f"longmemeval-{dimension}"):
        question_text = q.get("question", q.get("query", q.get("input", "")))
        answer_text = q.get("answer", q.get("expected", q.get("output", "")))
        answerable = q.get("answerable", q.get("has_answer", True))

        if not question_text:
            continue

        retrieved = adapter.query_text(question_text)
        predictions.append(retrieved)
        references.append(answer_text)
        has_answer_flags.append(answerable)

        result.samples.append({
            "question": question_text,
            "expected": answer_text,
            "retrieved": retrieved[:200],
            "answerable": answerable,
            "dimension": dimension,
        })

    if predictions:
        metrics = BenchmarkMetrics.evaluate_qa_batch(predictions, references)
        for name, value in metrics.items():
            result.add_metric(name, value, count=len(predictions))

    # Abstention accuracy (for abstention dimension or if has_answer flags exist)
    if has_answer_flags and not all(has_answer_flags):
        abs_acc = BenchmarkMetrics.abstention_accuracy(predictions, has_answer_flags)
        result.add_metric("abstention_accuracy", abs_acc, count=len(predictions))

    return result


def run(
    adapter: EngineAdapter,
    data_dir: Path | None = None,
    limit: int | None = None,
    tasks: list[str] | None = None,
    scale: str = "s",
) -> list[BenchmarkResult]:
    """Run all LongMemEval dimensions.

    Args:
        adapter: Initialized EngineAdapter.
        data_dir: Path to dataset.
        limit: Max questions per dimension.
        tasks: Subset of dimensions. None = all.
        scale: Dataset scale — "s" (115K tokens), "m" (500K), "l" (1.5M).
    """
    data = _load_dataset(data_dir, scale=scale)
    if not data:
        print(f"  WARNING: No data found. Run `re-memory-bench download longmemeval` first.")
        return []

    sessions = data.get("sessions", [])
    questions = data.get("questions", data.get("qa", []))
    print(f"  Loaded LongMemEval (scale={scale}): {len(sessions)} sessions, {len(questions)} questions")

    # Ingest all sessions once (shared context)
    adapter.reset()
    print(f"  Ingesting sessions...")
    stored = _ingest_sessions(adapter, sessions)
    print(f"  Stored {stored} observations")

    # Consolidate for better retrieval
    print(f"  Consolidating...")
    adapter.consolidate()

    # Determine which dimensions have questions
    dim_counts: dict[str, int] = {}
    for q in questions:
        d = _classify_question(q)
        dim_counts[d] = dim_counts.get(d, 0) + 1
    print(f"  Dimensions: {', '.join(f'{k}({v})' for k, v in dim_counts.items())}")

    dimensions = tasks or list(dim_counts.keys()) or DIMENSIONS

    results = []
    for dim in dimensions:
        print(f"\n  Running dimension: {dim}...")
        result = run_dimension(adapter, data, dim, limit=limit)
        results.append(result)

    return results
