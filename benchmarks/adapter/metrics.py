"""Common metrics for evaluating memory system performance.

Computes standard NLP and retrieval metrics used across all benchmarks:
- Exact match, F1, precision, recall (token-level)
- ROUGE-L for recall quality
- BLEU for generation quality
- Hit@K for retrieval ranking
- Latency statistics
- Verdict distribution analysis
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MetricResult:
    """Single metric measurement."""

    name: str
    value: float
    count: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregate results for a benchmark run."""

    benchmark_name: str
    task_name: str
    metrics: list[MetricResult] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def add_metric(self, name: str, value: float, count: int = 0, **details):
        self.metrics.append(MetricResult(name=name, value=value, count=count, details=details))

    def get_metric(self, name: str) -> float | None:
        for m in self.metrics:
            if m.name == name:
                return m.value
        return None

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark_name,
            "task": self.task_name,
            "metrics": {m.name: {"value": m.value, "count": m.count, **m.details} for m in self.metrics},
            "num_samples": len(self.samples),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class BenchmarkMetrics:
    """Stateless metric computation utilities."""

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Simple whitespace tokenization after normalization."""
        return BenchmarkMetrics.normalize(text).split()

    # ── Exact Match ──────────────────────────────────────────────────────

    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        """1.0 if normalized prediction == normalized reference, else 0.0."""
        return 1.0 if BenchmarkMetrics.normalize(prediction) == BenchmarkMetrics.normalize(reference) else 0.0

    @staticmethod
    def exact_match_batch(predictions: list[str], references: list[str]) -> float:
        """Average exact match across pairs."""
        if not predictions:
            return 0.0
        scores = [BenchmarkMetrics.exact_match(p, r) for p, r in zip(predictions, references)]
        return statistics.mean(scores)

    # ── Token-level F1, Precision, Recall ────────────────────────────────

    @staticmethod
    def token_f1(prediction: str, reference: str) -> tuple[float, float, float]:
        """Compute token-level precision, recall, and F1.

        Returns: (f1, precision, recall)
        """
        pred_tokens = Counter(BenchmarkMetrics.tokenize(prediction))
        ref_tokens = Counter(BenchmarkMetrics.tokenize(reference))

        common = sum((pred_tokens & ref_tokens).values())

        if common == 0:
            return 0.0, 0.0, 0.0

        precision = common / sum(pred_tokens.values())
        recall = common / sum(ref_tokens.values())
        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall

    @staticmethod
    def token_f1_batch(predictions: list[str], references: list[str]) -> tuple[float, float, float]:
        """Average token-level F1, precision, recall across pairs."""
        if not predictions:
            return 0.0, 0.0, 0.0
        results = [BenchmarkMetrics.token_f1(p, r) for p, r in zip(predictions, references)]
        avg_f1 = statistics.mean(r[0] for r in results)
        avg_p = statistics.mean(r[1] for r in results)
        avg_r = statistics.mean(r[2] for r in results)
        return avg_f1, avg_p, avg_r

    # ── Contains Match ───────────────────────────────────────────────────

    @staticmethod
    def contains_match(prediction: str, reference: str) -> float:
        """1.0 if the normalized reference is contained in normalized prediction."""
        return 1.0 if BenchmarkMetrics.normalize(reference) in BenchmarkMetrics.normalize(prediction) else 0.0

    @staticmethod
    def contains_any(prediction: str, references: list[str]) -> float:
        """1.0 if any reference is contained in the prediction."""
        pred_norm = BenchmarkMetrics.normalize(prediction)
        return 1.0 if any(BenchmarkMetrics.normalize(r) in pred_norm for r in references) else 0.0

    # ── ROUGE-L ──────────────────────────────────────────────────────────

    @staticmethod
    def rouge_l(prediction: str, reference: str) -> float:
        """ROUGE-L F1 score using longest common subsequence."""
        pred_tokens = BenchmarkMetrics.tokenize(prediction)
        ref_tokens = BenchmarkMetrics.tokenize(reference)

        if not pred_tokens or not ref_tokens:
            return 0.0

        # LCS via dynamic programming
        m, n = len(pred_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_len = dp[m][n]
        if lcs_len == 0:
            return 0.0

        precision = lcs_len / m
        recall = lcs_len / n
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def rouge_l_batch(predictions: list[str], references: list[str]) -> float:
        """Average ROUGE-L across pairs."""
        if not predictions:
            return 0.0
        return statistics.mean(
            BenchmarkMetrics.rouge_l(p, r) for p, r in zip(predictions, references)
        )

    # ── Hit@K ────────────────────────────────────────────────────────────

    @staticmethod
    def hit_at_k(retrieved_texts: list[str], gold_answer: str, k: int = 5) -> float:
        """1.0 if gold answer appears in top-k retrieved texts."""
        gold_norm = BenchmarkMetrics.normalize(gold_answer)
        for text in retrieved_texts[:k]:
            if gold_norm in BenchmarkMetrics.normalize(text):
                return 1.0
        return 0.0

    @staticmethod
    def mrr(retrieved_texts: list[str], gold_answer: str) -> float:
        """Mean Reciprocal Rank: 1/(rank of first correct result)."""
        gold_norm = BenchmarkMetrics.normalize(gold_answer)
        for i, text in enumerate(retrieved_texts):
            if gold_norm in BenchmarkMetrics.normalize(text):
                return 1.0 / (i + 1)
        return 0.0

    # ── Abstention (knowing when you DON'T know) ────────────────────────

    @staticmethod
    def abstention_accuracy(
        predictions: list[str],
        has_answer: list[bool],
        abstention_phrases: list[str] | None = None,
    ) -> float:
        """Accuracy of knowing when to abstain vs. answer.

        Args:
            predictions: Model's answers.
            has_answer: Whether a correct answer exists in memory.
            abstention_phrases: Phrases indicating abstention (e.g., "I don't know").
        """
        if abstention_phrases is None:
            abstention_phrases = [
                "i don't know", "i dont know", "no information",
                "not found", "no relevant", "cannot answer",
                "no memory", "unknown", "not sure",
            ]

        correct = 0
        for pred, answerable in zip(predictions, has_answer):
            pred_norm = BenchmarkMetrics.normalize(pred)
            did_abstain = any(phrase in pred_norm for phrase in abstention_phrases)

            if answerable and not did_abstain:
                correct += 1  # Correctly answered
            elif not answerable and did_abstain:
                correct += 1  # Correctly abstained

        return correct / len(predictions) if predictions else 0.0

    # ── Contradiction Detection ──────────────────────────────────────────

    @staticmethod
    def contradiction_resolution_rate(
        results: list[dict],
    ) -> float:
        """Fraction of contradiction pairs where the engine detected the conflict.

        Each result dict should have:
            - "verdict": the engine's verdict for the contradicting observation
            - "expected": "contradicts" or "update"
        """
        if not results:
            return 0.0

        detected = sum(
            1 for r in results
            if r.get("verdict", "").lower() in ("contradicts", "update")
        )
        return detected / len(results)

    # ── Latency ──────────────────────────────────────────────────────────

    @staticmethod
    def latency_stats(latencies_ms: list[float]) -> dict:
        """Compute latency statistics from millisecond values."""
        if not latencies_ms:
            return {"mean_ms": 0, "median_ms": 0, "p95_ms": 0, "p99_ms": 0, "min_ms": 0, "max_ms": 0}

        sorted_l = sorted(latencies_ms)
        p95_idx = int(len(sorted_l) * 0.95)
        p99_idx = int(len(sorted_l) * 0.99)

        return {
            "mean_ms": statistics.mean(sorted_l),
            "median_ms": statistics.median(sorted_l),
            "p95_ms": sorted_l[min(p95_idx, len(sorted_l) - 1)],
            "p99_ms": sorted_l[min(p99_idx, len(sorted_l) - 1)],
            "min_ms": sorted_l[0],
            "max_ms": sorted_l[-1],
        }

    # ── Aggregate scoring ────────────────────────────────────────────────

    @staticmethod
    def evaluate_qa_pair(
        prediction: str,
        reference: str,
    ) -> dict[str, float]:
        """Run all standard metrics on a single QA pair."""
        f1, prec, rec = BenchmarkMetrics.token_f1(prediction, reference)
        return {
            "exact_match": BenchmarkMetrics.exact_match(prediction, reference),
            "token_f1": f1,
            "token_precision": prec,
            "token_recall": rec,
            "rouge_l": BenchmarkMetrics.rouge_l(prediction, reference),
            "contains_match": BenchmarkMetrics.contains_match(prediction, reference),
        }

    @staticmethod
    def evaluate_qa_batch(
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Run all standard metrics on a batch of QA pairs."""
        if not predictions:
            return {k: 0.0 for k in ["exact_match", "token_f1", "token_precision", "token_recall", "rouge_l", "contains_match"]}

        all_results = [BenchmarkMetrics.evaluate_qa_pair(p, r) for p, r in zip(predictions, references)]

        aggregated = {}
        for key in all_results[0]:
            aggregated[key] = statistics.mean(r[key] for r in all_results)
        return aggregated
