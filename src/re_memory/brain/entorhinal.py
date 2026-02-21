"""Entorhinal Cortex (EC): Input Gateway

The entorhinal cortex is the main interface between the hippocampus and neocortex.
It preprocesses all incoming information: parsing, feature extraction, embedding.

In our system: receives raw text → extracts structured features → generates embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..providers.base import LLMProvider, EmbeddingProvider


@dataclass
class ParsedInput:
    """Structured representation of parsed input from the entorhinal cortex."""

    raw_text: str
    embedding: list[float] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 to 1
    temporal_refs: list[str] = field(default_factory=list)
    summary: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EntorhinalCortex:
    """Input parsing and embedding generation."""

    def __init__(self, llm: LLMProvider, embedder: EmbeddingProvider):
        self.llm = llm
        self.embedder = embedder

    async def parse(self, text: str) -> ParsedInput:
        """Full parse: extract features + generate embedding."""
        # Generate embedding
        embedding = await self.embedder.embed_single(text)

        # Extract entities and features via LLM
        features = await self._extract_features(text)

        return ParsedInput(
            raw_text=text,
            embedding=embedding,
            entities=features.get("entities", []),
            topics=features.get("topics", []),
            sentiment=features.get("sentiment", 0.0),
            temporal_refs=features.get("temporal_refs", []),
            summary=features.get("summary", text[:200]),
        )

    async def embed_only(self, text: str) -> list[float]:
        """Quick embedding without full parse."""
        return await self.embedder.embed_single(text)

    async def parse_with_importance(self, text: str) -> tuple[ParsedInput, float]:
        """Full parse + importance scoring in a single LLM call.

        Combines feature extraction and importance scoring to halve API cost.
        Returns (ParsedInput, importance_score) tuple.
        """
        embedding = await self.embedder.embed_single(text)
        features_and_importance = await self._extract_features_with_importance(text)

        importance = float(features_and_importance.get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))

        return ParsedInput(
            raw_text=text,
            embedding=embedding,
            entities=features_and_importance.get("entities", []),
            topics=features_and_importance.get("topics", []),
            sentiment=features_and_importance.get("sentiment", 0.0),
            temporal_refs=features_and_importance.get("temporal_refs", []),
            summary=features_and_importance.get("summary", text[:200]),
        ), importance

    async def _extract_features_with_importance(self, text: str) -> dict:
        """Single LLM call to extract features AND score importance."""
        prompt = f"""Extract structured information from this text and rate its importance. Return JSON with:
- "entities": list of named entities (people, places, orgs, tools)
- "topics": list of topic categories (e.g., "work", "preferences", "technical")
- "sentiment": float from -1 (negative) to 1 (positive)
- "temporal_refs": list of time references (e.g., "today", "last week", "2024")
- "summary": one-sentence summary
- "importance": float from 0.0 to 1.0 rating how important this is to remember (higher for personal preferences, facts about the user, project-critical info, corrections; lower for trivial chitchat, well-known facts)

Text: {text}"""

        try:
            return await self.llm.complete_json(prompt)
        except Exception:
            return {
                "entities": [],
                "topics": [],
                "sentiment": 0.0,
                "temporal_refs": [],
                "summary": text[:200],
                "importance": 0.5,
            }

    async def parse_batch_with_importance(
        self, texts: list[str]
    ) -> list[tuple[ParsedInput, float]]:
        """Batch parse + importance in a single LLM call + batch embedding.

        Returns list of (ParsedInput, importance) tuples.
        """
        if not texts:
            return []

        # Batch embed all texts at once
        embeddings = await self.embedder.embed(texts)

        # Single LLM call for all texts
        numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
        prompt = f"""Extract structured information from each numbered text and rate importance. Return a JSON array where each element has:
- "entities": list of named entities (people, places, orgs, tools)
- "topics": list of topic categories (e.g., "work", "preferences", "technical")
- "sentiment": float from -1 (negative) to 1 (positive)
- "temporal_refs": list of time references (e.g., "today", "last week", "2024")
- "summary": one-sentence summary
- "importance": float from 0.0 to 1.0 rating how important this is to remember

Return JSON: {{"items": [{{...}}, ...]}}

Texts:
{numbered}"""

        try:
            result = await self.llm.complete_json(prompt)
            items = result.get("items", [])
        except Exception:
            items = []

        results = []
        for i, text in enumerate(texts):
            if i < len(items):
                feat = items[i]
            else:
                feat = {}
            importance = max(0.0, min(1.0, float(feat.get("importance", 0.5))))
            parsed = ParsedInput(
                raw_text=text,
                embedding=embeddings[i] if i < len(embeddings) else [],
                entities=feat.get("entities", []),
                topics=feat.get("topics", []),
                sentiment=feat.get("sentiment", 0.0),
                temporal_refs=feat.get("temporal_refs", []),
                summary=feat.get("summary", text[:200]),
            )
            results.append((parsed, importance))

        return results

    async def _extract_features(self, text: str) -> dict:
        """Use LLM to extract structured features from text."""
        prompt = f"""Extract structured information from this text. Return JSON with:
- "entities": list of named entities (people, places, orgs, tools)
- "topics": list of topic categories (e.g., "work", "preferences", "technical")
- "sentiment": float from -1 (negative) to 1 (positive)
- "temporal_refs": list of time references (e.g., "today", "last week", "2024")
- "summary": one-sentence summary

Text: {text}"""

        try:
            return await self.llm.complete_json(prompt)
        except Exception:
            # Graceful fallback — don't let feature extraction failure block encoding
            return {
                "entities": [],
                "topics": [],
                "sentiment": 0.0,
                "temporal_refs": [],
                "summary": text[:200],
            }
