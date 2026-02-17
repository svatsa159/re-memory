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
