"""Neocortex: Long-Term Storage Coordinator

The neocortex is the final destination for consolidated memories.
It coordinates between the knowledge graph (semantic memory) and
schema summaries (abstract knowledge).

Handles:
- Triple extraction from episodic events → knowledge graph
- Schema generation from related triples → category summaries
- Conflict detection and resolution
"""

from __future__ import annotations

from ..providers.base import LLMProvider
from ..storage.graph import GraphStore
from ..storage.file_store import FileStore


class Neocortex:
    """Long-term memory coordinator: KG + Schemas."""

    def __init__(self, llm: LLMProvider, graph: GraphStore, file_store: FileStore):
        self.llm = llm
        self.graph = graph
        self.file_store = file_store

    async def extract_triples(self, text: str, source_event_id: str = "") -> list[dict]:
        """Extract subject-predicate-object triples from text using LLM."""
        prompt = f"""Extract knowledge triples from this text.
Return JSON: {{"triples": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}
Only extract factual, specific information. Skip vague or opinion-based content.

Text: {text}"""

        try:
            result = await self.llm.complete_json(prompt)
            triples = result.get("triples", [])

            # Store each triple in the graph
            for triple in triples:
                # Check for conflicts first
                conflicts = self.graph.get_conflicting_triples(
                    triple["subject"], triple["predicate"]
                )
                for conflict in conflicts:
                    if conflict["object"] != triple["object"]:
                        self.graph.archive_triple(
                            conflict["subject"],
                            conflict["predicate"],
                            conflict["object"],
                        )

                self.graph.add_triple(
                    subject=triple["subject"],
                    predicate=triple["predicate"],
                    obj=triple["object"],
                    source_event_id=source_event_id,
                )

            return triples
        except Exception:
            return []

    async def update_schema(self, category: str, triples: list[dict]):
        """Update a schema summary based on related triples."""
        existing = self.file_store.get(category)

        triples_text = "\n".join(
            f"- {t['subject']} {t['predicate']} {t['object']}" for t in triples
        )

        prompt = f"""{"Update this" if existing else "Create a"} knowledge summary for the category "{category}".

{f"Existing summary:{chr(10)}{existing}" if existing else ""}

New knowledge triples:
{triples_text}

Write a concise, well-organized markdown summary that incorporates all known facts.
Focus on what's most important and actionable. Keep it under 500 words."""

        try:
            summary = await self.llm.complete(prompt)
            self.file_store.put(category, summary)
            return summary
        except Exception:
            return existing or ""
