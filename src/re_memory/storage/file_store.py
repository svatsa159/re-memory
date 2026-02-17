"""File-based schema memory store.

Layer 4 schemas are stored as markdown files — one per category/topic.
These are human-readable, token-efficient summaries of knowledge.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class FileStore:
    """Markdown-file-based schema storage."""

    def __init__(self, schema_dir: str | Path):
        self.schema_dir = Path(schema_dir).expanduser()

    def init(self):
        """Ensure the schema directory exists."""
        self.schema_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, category: str) -> Path:
        """Get the path for a category file."""
        safe_name = category.lower().replace(" ", "_").replace("/", "_")
        return self.schema_dir / f"{safe_name}.md"

    def get(self, category: str) -> str | None:
        """Read a schema by category."""
        path = self._path(category)
        if not path.exists():
            return None
        return path.read_text()

    def put(self, category: str, content: str):
        """Write or update a schema."""
        self.init()
        path = self._path(category)
        now = datetime.now(timezone.utc).isoformat()
        header = f"<!-- updated: {now} -->\n"
        path.write_text(header + content)

    def list_categories(self) -> list[str]:
        """List all schema categories."""
        if not self.schema_dir.exists():
            return []
        return [
            p.stem.replace("_", " ")
            for p in sorted(self.schema_dir.glob("*.md"))
        ]

    def search(self, query: str) -> list[dict]:
        """Search across all schemas for matching content."""
        results = []
        query_lower = query.lower()
        for path in self.schema_dir.glob("*.md"):
            content = path.read_text()
            if query_lower in content.lower():
                results.append({
                    "category": path.stem.replace("_", " "),
                    "path": str(path),
                    "snippet": _extract_snippet(content, query_lower),
                })
        return results

    def delete(self, category: str) -> bool:
        """Delete a schema file."""
        path = self._path(category)
        if path.exists():
            path.unlink()
            return True
        return False


def _extract_snippet(text: str, query: str, context_chars: int = 100) -> str:
    """Extract a snippet around the first occurrence of query."""
    idx = text.lower().find(query)
    if idx == -1:
        return text[:200]
    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(query) + context_chars)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet
