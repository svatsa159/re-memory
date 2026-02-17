"""Ollama provider for local LLM and embedding inference."""

from __future__ import annotations

import json

import httpx

from .base import EmbeddingProvider, LLMProvider


class OllamaLLM(LLMProvider):
    """Local LLM via Ollama HTTP API."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    **kwargs,
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def complete_json(self, prompt: str, system: str = "", **kwargs) -> dict:
        system_with_json = (
            (system + "\n\n" if system else "")
            + "Respond with valid JSON only. No markdown, no explanation."
        )
        text = await self.complete(
            prompt,
            system=system_with_json,
            format="json",
            **kwargs,
        )
        return json.loads(text)


class OllamaEmbedding(EmbeddingProvider):
    """Local embeddings via Ollama HTTP API."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dims: int = 768,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                resp = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text},
                )
                resp.raise_for_status()
                data = resp.json()
                results.append(data["embeddings"][0])
        return results
