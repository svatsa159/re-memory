"""OpenAI provider for cloud LLM and embedding inference."""

from __future__ import annotations

import json

from openai import AsyncOpenAI

from .base import EmbeddingProvider, LLMProvider


class OpenAILLM(LLMProvider):
    """OpenAI cloud LLM."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = ""):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key or None)

    async def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    async def complete_json(self, prompt: str, system: str = "", **kwargs) -> dict:
        messages = []
        sys_content = (
            (system + "\n\n" if system else "")
            + "Respond with valid JSON only. No markdown, no explanation."
        )
        messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": prompt})

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        text = resp.choices[0].message.content or "{}"
        return json.loads(text)


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI cloud embeddings."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        dims: int = 768,
    ):
        self.model = model
        self._dims = dims
        self.client = AsyncOpenAI(api_key=api_key or None)

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self._dims,
        )
        return [item.embedding for item in resp.data]
