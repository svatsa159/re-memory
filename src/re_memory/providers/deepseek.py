"""DeepSeek provider for cloud LLM inference.

DeepSeek uses an OpenAI-compatible API at https://api.deepseek.com.
"""

from __future__ import annotations

import json

from openai import AsyncOpenAI

from .base import LLMProvider

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekLLM(LLMProvider):
    """DeepSeek cloud LLM (OpenAI-compatible API)."""

    def __init__(self, model: str = "deepseek-reasoner", api_key: str = ""):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key or None,
            base_url=DEEPSEEK_BASE_URL,
        )

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
            **kwargs,
        )
        text = resp.choices[0].message.content or "{}"
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
