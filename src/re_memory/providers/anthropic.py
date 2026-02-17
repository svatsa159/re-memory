"""Anthropic provider for cloud LLM inference."""

from __future__ import annotations

import json

from anthropic import AsyncAnthropic

from .base import LLMProvider


class AnthropicLLM(LLMProvider):
    """Anthropic cloud LLM (Claude)."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", api_key: str = ""):
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key or None)

    async def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return resp.content[0].text

    async def complete_json(self, prompt: str, system: str = "", **kwargs) -> dict:
        sys_content = (
            (system + "\n\n" if system else "")
            + "Respond with valid JSON only. No markdown, no explanation."
        )
        text = await self.complete(prompt, system=sys_content, **kwargs)
        # Strip any markdown code fences
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
