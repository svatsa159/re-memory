"""LLM dispatch utilities."""

from __future__ import annotations

from ..config import Config
from ..providers.base import LLMProvider, get_providers


async def get_llm(config: Config) -> LLMProvider:
    """Get the configured LLM provider."""
    llm, _ = get_providers(config)
    return llm
