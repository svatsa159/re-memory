"""Embedding dispatch utilities."""

from __future__ import annotations

from ..config import Config
from ..providers.base import EmbeddingProvider, get_providers


async def get_embedder(config: Config) -> EmbeddingProvider:
    """Get the configured embedding provider."""
    _, embedder = get_providers(config)
    return embedder
