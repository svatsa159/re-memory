"""Abstract interfaces for LLM and Embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    async def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        """Generate a completion from a prompt."""
        ...

    @abstractmethod
    async def complete_json(self, prompt: str, system: str = "", **kwargs) -> dict:
        """Generate a JSON-structured completion."""
        ...


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        ...

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        result = await self.embed([text])
        return result[0]


def get_providers(config: Config) -> tuple[LLMProvider, EmbeddingProvider]:
    """Factory: create LLM and Embedding providers from config."""
    llm = _create_llm(config)
    embedder = _create_embedder(config)
    return llm, embedder


def _create_llm(config: Config) -> LLMProvider:
    match config.llm.provider:
        case "ollama":
            from .ollama import OllamaLLM
            return OllamaLLM(
                model=config.llm.model,
                base_url=config.llm.base_url,
            )
        case "openai":
            from .openai import OpenAILLM
            return OpenAILLM(
                model=config.llm.model,
                api_key=config.llm.api_key,
            )
        case "anthropic":
            from .anthropic import AnthropicLLM
            return AnthropicLLM(
                model=config.llm.model,
                api_key=config.llm.api_key,
            )
        case "deepseek":
            from .deepseek import DeepSeekLLM
            return DeepSeekLLM(
                model=config.llm.model,
                api_key=config.llm.api_key,
            )
        case _:
            raise ValueError(f"Unknown LLM provider: {config.llm.provider}")


def _create_embedder(config: Config) -> EmbeddingProvider:
    match config.embedding.provider:
        case "ollama":
            from .ollama import OllamaEmbedding
            return OllamaEmbedding(
                model=config.embedding.model,
                base_url=config.embedding.base_url,
                dims=config.embedding.dimensions,
            )
        case "openai":
            from .openai import OpenAIEmbedding
            return OpenAIEmbedding(
                model=config.embedding.model,
                api_key=config.embedding.api_key,
                dims=config.embedding.dimensions,
            )
        case "anthropic" | "deepseek":
            raise ValueError(
                f"{config.embedding.provider.title()} does not provide an embedding API. "
                "Use 'ollama' or 'openai' for embeddings."
            )
        case _:
            raise ValueError(f"Unknown embedding provider: {config.embedding.provider}")
