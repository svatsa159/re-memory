"""Configuration management for re-memory.

Loads config from (in priority order):
1. ./re_memory.toml (project-local)
2. ~/.re-memory/config.toml (user-global)
3. Defaults
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import tomli_w


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    api_key: str = ""


class EmbeddingConfig(BaseModel):
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    dimensions: int = 768
    api_key: str = ""


class StorageConfig(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    sqlite_path: str = "~/.re-memory/events.db"
    schema_dir: str = "~/.re-memory/schemas"


class ConsolidationConfig(BaseModel):
    interval_hours: int = 24
    decay_rate: float = 0.1
    confidence_threshold: float = 0.3
    max_episodic_age_days: int = 30


class MemoryConfig(BaseModel):
    working_memory_slots: int = 7
    max_retrieval_tokens: int = 2000
    novelty_threshold: float = Field(default=0.3, description="CA1 prediction error threshold")
    importance_threshold: float = Field(default=0.5, description="Amygdala salience gate")
    observe_tier: str = Field(default="standard", description="Observe pipeline tier: fast, standard, full")


class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    storage: StorageConfig = StorageConfig()
    consolidation: ConsolidationConfig = ConsolidationConfig()
    memory: MemoryConfig = MemoryConfig()
    data_dir: str = "~/.re-memory"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser()

    @property
    def sqlite_path(self) -> Path:
        return Path(self.storage.sqlite_path).expanduser()

    @property
    def schema_path(self) -> Path:
        return Path(self.storage.schema_dir).expanduser()


def _find_config_file() -> Path | None:
    """Find config file in priority order."""
    candidates = [
        Path.cwd() / "re_memory.toml",
        Path.home() / ".re-memory" / "config.toml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_config(path: Path | None = None) -> Config:
    """Load configuration from TOML file, falling back to defaults."""
    if path is None:
        path = _find_config_file()

    if path is not None and path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return Config(**data)

    return Config()


def save_config(config: Config, path: Path | None = None) -> Path:
    """Save configuration to TOML file."""
    if path is None:
        path = config.data_path / "config.toml"

    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump()
    with open(path, "wb") as f:
        tomli_w.dump(data, f)
    return path
