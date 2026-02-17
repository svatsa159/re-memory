"""Tests for configuration management."""

from pathlib import Path

from re_memory.config import Config, load_config, save_config


def test_default_config():
    config = Config()
    assert config.llm.provider == "ollama"
    assert config.embedding.dimensions == 768
    assert config.memory.working_memory_slots == 7


def test_save_and_load(tmp_path):
    config = Config(data_dir=str(tmp_path / ".re-memory"))
    config.llm.model = "custom-model"

    path = save_config(config, tmp_path / "test_config.toml")
    assert path.exists()

    loaded = load_config(path)
    assert loaded.llm.model == "custom-model"


def test_data_path_expansion():
    config = Config(data_dir="~/.re-memory")
    assert config.data_path == Path.home() / ".re-memory"
