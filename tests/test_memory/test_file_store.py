"""Tests for the file-based schema store."""

import pytest

from re_memory.storage.file_store import FileStore


@pytest.fixture
def store(tmp_path):
    return FileStore(tmp_path / "schemas")


def test_put_and_get(store):
    store.put("work", "# Work\nUser works at OpenAI.")
    content = store.get("work")
    assert "OpenAI" in content


def test_get_nonexistent(store):
    assert store.get("nonexistent") is None


def test_list_categories(store):
    store.put("work", "work stuff")
    store.put("hobbies", "hobby stuff")
    cats = store.list_categories()
    assert "work" in cats
    assert "hobbies" in cats


def test_search(store):
    store.put("tech", "User prefers Python and Rust")
    store.put("food", "User likes pizza")
    results = store.search("Python")
    assert len(results) == 1
    assert results[0]["category"] == "tech"


def test_delete(store):
    store.put("temp", "temporary")
    assert store.delete("temp")
    assert store.get("temp") is None
