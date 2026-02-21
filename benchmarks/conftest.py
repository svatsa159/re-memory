"""Pytest fixtures for benchmark tests."""

from __future__ import annotations

import pytest

from benchmarks.adapter import EngineAdapter


@pytest.fixture
def adapter():
    """Create an isolated EngineAdapter for testing."""
    a = EngineAdapter.create_isolated(run_id="pytest", tier="fast")
    yield a
    a.close()
