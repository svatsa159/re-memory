"""Tests for Prefrontal Cortex working memory."""

from re_memory.brain.prefrontal import PrefrontalCortex, WorkingMemoryItem


def test_add_within_capacity():
    pfc = PrefrontalCortex(capacity=3)
    item = WorkingMemoryItem(id="1", content="test")
    evicted = pfc.add(item)
    assert evicted is None
    assert pfc.size == 1


def test_lru_eviction():
    pfc = PrefrontalCortex(capacity=2)
    pfc.add(WorkingMemoryItem(id="1", content="first"))
    pfc.add(WorkingMemoryItem(id="2", content="second"))
    evicted = pfc.add(WorkingMemoryItem(id="3", content="third"))
    assert evicted is not None
    assert evicted.id == "1"  # LRU item evicted
    assert pfc.size == 2


def test_access_moves_to_end():
    pfc = PrefrontalCortex(capacity=2)
    pfc.add(WorkingMemoryItem(id="1", content="first"))
    pfc.add(WorkingMemoryItem(id="2", content="second"))
    pfc.get("1")  # touch "1", making "2" the LRU
    evicted = pfc.add(WorkingMemoryItem(id="3", content="third"))
    assert evicted.id == "2"


def test_clear():
    pfc = PrefrontalCortex(capacity=7)
    for i in range(5):
        pfc.add(WorkingMemoryItem(id=str(i), content=f"item {i}"))
    assert pfc.size == 5
    pfc.clear()
    assert pfc.size == 0


def test_context_string():
    pfc = PrefrontalCortex(capacity=7)
    pfc.add(WorkingMemoryItem(id="1", content="hello"))
    pfc.add(WorkingMemoryItem(id="2", content="world"))
    ctx = pfc.to_context_string()
    assert "hello" in ctx
    assert "world" in ctx
