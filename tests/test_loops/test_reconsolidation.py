"""Tests for the reconsolidation logic."""

from re_memory.loops.reconsolidation import (
    ReconsolidationAction,
    check_reconsolidation,
)


def test_low_pe_confirms():
    action = check_reconsolidation("old memory", "similar context", 0.1)
    assert action == ReconsolidationAction.CONFIRM


def test_high_pe_bifurcates():
    action = check_reconsolidation("old memory", "very different context", 0.8)
    assert action == ReconsolidationAction.BIFURCATE


def test_medium_pe_updates():
    action = check_reconsolidation("old memory", "slightly different", 0.5)
    assert action == ReconsolidationAction.UPDATE
