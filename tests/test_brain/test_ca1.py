"""Tests for CA1 novelty detection."""

from re_memory.brain.ca1 import CA1NoveltyDetector, NoveltyVerdict


def test_novel_when_no_matches():
    detector = CA1NoveltyDetector(novelty_threshold=0.3)
    result = detector.detect([0.1] * 768, [])
    assert result.verdict == NoveltyVerdict.NOVEL
    assert result.prediction_error == 1.0


def test_redundant_when_very_similar():
    detector = CA1NoveltyDetector(novelty_threshold=0.3)
    # similarity 0.95 → PE 0.05 → below threshold → redundant
    result = detector.detect([0.1] * 768, [("mem-1", 0.95)])
    assert result.verdict == NoveltyVerdict.REDUNDANT
    assert result.closest_memory_id == "mem-1"


def test_novel_when_very_different():
    detector = CA1NoveltyDetector(novelty_threshold=0.3)
    # similarity 0.1 → PE 0.9 → above (1 - threshold) → novel
    result = detector.detect([0.1] * 768, [("mem-1", 0.1)])
    assert result.verdict == NoveltyVerdict.NOVEL


def test_update_when_moderate():
    detector = CA1NoveltyDetector(novelty_threshold=0.3)
    # similarity 0.62 → PE 0.38 → in UPDATE band (0.30 to 0.45) → update
    result = detector.detect([0.1] * 768, [("mem-1", 0.62)])
    assert result.verdict == NoveltyVerdict.UPDATE


def test_novel_when_moderate_but_different_topic():
    detector = CA1NoveltyDetector(novelty_threshold=0.3)
    # similarity 0.50 → PE 0.50 → above UPDATE ceiling (0.45) → novel
    result = detector.detect([0.1] * 768, [("mem-1", 0.50)])
    assert result.verdict == NoveltyVerdict.NOVEL
