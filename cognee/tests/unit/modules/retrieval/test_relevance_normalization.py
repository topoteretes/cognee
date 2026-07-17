"""Unit tests for ``normalize_distance_to_relevance``.

The normalizer maps raw distance-based scores onto a shared
``[0, 1]``, higher-is-better relevance scale. These tests pin the
monotonicity property (lower score always yields higher relevance),
the boundary behavior at ``0`` and large scores, and the guard against
negative inputs from misbehaving adapters.
"""

from __future__ import annotations

import math

import pytest

from cognee.infrastructure.databases.vector.models.ScoredResult import (
    ScoredResult,
    normalize_distance_to_relevance,
)


def test_perfect_match_score_maps_to_full_relevance():
    assert normalize_distance_to_relevance(0.0) == 1.0


def test_relevance_is_monotonically_decreasing_in_score():
    prior_relevance = normalize_distance_to_relevance(0.0)
    for score in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 100.0):
        current_relevance = normalize_distance_to_relevance(score)
        assert current_relevance < prior_relevance, (
            f"relevance did not decrease when score moved forward to {score}: "
            f"prior={prior_relevance}, current={current_relevance}"
        )
        prior_relevance = current_relevance


def test_relevance_asymptotes_toward_zero_for_large_scores():
    for score in (10.0, 100.0, 1_000.0):
        relevance = normalize_distance_to_relevance(score)
        assert 0.0 < relevance < 0.5


def test_relevance_stays_in_unit_interval_at_the_bounds():
    for score in (0.0, 1.0, 2.0, 10.0, 1e6):
        relevance = normalize_distance_to_relevance(score)
        assert 0.0 <= relevance <= 1.0


def test_negative_scores_are_treated_as_perfect_matches():
    # Misbehaving adapters sometimes emit slightly-negative distance
    # scores; the normalizer clamps to 0 so callers still get a
    # meaningful upper bound rather than a >1 relevance.
    for score in (-0.001, -1.0, -100.0):
        assert normalize_distance_to_relevance(score) == 1.0


def test_scored_result_defaults_relevance_to_none():
    result = ScoredResult(id="00000000-0000-0000-0000-000000000001", score=0.5)
    assert result.relevance is None


def test_scored_result_accepts_explicit_relevance_from_adapter():
    result = ScoredResult(
        id="00000000-0000-0000-0000-000000000001",
        score=0.5,
        relevance=0.9,
    )
    assert result.relevance == pytest.approx(0.9)


def test_scored_result_relevance_can_be_populated_via_helper():
    score = 0.25
    expected = normalize_distance_to_relevance(score)
    result = ScoredResult(
        id="00000000-0000-0000-0000-000000000001",
        score=score,
        relevance=expected,
    )
    assert result.relevance == pytest.approx(expected)
    assert not math.isnan(result.relevance)
