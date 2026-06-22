import math

import pytest

from cognee.tasks.memify.global_context_index.bucketing.graph.scoring import (
    compute_idf_from_counts,
    entities_weight,
    entity_weight,
    weighted_jaccard,
)


def test_compute_idf_from_counts_uses_expected_math():
    weights = compute_idf_from_counts(
        4,
        {
            "alice": 1,
            "project-x": 2,
            "standup": 4,
        },
    )

    assert weights == pytest.approx(
        {
            "alice": math.log(4),
            "project-x": math.log(2),
            "standup": 0.0,
        }
    )


def test_compute_idf_from_counts_returns_empty_for_empty_population():
    assert compute_idf_from_counts(0, {"alice": 1}) == {}
    assert compute_idf_from_counts(-1, {"alice": 1}) == {}


def test_compute_idf_from_counts_ignores_non_positive_entity_counts():
    weights = compute_idf_from_counts(
        3,
        {
            "alice": 1,
            "missing": 0,
            "invalid": -1,
        },
    )

    assert weights == pytest.approx({"alice": math.log(3)})


def test_compute_idf_from_counts_rejects_entity_count_greater_than_chunk_count():
    with pytest.raises(ValueError, match="greater than chunk_count"):
        compute_idf_from_counts(2, {"alice": 3})


def test_entity_weight_returns_zero_for_missing_entity():
    assert entity_weight("missing", {"alice": 1.5}) == 0.0


def test_entities_weight_deduplicates_entities_and_uses_missing_weight_zero():
    assert entities_weight(["alice", "alice", "missing"], {"alice": 1.5}) == pytest.approx(1.5)


def test_weighted_jaccard_returns_zero_for_zero_weight_union():
    assert weighted_jaccard({"standup"}, {"standup", "missing"}, {"standup": 0.0}) == 0.0


def test_weighted_jaccard_scores_weighted_overlap():
    weights = {
        "alice": 2.0,
        "project-x": 1.0,
        "bob": 3.0,
    }

    score = weighted_jaccard(
        {"alice", "project-x"},
        {"project-x", "bob"},
        weights,
    )

    assert score == pytest.approx(1.0 / 6.0)


def test_ubiquitous_entities_get_zero_weight():
    weights = compute_idf_from_counts(3, {"standup": 3})

    assert weights["standup"] == pytest.approx(0.0)
