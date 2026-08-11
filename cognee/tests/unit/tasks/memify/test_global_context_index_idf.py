import math

import pytest

from cognee.tasks.memify.global_context_index.bucketing.graph.scoring import (
    combined_similarity,
    compute_idf_from_counts,
    cosine_distance,
    entities_weight,
    entity_weight,
    pattern_similarity,
    relationship_match,
    type_similarity,
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


def test_type_similarity_is_weighted_jaccard_over_type_ids():
    type_weights = {"person": 2.0, "location": 1.0}

    score = type_similarity({"person", "location"}, {"person"}, type_weights)

    assert score == weighted_jaccard({"person", "location"}, {"person"}, type_weights)


def test_combined_similarity_defaults_reproduce_entity_score_alone():
    assert combined_similarity(0.4, 0.9, 0.7) == pytest.approx(0.4)
    assert combined_similarity(0.0, 1.0, 1.0) == pytest.approx(0.0)


def test_combined_similarity_lets_type_score_matter_when_entity_score_is_zero():
    score = combined_similarity(
        entity_score=0.0,
        type_score=0.8,
        pattern_score=0.0,
        entity_weight=0.7,
        type_weight=0.3,
        pattern_weight=0.0,
    )

    assert score > 0.0
    assert score == pytest.approx(0.24)


def test_combined_similarity_returns_zero_when_all_weights_are_zero():
    assert combined_similarity(0.9, 0.9, 0.9, 0.0, 0.0, 0.0) == 0.0


def test_cosine_distance_is_zero_for_identical_direction_vectors():
    assert cosine_distance([1.0, 2.0], [2.0, 4.0]) == pytest.approx(0.0)


def test_cosine_distance_is_one_for_orthogonal_vectors():
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_cosine_distance_is_two_for_opposite_vectors():
    assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)


def test_cosine_distance_returns_one_for_zero_vector():
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0


def test_relationship_match_is_true_for_identical_names_without_embeddings():
    assert relationship_match("goes_to", "goes_to", {}, distance_threshold=0.5) is True


def test_relationship_match_uses_embedding_distance_for_different_names():
    embeddings = {
        "goes_to": [1.0, 0.0],
        "travels_to": [0.9, 0.1],
        "unrelated": [0.0, 1.0],
    }

    assert relationship_match("goes_to", "travels_to", embeddings, distance_threshold=0.5) is True
    assert relationship_match("goes_to", "unrelated", embeddings, distance_threshold=0.5) is False


def test_relationship_match_is_false_when_embedding_missing():
    assert relationship_match("goes_to", "travels_to", {}, distance_threshold=0.5) is False


def test_pattern_similarity_discards_pair_on_relation_mismatch():
    score = pattern_similarity(
        ("alice", "alps", "goes_to"),
        ("alice", "alps", "lives_in"),
        entity_type_by_entity_id={},
        idf_weights={"alice": 1.5, "alps": 2.0},
        type_idf_weights={},
        edge_type_embeddings={},
        distance_threshold=0.5,
    )

    assert score == 0.0


def test_pattern_similarity_is_one_for_identical_endpoints_and_relation():
    score = pattern_similarity(
        ("alice", "alps", "goes_to"),
        ("alice", "alps", "goes_to"),
        entity_type_by_entity_id={},
        idf_weights={"alice": 1.5, "alps": 2.0},
        type_idf_weights={},
        edge_type_embeddings={},
        distance_threshold=0.5,
    )

    assert score == pytest.approx(1.0)


def test_pattern_similarity_scores_same_type_different_entity_endpoints():
    """Alice/goes_to/Alps vs Bob/goes_to/Balkans: no shared entity at all, but
    both endpoints share an entity type and the relation matches exactly."""
    entity_type_by_entity_id = {
        "alice": "person",
        "bob": "person",
        "alps": "location",
        "balkans": "location",
    }
    type_idf_weights = {"person": 1.0, "location": 1.0}

    score = pattern_similarity(
        ("alice", "alps", "goes_to"),
        ("bob", "balkans", "goes_to"),
        entity_type_by_entity_id=entity_type_by_entity_id,
        idf_weights={},
        type_idf_weights=type_idf_weights,
        edge_type_embeddings={},
        distance_threshold=0.5,
        entity_weight=0.0,
        type_weight=1.0,
    )

    assert score == pytest.approx(1.0)


def test_pattern_similarity_uses_embedding_distance_for_relationship_match():
    embeddings = {"goes_to": [1.0, 0.0], "travels_to": [0.95, 0.05]}

    score = pattern_similarity(
        ("alice", "alps", "goes_to"),
        ("alice", "alps", "travels_to"),
        entity_type_by_entity_id={},
        idf_weights={"alice": 1.5, "alps": 2.0},
        type_idf_weights={},
        edge_type_embeddings=embeddings,
        distance_threshold=0.5,
    )

    assert score == pytest.approx(1.0)
