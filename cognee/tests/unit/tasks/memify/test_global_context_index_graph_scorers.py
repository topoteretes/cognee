import pytest

from cognee.tasks.memify.global_context_index.bucketing.graph.scorers import (
    build_combined_similarity,
    build_entity_group_profile,
    build_entity_jaccard,
    build_pattern_index,
    build_pattern_similarity,
    build_scorers,
    build_type_jaccard,
)
from cognee.tasks.memify.global_context_index.bucketing.graph.scoring import weighted_jaccard


def _profile(entity_ids, entity_type_by_entity_id=None, relation_index=None):
    return build_entity_group_profile(
        entity_ids, entity_type_by_entity_id or {}, relation_index or {}
    )


def test_build_entity_group_profile_derives_type_ids_and_relations():
    entity_type_by_entity_id = {"alice": "person", "acme": "org"}
    relation_index = build_pattern_index([("alice", "acme", "works_at")])

    profile = _profile({"alice", "acme"}, entity_type_by_entity_id, relation_index)

    assert profile.entity_ids == frozenset({"alice", "acme"})
    assert profile.type_ids == frozenset({"person", "org"})
    assert profile.relations == (("alice", "acme", "works_at"),)


def test_build_entity_group_profile_excludes_relations_with_an_endpoint_outside_the_set():
    relation_index = build_pattern_index([("alice", "acme", "works_at")])

    profile = _profile({"alice"}, relation_index=relation_index)

    assert profile.relations == ()


def test_entity_jaccard_matches_weighted_jaccard():
    idf_weights = {"alice": 2.0, "bob": 1.0, "carol": 3.0}
    entity_jaccard = build_entity_jaccard(idf_weights)

    left = _profile({"alice", "bob"})
    right = _profile({"bob", "carol"})

    assert entity_jaccard(left, right) == pytest.approx(
        weighted_jaccard({"alice", "bob"}, {"bob", "carol"}, idf_weights)
    )


def test_type_jaccard_compares_type_ids_not_entity_ids():
    entity_type_by_entity_id = {"alice": "person", "bob": "person", "acme": "org"}
    type_idf_weights = {"person": 1.0, "org": 1.0}
    type_jaccard = build_type_jaccard(type_idf_weights)

    # "alice" and "bob" share no entity id but share the same type -> full score.
    left = _profile({"alice"}, entity_type_by_entity_id)
    right = _profile({"bob"}, entity_type_by_entity_id)

    assert type_jaccard(left, right) == pytest.approx(1.0)


def test_pattern_similarity_zero_when_either_side_has_no_relations():
    pattern_similarity = build_pattern_similarity(
        entity_type_by_entity_id={},
        idf_weights={},
        type_idf_weights={},
        edge_type_embeddings={},
        pattern_distance_threshold=0.5,
        entity_weight=1.0,
        type_weight=0.0,
    )
    relation_index = build_pattern_index([("alice", "acme", "works_at")])

    with_relations = _profile({"alice", "acme"}, relation_index=relation_index)
    without_relations = _profile({"bob"})

    assert pattern_similarity(with_relations, without_relations) == 0.0


def test_pattern_similarity_positive_for_matching_relationship_and_entities():
    relation_index = build_pattern_index(
        [("alice", "acme", "works_at"), ("alice", "acme", "works_at")]
    )
    pattern_similarity = build_pattern_similarity(
        entity_type_by_entity_id={},
        idf_weights={"alice": 1.0, "acme": 1.0},
        type_idf_weights={},
        edge_type_embeddings={},
        pattern_distance_threshold=0.5,
        entity_weight=1.0,
        type_weight=0.0,
    )

    profile = _profile({"alice", "acme"}, relation_index=relation_index)

    # identical entities on both sides of an identical relationship -> full score.
    assert pattern_similarity(profile, profile) == pytest.approx(1.0)


def test_build_scorers_combined_matches_entity_only_by_default():
    idf_weights = {"alice": 1.0, "bob": 1.0, "carol": 1.0}
    scorers = build_scorers(
        idf_weights=idf_weights,
        entity_type_by_entity_id={},
        type_idf_weights={},
        edge_type_embeddings={},
        pattern_distance_threshold=0.5,
        entity_weight=1.0,
        type_weight=0.0,
        pattern_weight=0.0,
    )

    left = _profile({"alice", "bob"})
    right = _profile({"bob", "carol"})

    # Default weights (entity=1, type=0, pattern=0): combined == entity alone.
    assert scorers["combined"](left, right) == pytest.approx(scorers["entity"](left, right))


def test_build_scorers_combined_blends_all_three_signals_when_weighted_equally():
    entity_type_by_entity_id = {"alice": "person", "bob": "person", "acme": "org", "globex": "org"}
    relation_index = build_pattern_index(
        [("alice", "acme", "works_at"), ("bob", "globex", "works_at")]
    )
    idf_weights = {"alice": 1.0, "bob": 1.0, "acme": 1.0, "globex": 1.0}
    type_idf_weights = {"person": 1.0, "org": 1.0}

    scorers = build_scorers(
        idf_weights=idf_weights,
        entity_type_by_entity_id=entity_type_by_entity_id,
        type_idf_weights=type_idf_weights,
        edge_type_embeddings={},
        pattern_distance_threshold=0.5,
        entity_weight=1 / 3,
        type_weight=1 / 3,
        pattern_weight=1 / 3,
    )

    left = _profile({"alice", "acme"}, entity_type_by_entity_id, relation_index)
    right = _profile({"bob", "globex"}, entity_type_by_entity_id, relation_index)

    entity_score = scorers["entity"](left, right)
    type_score = scorers["type"](left, right)
    pattern_score = scorers["pattern"](left, right)
    expected = (entity_score + type_score + pattern_score) / 3

    # Zero shared entities, but same types and same relationship pattern. The
    # pattern score itself blends entity+type per endpoint (using the same
    # entity_weight/type_weight), so it lands at 0.5 (0 entity-match, 1.0
    # type-match, averaged), not a clean 1.0.
    assert entity_score == pytest.approx(0.0)
    assert type_score == pytest.approx(1.0)
    assert pattern_score == pytest.approx(0.5)
    assert scorers["combined"](left, right) == pytest.approx(expected)


def test_build_scorers_skips_pattern_computation_when_pattern_weight_is_zero():
    calls = []
    relation_index = build_pattern_index([("alice", "acme", "works_at")])

    def counting_pattern_similarity(summary, bucket):
        calls.append((summary, bucket))
        return 1.0

    entity_jaccard = build_entity_jaccard({"alice": 1.0, "acme": 1.0})
    type_jaccard = build_type_jaccard({})
    combined_similarity = build_combined_similarity(
        entity_jaccard,
        type_jaccard,
        counting_pattern_similarity,
        entity_weight=1.0,
        type_weight=0.0,
        pattern_weight=0.0,
    )

    profile = _profile({"alice", "acme"}, relation_index=relation_index)
    combined_similarity(profile, profile)

    assert calls == []
