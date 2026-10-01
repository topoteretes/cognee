"""Unit tests for temporal contradiction resolution (issue #3631, Approach E).

Pure-Python and deterministic: no LLM, no database, no network.
"""

import copy

from cognee.modules.graph.utils import tag_superseded_edges


def _edge(source, target, relationship_name, updated_at="", edge_object_id=None, **extra):
    props = {"updated_at": updated_at, **extra}
    if edge_object_id is not None:
        props["edge_object_id"] = edge_object_id
    return (source, target, relationship_name, props)


def test_no_functional_relationships_is_a_noop():
    edges = [_edge("c", "alice", "ceo_of", "2020-01-01 00:00:00")]
    assert tag_superseded_edges(edges, set()) == []


def test_empty_edges():
    assert tag_superseded_edges([], {"ceo_of"}) == []


def test_single_target_is_not_a_contradiction():
    # Same target re-asserted at a later time is not a conflict.
    edges = [
        _edge("c", "alice", "ceo_of", "2020-01-01 00:00:00"),
        _edge("c", "alice", "ceo_of", "2024-01-01 00:00:00"),
    ]
    assert tag_superseded_edges(edges, {"ceo_of"}) == []


def test_conflicting_functional_relationship_keeps_most_recent():
    edges = [
        _edge("c", "alice", "ceo_of", "2020-01-01 00:00:00", edge_object_id="e-alice"),
        _edge("c", "bob", "ceo_of", "2024-01-01 00:00:00", edge_object_id="e-bob"),
    ]
    superseded = tag_superseded_edges(edges, {"ceo_of"})

    # Only the older assertion is tagged; the winner (bob) is left current.
    assert len(superseded) == 1
    old = superseded[0]
    assert old[1] == "alice"
    assert old[3]["superseded"] is True
    assert old[3]["superseded_by"] == "e-bob"
    assert "ceo_of" in old[3]["supersession_reason"]


def test_many_valued_relationship_is_never_collapsed():
    # 'mentions' is NOT declared functional: two different targets both survive.
    edges = [
        _edge("doc", "alice", "mentions", "2020-01-01 00:00:00"),
        _edge("doc", "bob", "mentions", "2024-01-01 00:00:00"),
    ]
    assert tag_superseded_edges(edges, {"ceo_of"}) == []


def test_recency_ties_break_by_position():
    # Identical timestamps -> the later assertion wins.
    edges = [
        _edge("c", "alice", "ceo_of", "2024-01-01 00:00:00", edge_object_id="e-alice"),
        _edge("c", "bob", "ceo_of", "2024-01-01 00:00:00", edge_object_id="e-bob"),
    ]
    superseded = tag_superseded_edges(edges, {"ceo_of"})
    assert [e[1] for e in superseded] == ["alice"]
    assert superseded[0][3]["superseded_by"] == "e-bob"


def test_input_is_not_mutated():
    edges = [
        _edge("c", "alice", "ceo_of", "2020-01-01 00:00:00", edge_object_id="e-alice"),
        _edge("c", "bob", "ceo_of", "2024-01-01 00:00:00", edge_object_id="e-bob"),
    ]
    before = copy.deepcopy(edges)
    tag_superseded_edges(edges, {"ceo_of"})
    assert edges == before


def test_existing_properties_are_preserved_on_superseded_edge():
    edges = [
        _edge(
            "c",
            "alice",
            "ceo_of",
            "2020-01-01 00:00:00",
            edge_object_id="e-alice",
            source_ref_keys="ds:data",
            feedback_weight=0.9,
        ),
        _edge("c", "bob", "ceo_of", "2024-01-01 00:00:00", edge_object_id="e-bob"),
    ]
    old = tag_superseded_edges(edges, {"ceo_of"})[0][3]
    assert old["source_ref_keys"] == "ds:data"
    assert old["feedback_weight"] == 0.9
    assert old["superseded"] is True


def test_mixed_batch_tags_only_functional_conflicts():
    edges = [
        _edge("c", "alice", "ceo_of", "2020-01-01 00:00:00", edge_object_id="e-alice"),
        _edge("doc", "alice", "mentions", "2020-01-01 00:00:00"),
        _edge("c", "bob", "ceo_of", "2024-01-01 00:00:00", edge_object_id="e-bob"),
        _edge("doc", "bob", "mentions", "2024-01-01 00:00:00"),
        _edge("c2", "carol", "located_in", "2021-01-01 00:00:00"),
    ]
    superseded = tag_superseded_edges(edges, {"ceo_of"})
    assert [(e[0], e[1], e[2]) for e in superseded] == [("c", "alice", "ceo_of")]


def test_multiple_independent_subjects_resolved_independently():
    edges = [
        _edge("acme", "alice", "ceo_of", "2020-01-01 00:00:00", edge_object_id="a-alice"),
        _edge("acme", "bob", "ceo_of", "2024-01-01 00:00:00", edge_object_id="a-bob"),
        _edge("globex", "carol", "ceo_of", "2019-01-01 00:00:00", edge_object_id="g-carol"),
        _edge("globex", "dave", "ceo_of", "2023-01-01 00:00:00", edge_object_id="g-dave"),
    ]
    superseded = tag_superseded_edges(edges, {"ceo_of"})
    superseded_by_target = {e[1]: e[3]["superseded_by"] for e in superseded}
    assert superseded_by_target == {"alice": "a-bob", "carol": "g-dave"}


def test_superseded_edges_keep_input_order():
    edges = [
        _edge("acme", "alice", "ceo_of", "2019-01-01 00:00:00", edge_object_id="a-alice"),
        _edge("globex", "carol", "ceo_of", "2019-01-01 00:00:00", edge_object_id="g-carol"),
        _edge("acme", "bob", "ceo_of", "2024-01-01 00:00:00", edge_object_id="a-bob"),
        _edge("globex", "dave", "ceo_of", "2023-01-01 00:00:00", edge_object_id="g-dave"),
    ]
    superseded = tag_superseded_edges(edges, {"ceo_of"})
    assert [e[1] for e in superseded] == ["alice", "carol"]
