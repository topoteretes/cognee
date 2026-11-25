"""Merging two hybrid retrievals keeps the result shape and its per-channel budgets."""

from unittest.mock import MagicMock

import pytest

from cognee.modules.retrieval.hybrid.merge import (
    merge_hybrid_results,
    merge_hybrid_results_union,
)

EMPTY_CHANNELS = {"chunks": [], "entities": [], "facts": []}


def result(identifier, text):
    item = MagicMock()
    item.id = identifier
    item.payload = {"id": identifier, "text": text}
    return item


def item_id(item):
    return item["id"] if isinstance(item, dict) else item.id


def test_merge_preserves_caps_summaries_and_unowned_primary_keys():
    """Fixtures match HybridRetriever's real output: chunks, summaries, entities, facts."""
    primary_chunk = result("chunk-1", "primary")
    primary = {
        "chunks": [primary_chunk],
        "chunk_summaries": {"chunk-1": "primary summary", "dropped": "drop"},
        "entities": [{"id": "entity-1", "name": "Primary"}],
        "facts": [],
        # A key this module does not own; it should survive untouched.
        "global_context": "built once from the raw query",
    }
    secondary = {
        "chunks": [result("chunk-1", "secondary"), result("chunk-2", "second")],
        "chunk_summaries": {"chunk-1": "secondary summary", "chunk-2": "second"},
        "entities": [{"id": "entity-1", "name": "Secondary"}],
        "facts": [{"id": "fact-1", "text": "fact"}],
        "global_context": "ignored",
    }

    merged = merge_hybrid_results(
        primary,
        secondary,
        chunks_limit=1,
        entities_limit=1,
        facts_limit=1,
    )

    assert merged["chunks"] == [primary_chunk]
    # Summaries follow the chunks that survived the cap; "dropped" and "chunk-2" go.
    assert merged["chunk_summaries"] == {"chunk-1": "primary summary"}
    assert merged["entities"] == [primary["entities"][0]]
    assert merged["facts"] == secondary["facts"]
    assert merged["global_context"] == "built once from the raw query"
    assert set(merged) == {"chunks", "chunk_summaries", "entities", "facts", "global_context"}


def test_merge_of_the_real_shape_adds_no_extra_keys():
    merged = merge_hybrid_results(
        {"chunks": [result("chunk-1", "one")], "entities": [], "facts": []},
        None,
        chunks_limit=5,
        entities_limit=5,
        facts_limit=5,
    )

    assert set(merged) == {"chunks", "chunk_summaries", "entities", "facts"}


@pytest.mark.parametrize(
    ("channel", "make_item"),
    [
        ("chunks", lambda identifier: result(identifier, identifier)),
        ("entities", lambda identifier: {"id": identifier, "name": identifier}),
        ("facts", lambda identifier: {"id": identifier, "text": identifier}),
    ],
    ids=["chunks", "entities", "facts"],
)
def test_each_channel_reserves_slots_for_conversational_only_items(channel, make_item):
    """Both lanes fill the channel's budget; the reserve still admits ctx-only hits."""
    limit = 5
    primary_items = [make_item(f"raw{index}") for index in range(limit)]
    # "raw3" is found by both lanes so it ranks first; every "ctx" is conversational-only.
    secondary_items = [make_item(name) for name in ("ctx0", "ctx1", "raw3", "ctx2", "ctx3")]
    limits = {"chunks_limit": 1, "entities_limit": 1, "facts_limit": 1}
    limits[f"{channel}_limit"] = limit

    merged = merge_hybrid_results(
        {**EMPTY_CHANNELS, channel: primary_items},
        {**EMPTY_CHANNELS, channel: secondary_items},
        **limits,
    )

    # One reserved slot at limit=5, so the lowest-ranked raw item yields to "ctx0".
    assert [item_id(item) for item in merged[channel]] == [
        "raw3",
        "raw0",
        "raw1",
        "raw2",
        "ctx0",
    ]


# ------------------------------------------------------------- N-way union


def _leg(chunks=(), entities=(), facts=(), summaries=None, **extra):
    return {
        "chunks": list(chunks),
        "chunk_summaries": dict(summaries or {}),
        "entities": list(entities),
        "facts": list(facts),
        **extra,
    }


def _edge(text, edge_type_id=None, source="s", target="t"):
    return {
        "text": text,
        "source_id": source,
        "relationship": text,
        "target_id": target,
        "edge_type_id": edge_type_id,
    }


def test_union_interleaves_results_round_robin_and_dedupes_by_id():
    pass_one = _leg(chunks=[result("c1", "pass one")])
    leg_a = _leg(chunks=[result("c2", "a"), result("c3", "from a")])
    leg_b = _leg(chunks=[result("c3", "from b"), result("c4", "b")])

    merged = merge_hybrid_results_union([pass_one, leg_a, leg_b])

    assert [item_id(chunk) for chunk in merged["chunks"]] == ["c1", "c2", "c3", "c4"]
    # c3 sits at rank 1 for leg a and rank 0 for leg b: it is placed by its best rank and
    # represented by that result's object.
    assert merged["chunks"][2].payload["text"] == "from b"


def test_union_keeps_every_item_of_every_result():
    legs = [
        _leg(
            chunks=[result(f"{leg}-{rank}", "t") for rank in range(3)],
            facts=[{"id": f"f-{leg}-{rank}", "text": "fact"} for rank in range(3)],
        )
        for leg in ("p", "a", "b")
    ]

    merged = merge_hybrid_results_union(legs)

    assert len(merged["chunks"]) == len(merged["facts"]) == 9
    assert [item_id(chunk) for chunk in merged["chunks"][:3]] == ["p-0", "a-0", "b-0"]


def test_union_keeps_the_edges_every_result_showed_for_an_entity():
    shared = _edge("Acme ordered 40 units", "edge-order")
    pass_one = _leg(entities=[{"id": "acme", "name": "Acme", "edges": [shared]}])
    leg_a = _leg(
        entities=[
            {"id": "other", "name": "Other", "edges": []},
            {"id": "acme", "name": "Acme", "edges": [_edge("Acme ships in April"), shared]},
        ]
    )

    merged = merge_hybrid_results_union([pass_one, leg_a])

    assert [entity["id"] for entity in merged["entities"]] == ["acme", "other"]
    assert [edge["text"] for edge in merged["entities"][0]["edges"]] == [
        "Acme ordered 40 units",
        "Acme ships in April",
    ]
    # The input results are not modified.
    assert pass_one["entities"][0]["edges"] == [shared]


def test_union_drops_a_fact_another_result_shows_as_an_entity_edge():
    pass_one = _leg(
        entities=[{"id": "acme", "name": "Acme", "edges": [_edge("Acme ordered", "edge-1")]}]
    )
    leg_a = _leg(facts=[{"id": "edge-1", "text": "Acme ordered"}, {"id": "f2", "text": "other"}])

    merged = merge_hybrid_results_union([pass_one, leg_a])

    assert [fact["id"] for fact in merged["facts"]] == ["f2"]


def test_two_way_merge_keeps_a_fact_an_entity_also_shows_as_an_edge():
    """The session merge behind plain HYBRID_COMPLETION is unchanged: only the union drops
    a fact that is already an entity edge."""
    primary = _leg(
        entities=[{"id": "acme", "name": "Acme", "edges": [_edge("Acme ordered", "edge-1")]}]
    )
    secondary = _leg(facts=[{"id": "edge-1", "text": "Acme ordered"}, {"id": "f2", "text": "x"}])

    merged = merge_hybrid_results(
        primary, secondary, chunks_limit=5, entities_limit=5, facts_limit=5
    )

    assert [fact["id"] for fact in merged["facts"]] == ["edge-1", "f2"]


def test_union_rebuilds_summaries_for_its_chunks_only():
    pass_one = _leg(chunks=[result("c1", "t")], summaries={"c1": "from pass one"})
    leg_a = _leg(chunks=[result("c2", "t")], summaries={"c1": "from leg a", "c2": "two"})

    merged = merge_hybrid_results_union([pass_one, leg_a, _leg(summaries={"gone": "x"})])

    # The earliest result with a summary for a chunk wins; summaries of absent chunks vanish.
    assert merged["chunk_summaries"] == {"c1": "from pass one", "c2": "two"}


def test_union_takes_unowned_keys_from_the_first_result_and_skips_empty_results():
    pass_one = _leg(chunks=[result("c1", "t")], global_context="built once")
    leg_a = _leg(chunks=[result("c2", "t")], global_context="ignored")

    merged = merge_hybrid_results_union([pass_one, None, {}, leg_a])

    assert merged["global_context"] == "built once"
    assert [item_id(chunk) for chunk in merged["chunks"]] == ["c1", "c2"]


def test_union_of_nothing_is_the_empty_shape():
    from cognee.modules.retrieval.hybrid.results import empty_hybrid_result

    assert merge_hybrid_results_union([]) == empty_hybrid_result()
    assert merge_hybrid_results_union([None, {}]) == empty_hybrid_result()


def test_union_never_merges_away_items_without_an_identity():
    anonymous = MagicMock()
    anonymous.id = None
    anonymous.payload = {"text": "no id"}

    merged = merge_hybrid_results_union([_leg(chunks=[anonymous]), _leg(chunks=[anonymous])])

    assert len(merged["chunks"]) == 2
