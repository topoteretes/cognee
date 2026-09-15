"""Merging two hybrid retrievals keeps the result shape and its per-channel budgets."""

from unittest.mock import MagicMock

import pytest

from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results

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


def test_union_interleaves_legs_round_robin_and_dedupes_by_content_id():
    from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results_union

    pass_one = _leg(chunks=[result("c1", "pass one")])
    leg_a = _leg(chunks=[result("c2", "a"), result("c3", "from a")])
    leg_b = _leg(chunks=[result("c3", "from b"), result("c4", "b")])

    merged = merge_hybrid_results_union(
        [pass_one, leg_a, leg_b], chunks_limit=None, entities_limit=None, facts_limit=None
    )

    assert [item_id(chunk) for chunk in merged["chunks"]] == ["c1", "c2", "c3", "c4"]
    # c3 sits at rank 1 in leg a and rank 0 in leg b: it is placed by its best rank and
    # represented by that leg's object.
    assert merged["chunks"][2].payload["text"] == "from b"


def test_union_caps_each_channel_and_keeps_every_leg_represented():
    from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results_union

    legs = [
        _leg(
            chunks=[result(f"{leg}-{rank}", "t") for rank in range(3)],
            entities=[{"id": f"e-{leg}-{rank}"} for rank in range(3)],
            facts=[{"id": f"f-{leg}-{rank}", "text": "fact"} for rank in range(3)],
        )
        for leg in ("p", "a", "b")
    ]

    merged = merge_hybrid_results_union(legs, chunks_limit=3, entities_limit=2, facts_limit=0)

    # Rank 0 of every leg before rank 1 of any leg.
    assert [item_id(chunk) for chunk in merged["chunks"]] == ["p-0", "a-0", "b-0"]
    assert [entity["id"] for entity in merged["entities"]] == ["e-p-0", "e-a-0"]
    assert merged["facts"] == []


def test_union_rebuilds_summaries_for_surviving_chunks_only():
    from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results_union

    pass_one = _leg(chunks=[result("c1", "t")], summaries={"c1": "from pass one", "gone": "x"})
    leg_a = _leg(
        chunks=[result("c1", "t"), result("c2", "t")],
        summaries={"c1": "from leg a", "c2": "summary two"},
    )

    merged = merge_hybrid_results_union(
        [pass_one, leg_a], chunks_limit=1, entities_limit=None, facts_limit=None
    )

    assert [item_id(chunk) for chunk in merged["chunks"]] == ["c1"]
    # The earliest leg with a summary for a surviving chunk wins; dropped chunks vanish.
    assert merged["chunk_summaries"] == {"c1": "from pass one"}


def test_union_takes_unowned_keys_from_the_first_result_and_skips_empty_legs():
    from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results_union

    pass_one = _leg(chunks=[result("c1", "t")], global_context="built once")
    leg_a = _leg(chunks=[result("c2", "t")], global_context="ignored")

    merged = merge_hybrid_results_union(
        [pass_one, None, {}, leg_a], chunks_limit=None, entities_limit=None, facts_limit=None
    )

    assert merged["global_context"] == "built once"
    assert [item_id(chunk) for chunk in merged["chunks"]] == ["c1", "c2"]


def test_union_of_nothing_is_the_empty_shape():
    from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results_union
    from cognee.modules.retrieval.hybrid.results import empty_hybrid_result

    assert (
        merge_hybrid_results_union([], chunks_limit=5, entities_limit=5, facts_limit=5)
        == empty_hybrid_result()
    )
    assert (
        merge_hybrid_results_union([None, {}], chunks_limit=5, entities_limit=5, facts_limit=5)
        == empty_hybrid_result()
    )


def test_union_never_merges_away_items_without_an_identity():
    from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results_union

    anonymous = MagicMock()
    anonymous.id = None
    anonymous.payload = {"text": "no id"}
    legs = [_leg(chunks=[anonymous]), _leg(chunks=[anonymous])]

    merged = merge_hybrid_results_union(
        legs, chunks_limit=None, entities_limit=None, facts_limit=None
    )

    assert len(merged["chunks"]) == 2
