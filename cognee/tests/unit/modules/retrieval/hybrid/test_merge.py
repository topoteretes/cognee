"""Merging two hybrid retrievals keeps the result shape and its per-channel budgets."""

from unittest.mock import MagicMock

import pytest

from cognee.modules.retrieval.hybrid.chunks import PASSAGES_DROPPED_BY_CUTOFF
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


def test_merge_clears_cutoff_when_secondary_lane_keeps_a_passage():
    """Primary was fully cut off; conversational lane still has a chunk."""
    surviving = result("chunk-ctx", "kept by conversational rewrite")
    primary = {
        **EMPTY_CHANNELS,
        "chunk_summaries": {},
        PASSAGES_DROPPED_BY_CUTOFF: True,
        "entities": [],
        "facts": [],
    }
    secondary = {
        "chunks": [surviving],
        "chunk_summaries": {"chunk-ctx": "ctx summary"},
        "entities": [{"id": "entity-ctx", "name": "Kept"}],
        "facts": [{"id": "fact-ctx", "text": "kept fact"}],
    }

    merged = merge_hybrid_results(
        primary,
        secondary,
        chunks_limit=5,
        entities_limit=5,
        facts_limit=5,
    )

    assert merged["chunks"] == [surviving]
    assert PASSAGES_DROPPED_BY_CUTOFF not in merged
    assert merged["entities"] == secondary["entities"]


def test_merge_keeps_cutoff_when_both_lanes_were_fully_cut_off():
    primary = {**EMPTY_CHANNELS, "chunk_summaries": {}, PASSAGES_DROPPED_BY_CUTOFF: True}
    secondary = {**EMPTY_CHANNELS, "chunk_summaries": {}, PASSAGES_DROPPED_BY_CUTOFF: True}

    merged = merge_hybrid_results(
        primary,
        secondary,
        chunks_limit=5,
        entities_limit=5,
        facts_limit=5,
    )

    assert merged["chunks"] == []
    assert merged[PASSAGES_DROPPED_BY_CUTOFF] is True


def test_merge_propagates_secondary_only_cutoff_when_merged_chunks_are_empty():
    primary = {**EMPTY_CHANNELS, "chunk_summaries": {}}
    secondary = {**EMPTY_CHANNELS, "chunk_summaries": {}, PASSAGES_DROPPED_BY_CUTOFF: True}

    merged = merge_hybrid_results(
        primary,
        secondary,
        chunks_limit=5,
        entities_limit=5,
        facts_limit=5,
    )

    assert merged["chunks"] == []
    assert merged[PASSAGES_DROPPED_BY_CUTOFF] is True
