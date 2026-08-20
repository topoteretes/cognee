"""Merging two hybrid retrievals keeps the result shape, its budgets, and its statuses."""

from unittest.mock import MagicMock

import pytest

from cognee.modules.retrieval.hybrid.merge import merge_channel_status, merge_hybrid_results

EMPTY_CHANNELS = {"chunks": [], "entities": [], "facts": []}


def result(identifier, text):
    item = MagicMock()
    item.id = identifier
    item.payload = {"id": identifier, "text": text}
    return item


def edge(edge_id):
    item = MagicMock()
    item.attributes = {"edge_object_id": edge_id, "relationship_name": "knows"}
    item.node1.id = "source"
    item.node2.id = "target"
    item.directed = True
    return item


def status(value, count=0):
    return {"status": value, "item_count": count}


def item_id(item):
    return item["id"] if isinstance(item, dict) else item.id


def test_merge_preserves_caps_summaries_statuses_and_primary_global_state():
    """Fixtures match HybridRetriever's real output: chunks, summaries, entities, facts."""
    primary_chunk = result("chunk-1", "primary")
    primary = {
        "chunks": [primary_chunk],
        "chunk_summaries": {"chunk-1": "primary summary", "dropped": "drop"},
        "chunk_attribution": [{"chunk_id": "chunk-1", "channels": ["primary"]}],
        "entities": [{"id": "entity-1", "name": "Primary"}],
        "facts": [],
        "graph_fallback": [edge("edge-1")],
        "retrieval_status": {
            "chunks": status("ok", 1),
            "entities": status("degraded"),
            "facts": status("skipped"),
            "graph_fallback": status("ok", 1),
            "global_context": status("pending"),
        },
        "global_context": "built once from the raw query",
    }
    secondary = {
        "chunks": [result("chunk-1", "secondary"), result("chunk-2", "second")],
        "chunk_summaries": {"chunk-1": "secondary summary", "chunk-2": "second"},
        "entities": [{"id": "entity-1", "name": "Secondary"}],
        "facts": [{"id": "fact-1", "text": "fact"}],
        "graph_fallback": [edge("edge-2")],
        "retrieval_status": {
            "chunks": status("ok", 2),
            "entities": status("ok", 1),
            "facts": status("degraded"),
            "graph_fallback": status("degraded"),
            "global_context": status("degraded"),
        },
        "global_context": "ignored",
    }

    merged = merge_hybrid_results(
        primary,
        secondary,
        chunks_limit=1,
        entities_limit=1,
        facts_limit=1,
        graph_limit=1,
    )

    assert merged["chunks"] == [primary_chunk]
    # Summaries and attribution follow the chunks that survived the cap.
    assert merged["chunk_summaries"] == {"chunk-1": "primary summary"}
    assert merged["chunk_attribution"] == primary["chunk_attribution"]
    assert merged["entities"] == [primary["entities"][0]]
    assert merged["facts"] == secondary["facts"]
    assert merged["graph_fallback"] == primary["graph_fallback"]
    assert merged["retrieval_status"]["entities"] == status("ok", 1)
    assert merged["retrieval_status"]["facts"] == status("degraded")
    # global_context is built once from the raw query, so its status is not merged.
    assert merged["retrieval_status"]["global_context"] == status("pending")
    assert merged["global_context"] == "built once from the raw query"


def test_missing_graph_channel_stays_absent():
    merged = merge_hybrid_results(
        {"chunks": [], "entities": [], "facts": []},
        {"chunks": [], "entities": [], "facts": []},
        chunks_limit=1,
        entities_limit=1,
        facts_limit=1,
        graph_limit=1,
    )

    assert "graph_fallback" not in merged


def test_one_successful_lane_makes_the_channel_successful():
    assert merge_channel_status(status("degraded"), status("ok"), item_count=2) == status("ok", 2)
    assert merge_channel_status(None, status("degraded"), item_count=0) == status("degraded")
    assert merge_channel_status(None, None, item_count=0) == {"status": "skipped"}


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
