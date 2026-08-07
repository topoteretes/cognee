"""Merging two hybrid retrievals keeps the result shape, its budgets, and its statuses."""

from unittest.mock import MagicMock

from cognee.modules.retrieval.hybrid.merge import merge_channel_status, merge_hybrid_results


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


def test_merge_preserves_caps_summaries_statuses_and_primary_global_state():
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
