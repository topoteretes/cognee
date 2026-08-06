from unittest.mock import MagicMock

from cognee.modules.retrieval.session_result_fusion import (
    fuse_graph_results,
    fuse_hybrid_results,
    fuse_vector_results,
)


def result(result_id, text, *, truthy=True):
    item = MagicMock()
    item.id = result_id
    item.payload = {"id": result_id, "text": text}
    if not truthy:
        item.__bool__.return_value = False
    return item


def edge(edge_id, *, relationship="knows"):
    item = MagicMock()
    item.attributes = {"edge_object_id": edge_id, "relationship_name": relationship}
    item.node1.id = "source"
    item.node2.id = "target"
    item.directed = True
    return item


def status(value, count=0):
    return {"status": value, "item_count": count}


def test_vector_fusion_uses_weighted_rrf_and_keeps_raw_representative():
    raw_a = result("a", "raw a")
    raw_b = result("b", "raw b", truthy=False)
    contextual_b = result("b", "contextual b")
    contextual_c = result("c", "contextual c")

    fused = fuse_vector_results(
        [raw_a, raw_b],
        [contextual_b, contextual_c],
        limit=3,
    )

    assert fused == [raw_b, raw_a, contextual_c]


def test_one_lane_preserves_order_and_cap():
    items = [result("a", "a"), result("b", "b"), result("c", "c")]

    assert fuse_vector_results(items, None, limit=2) == items[:2]


def test_graph_fusion_deduplicates_by_edge_object_id():
    raw = edge("edge-1")
    contextual = edge("edge-1")

    assert fuse_graph_results([raw], [contextual], limit=5) == [raw]


def test_graph_fusion_falls_back_to_relationship_identity():
    raw = edge(None)
    contextual = edge(None)

    assert fuse_graph_results([raw], [contextual], limit=5) == [raw]


def test_hybrid_fusion_preserves_caps_summaries_statuses_and_raw_global_state():
    raw_chunk = result("chunk-1", "raw")
    contextual_chunk = result("chunk-1", "contextual")
    contextual_chunk_2 = result("chunk-2", "contextual 2")
    raw = {
        "chunks": [raw_chunk],
        "chunk_summaries": {"chunk-1": "raw summary", "dropped": "drop"},
        "chunk_attribution": [{"chunk_id": "chunk-1", "channels": ["raw"]}],
        "entities": [{"id": "entity-1", "name": "Raw"}],
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
    contextual = {
        "chunks": [contextual_chunk, contextual_chunk_2],
        "chunk_summaries": {"chunk-1": "context summary", "chunk-2": "second"},
        "entities": [{"id": "entity-1", "name": "Context"}],
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

    fused = fuse_hybrid_results(
        raw,
        contextual,
        chunks_limit=1,
        entities_limit=1,
        facts_limit=1,
        graph_limit=1,
    )

    assert fused["chunks"] == [raw_chunk]
    assert fused["chunk_summaries"] == {"chunk-1": "raw summary"}
    assert fused["chunk_attribution"] == raw["chunk_attribution"]
    assert fused["entities"] == [raw["entities"][0]]
    assert fused["facts"] == contextual["facts"]
    assert fused["graph_fallback"] == raw["graph_fallback"]
    assert fused["retrieval_status"]["entities"] == status("ok", 1)
    assert fused["retrieval_status"]["facts"] == status("degraded")
    assert fused["retrieval_status"]["global_context"] == status("pending")
