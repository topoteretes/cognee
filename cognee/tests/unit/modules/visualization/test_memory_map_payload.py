"""Unit tests for the Memory-tab payload builder (``_build_memory_map``).

Pure tests over ``preprocess()`` with synthetic graphs — no DB, no LLM.
They pin the STEP 1 contract the STEP 2 JS renderer builds on:

  - ``t_created`` is preserved from ``created_at`` before the pop.
  - Chunk cells order by ``chunk_index``; legacy chunks (attributed only via
    the ``is_part_of`` edge) append after; unattributable chunks are orphans.
  - Entity groups come from ``is_a`` → EntityType edges; the top-8 of each
    group are flagged ``important``.
  - Summaries carry ``chunk_ids`` from ``made_from`` edges.
  - ``context`` is ``None`` when no GlobalContextSummary nodes exist.
  - The timeline gap-clusters ``t_created`` into run events.
  - The whole payload is deterministic: same input twice → identical JSON,
    and node insertion order does not change any ordering.
"""

import json

import pytest

from cognee.modules.visualization.preprocessor import (
    MEMORY_GROUP_TOP_MEMBERS,
    MEMORY_TIMELINE_GAP_MS,
    preprocess,
)

T0 = 1_768_164_683_000  # first run batch (epoch ms)
T1 = T0 + 2 * MEMORY_TIMELINE_GAP_MS  # second run, 10 minutes later


def _memory_graph():
    """Two documents across two run batches, a legacy chunk attributed via
    is_part_of, an orphan chunk, one entity group, one ungrouped entity and
    one summary."""
    nodes_data = [
        ("doc1", {"type": "TextDocument", "name": "alice.md", "created_at": T0}),
        ("doc2", {"type": "TextDocument", "name": "bob.md", "created_at": T1}),
        (
            "c1",
            {
                "type": "DocumentChunk",
                "text": "chunk one",
                "chunk_index": 1,
                "document_id": "doc1",
                "created_at": T0 + 100,
                "source_pipeline": "cognify_pipeline",
            },
        ),
        (
            "c0",
            {
                "type": "DocumentChunk",
                "text": "chunk zero",
                "chunk_index": 0,
                "document_id": "doc1",
                "created_at": T0 + 200,
                "source_pipeline": "cognify_pipeline",
            },
        ),
        # Legacy chunk: no document_id, no created_at — attributed via edge.
        ("c_legacy", {"type": "DocumentChunk", "text": "legacy chunk"}),
        # Orphan chunk: no document_id and no is_part_of edge.
        ("c_orphan", {"type": "DocumentChunk", "text": "orphan chunk", "created_at": T1 + 50}),
        ("alice", {"type": "Entity", "name": "Alice", "created_at": T0 + 300}),
        ("bob", {"type": "Entity", "name": "Bob", "created_at": T0 + 300}),
        # Ungrouped entity: no is_a edge to an EntityType.
        ("zed", {"type": "Entity", "name": "Zed", "created_at": T1 + 100}),
        ("person", {"type": "EntityType", "name": "Person", "created_at": T0 + 250}),
        ("sum1", {"type": "TextSummary", "text": "summary one", "created_at": T0 + 400}),
    ]
    edges_data = [
        ("c0", "doc1", "is_part_of", {}),
        ("c_legacy", "doc1", "is_part_of", {}),
        ("c1", "alice", "contains", {}),
        ("c0", "bob", "contains", {}),
        ("alice", "person", "is_a", {}),
        ("bob", "person", "is_a", {}),
        ("alice", "bob", "knows", {"relationship_name": "knows"}),
        ("sum1", "c0", "made_from", {}),
    ]
    return (nodes_data, edges_data)


def _payload(graph=None):
    return preprocess(graph or _memory_graph()).memory_map


# ── t_created preservation ───────────────────────────────────────────────────


def test_t_created_preserved_and_created_at_dropped():
    result = preprocess(_memory_graph())
    by_id = {n["id"]: n for n in result.nodes}
    assert by_id["doc1"]["t_created"] == T0
    assert by_id["c1"]["t_created"] == T0 + 100
    for node in result.nodes:
        assert "created_at" not in node
        assert "updated_at" not in node
    # Legacy node without created_at: no t_created key fabricated.
    assert "t_created" not in by_id["c_legacy"]


def test_non_integer_created_at_is_not_preserved():
    nodes_data = [("d", {"type": "TextDocument", "name": "a", "created_at": "2026-06-10"})]
    result = preprocess((nodes_data, []))
    assert "t_created" not in result.nodes[0]
    assert "created_at" not in result.nodes[0]


# ── Documents and chunk cells ────────────────────────────────────────────────


def test_documents_sorted_by_t_first_then_name():
    payload = _payload()
    assert [d["id"] for d in payload["documents"]] == ["doc1", "doc2"]
    assert payload["documents"][0]["t_first"] == T0
    assert payload["documents"][1]["t_first"] == T1


def test_chunks_ordered_by_chunk_index_with_legacy_appended():
    payload = _payload()
    doc1 = payload["documents"][0]
    # c0 (index 0) before c1 (index 1) despite later t_created and later
    # input position; the legacy chunk (no chunk_index) appends last.
    assert [c["id"] for c in doc1["chunks"]] == ["c0", "c1", "c_legacy"]
    assert [c["chunk_index"] for c in doc1["chunks"]] == [0, 1, None]
    assert doc1["chunks"][0]["t_created"] == T0 + 200
    assert doc1["chunks"][2]["t_created"] is None


def test_orphan_chunks_collected():
    payload = _payload()
    assert payload["orphan_chunks"] == ["c_orphan"]


# ── Entity groups ────────────────────────────────────────────────────────────


def test_entity_grouping_via_is_a_edges():
    payload = _payload()
    assert len(payload["entity_groups"]) == 1
    group = payload["entity_groups"][0]
    assert group["type_id"] == "person"
    assert group["type_name"] == "Person"
    assert [m["id"] for m in group["members"]] == ["alice", "bob"]
    # Small group: everyone is within the top-8 budget.
    assert all(m["important"] for m in group["members"])


def test_ungrouped_entities_listed():
    payload = _payload()
    assert payload["ungrouped_entities"] == ["zed"]


def _big_group_graph():
    """12 equal-importance members in one group, plus high-degree ballast so
    the 75th-percentile label threshold sits above the members — making the
    top-8 ``important`` cut observable."""
    nodes = [("t", {"type": "EntityType", "name": "Person"})]
    edges = []
    for i in range(12):
        nodes.append((f"e{i}", {"type": "Entity", "name": f"E{i:02d}"}))
        edges.append((f"e{i}", "t", "is_a", {}))
    for i in range(40):
        nodes.append((f"h{i}", {"type": "Entity", "name": f"H{i:02d}"}))
    for i in range(40):
        for j in range(3):
            edges.append((f"h{i}", f"h{(i + j + 1) % 40}", "rel", {"relationship_name": "rel"}))
    return (nodes, edges)


def test_group_top_members_flagged_important():
    payload = _payload(_big_group_graph())
    groups = {g["type_name"]: g for g in payload["entity_groups"]}
    members = groups["Person"]["members"]
    assert len(members) == 12
    # Equal importance, no label_priority → deterministic name order,
    # first MEMORY_GROUP_TOP_MEMBERS important, the tail collapsible.
    assert [m["id"] for m in members] == [f"e{i}" for i in range(12)]
    assert [m["important"] for m in members] == [True] * MEMORY_GROUP_TOP_MEMBERS + [False] * 4


def test_entity_to_entity_is_a_does_not_group():
    """is_a edges between two Entity nodes must not create a group."""
    nodes = [
        ("a", {"type": "Entity", "name": "A"}),
        ("b", {"type": "Entity", "name": "B"}),
    ]
    edges = [("a", "b", "is_a", {})]
    payload = _payload((nodes, edges))
    assert payload["entity_groups"] == []
    assert payload["ungrouped_entities"] == ["a", "b"]


# ── Summaries ────────────────────────────────────────────────────────────────


def test_summaries_carry_chunk_ids_from_made_from():
    payload = _payload()
    assert payload["summaries"] == [{"id": "sum1", "chunk_ids": ["c0"], "bucket_id": None}]


def test_summaries_sorted_by_t_created():
    nodes = [
        ("s_late", {"type": "TextSummary", "text": "later", "created_at": T0 + 500}),
        ("s_early", {"type": "TextSummary", "text": "earlier", "created_at": T0 + 100}),
    ]
    payload = _payload((nodes, []))
    assert [s["id"] for s in payload["summaries"]] == ["s_early", "s_late"]


# ── Global context ───────────────────────────────────────────────────────────


def test_context_is_none_without_global_context_nodes():
    assert _payload()["context"] is None


def test_context_built_from_summarized_in_edges():
    nodes = [
        ("sum1", {"type": "TextSummary", "text": "s", "created_at": T0}),
        ("b1", {"type": "GlobalContextSummary", "text": "bucket", "level": 0}),
        ("root", {"type": "GlobalContextSummary", "text": "root", "level": 1, "is_root": True}),
    ]
    edges = [
        ("sum1", "b1", "summarized_in", {}),
        ("b1", "root", "summarized_in", {}),
    ]
    context = _payload((nodes, edges))["context"]
    assert context["root_id"] == "root"
    buckets = {b["id"]: b for b in context["buckets"]}
    assert buckets["b1"]["level"] == 0
    assert buckets["b1"]["child_ids"] == ["sum1"]
    assert buckets["root"]["child_ids"] == ["b1"]
    # Sorted by level ascending.
    assert [b["id"] for b in context["buckets"]] == ["b1", "root"]


# ── Structural edge index ────────────────────────────────────────────────────


def test_edge_index_points_into_links_array():
    result = preprocess(_memory_graph())
    edges = result.memory_map["edges"]
    assert edges["is_part_of"] == [0, 1]
    assert edges["contains"] == [2, 3]
    assert edges["semantic"] == [6]
    assert edges["made_from"] == [7]
    assert edges["summarized_in"] == []
    # Positions must resolve to the right links.
    knows = result.links[edges["semantic"][0]]
    assert (knows["source"], knows["target"], knows["relation"]) == ("alice", "bob", "knows")


def test_doc_to_chunk_contains_counts_as_membership():
    """Graphs where the doc→chunk edge is ``contains`` (not is_part_of)
    still attribute chunks to their document."""
    nodes = [
        ("d", {"type": "TextDocument", "name": "d.md"}),
        ("c", {"type": "DocumentChunk", "text": "x"}),
    ]
    edges = [("d", "c", "contains", {})]
    payload = _payload((nodes, edges))
    assert payload["documents"][0]["chunks"] == [
        {"id": "c", "chunk_index": None, "t_created": None}
    ]
    assert payload["orphan_chunks"] == []
    assert payload["edges"]["is_part_of"] == [0]
    assert payload["edges"]["contains"] == []


# ── Timeline ─────────────────────────────────────────────────────────────────


def test_timeline_two_batches_yield_two_events():
    timeline = _payload()["timeline"]
    assert len(timeline) == 2
    first, second = timeline
    assert first["index"] == 0 and second["index"] == 1
    assert first["kind"] == "run" and second["kind"] == "run"
    assert (first["t0"], first["t1"]) == (T0, T0 + 400)
    assert (second["t0"], second["t1"]) == (T1, T1 + 100)
    # Batch 1: doc1, c1, c0, alice, bob, person, sum1 + untimed c_legacy.
    assert first["node_count"] == 8
    assert "c_legacy" in first["node_ids"]  # untimed nodes join event 0
    assert set(second["node_ids"]) == {"doc2", "c_orphan", "zed"}
    assert second["node_count"] == 3


def test_timeline_labels_majority_pipeline_with_ingestion_fallback():
    timeline = _payload()["timeline"]
    assert timeline[0]["label"] == "cognify_pipeline"  # majority non-null pipeline
    assert timeline[1]["label"] == "ingestion"  # all-null fallback


def test_timeline_label_refines_to_global_context_index():
    nodes = [
        (
            "c",
            {
                "type": "DocumentChunk",
                "text": "x",
                "created_at": T0,
                "source_pipeline": "cognify_pipeline",
            },
        ),
        ("g", {"type": "GlobalContextSummary", "text": "ctx", "created_at": T0 + 10}),
    ]
    timeline = _payload((nodes, []))["timeline"]
    assert len(timeline) == 1
    assert timeline[0]["label"] == "global_context_index"


def test_timeline_same_batch_merges_into_one_event():
    nodes = [
        ("a", {"type": "Entity", "name": "A", "created_at": T0}),
        ("b", {"type": "Entity", "name": "B", "created_at": T0 + MEMORY_TIMELINE_GAP_MS}),
        ("c", {"type": "Entity", "name": "C", "created_at": T0 + 2 * MEMORY_TIMELINE_GAP_MS}),
    ]
    # Consecutive gaps equal the threshold — chained into a single cluster.
    timeline = _payload((nodes, []))["timeline"]
    assert len(timeline) == 1
    assert timeline[0]["node_count"] == 3


def test_timeline_without_any_timestamps_emits_one_synthetic_event():
    nodes = [("a", {"type": "Entity", "name": "A"}), ("b", {"type": "Entity", "name": "B"})]
    timeline = _payload((nodes, []))["timeline"]
    assert len(timeline) == 1
    assert timeline[0]["label"] == "ingestion"
    assert sorted(timeline[0]["node_ids"]) == ["a", "b"]


# ── Empty graph and determinism ──────────────────────────────────────────────


def test_empty_graph_payload_has_all_keys_empty():
    payload = _payload(([], []))
    assert payload == {
        "documents": [],
        "orphan_chunks": [],
        "entity_groups": [],
        "ungrouped_entities": [],
        "summaries": [],
        "context": None,
        "edges": {
            "contains": [],
            "made_from": [],
            "is_part_of": [],
            "summarized_in": [],
            "semantic": [],
        },
        "timeline": [],
    }


def test_payload_is_deterministic_across_runs():
    a = json.dumps(_payload(), sort_keys=True)
    b = json.dumps(_payload(), sort_keys=True)
    assert a == b


def test_payload_ordering_independent_of_node_input_order():
    """Reversing node insertion order must not change any ordering — every
    sort key is intrinsic to the data. (Edge order is kept fixed because the
    ``edges`` index stores positions into the links array.)"""
    nodes_data, edges_data = _memory_graph()
    baseline = _payload((nodes_data, edges_data))
    reversed_nodes = _payload((list(reversed(nodes_data)), edges_data))
    assert json.dumps(baseline, sort_keys=True) == json.dumps(reversed_nodes, sort_keys=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
