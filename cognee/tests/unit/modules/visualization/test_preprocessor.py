"""Unit tests for the visualization preprocessor.

The preprocessor is the single place where Story-view fields are derived
from raw graph adapter output. These tests pin the contract:

  - Every known node type maps to a non-default stage.
  - ``visual_rank`` prefers stamped ``topological_rank`` (Phase 1a) and
    falls back to a fixed stage order when unset.
  - ``contains`` / ``is_a`` edges are classified ``structural``.
  - Edges between the same stage pair sharing a relation collapse into one
    ``bundle_key`` so the renderer can bundle them.
  - Provenance is exposed only when at least one provenance field is set.
  - Color-map / schema-graph shape matches what the existing JS renderer
    already reads.
"""

import pytest

from cognee.modules.visualization.preprocessor import (
    STAGE_ORDER,
    PreprocessedGraph,
    preprocess,
)


def _alice_like_graph():
    """A small graph that mirrors the shape of the canonical Alice example:
    one document, two chunks, three entities of two types, and one summary."""
    nodes_data = [
        ("doc1", {"type": "TextDocument", "name": "alice.md", "topological_rank": 1}),
        (
            "c1",
            {
                "type": "DocumentChunk",
                "text": "Alice knows Bob.",
                "source_pipeline": "cognify_pipeline",
                "source_task": "extract_chunks_from_documents",
                "topological_rank": 2,
            },
        ),
        (
            "c2",
            {
                "type": "DocumentChunk",
                "text": "NLP is a subfield of CS.",
                "source_pipeline": "cognify_pipeline",
                "source_task": "extract_chunks_from_documents",
                "topological_rank": 2,
            },
        ),
        (
            "alice",
            {
                "type": "Entity",
                "name": "Alice",
                "source_pipeline": "cognify_pipeline",
                "source_task": "extract_graph_from_data",
                "topological_rank": 3,
            },
        ),
        (
            "bob",
            {
                "type": "Entity",
                "name": "Bob",
                "source_pipeline": "cognify_pipeline",
                "source_task": "extract_graph_from_data",
                "topological_rank": 3,
            },
        ),
        (
            "nlp",
            {
                "type": "Entity",
                "name": "NLP",
                "source_pipeline": "cognify_pipeline",
                "source_task": "extract_graph_from_data",
                "topological_rank": 3,
            },
        ),
        ("person", {"type": "EntityType", "name": "Person", "topological_rank": 4}),
        ("field", {"type": "EntityType", "name": "Field", "topological_rank": 4}),
        (
            "sum1",
            {
                "type": "TextSummary",
                "text": "Alice and Bob in NLP.",
                "topological_rank": 5,
            },
        ),
    ]
    edges_data = [
        ("doc1", "c1", "contains", {}),
        ("doc1", "c2", "contains", {}),
        ("c1", "alice", "contains", {}),
        ("c1", "bob", "contains", {}),
        ("c2", "nlp", "contains", {}),
        ("alice", "person", "is_a", {}),
        ("bob", "person", "is_a", {}),
        ("nlp", "field", "is_a", {}),
        ("alice", "bob", "knows", {"relationship_name": "knows"}),
        ("c1", "sum1", "made_from", {}),
    ]
    return (nodes_data, edges_data)


def test_preprocess_returns_preprocessed_graph_dataclass():
    result = preprocess(_alice_like_graph())
    assert isinstance(result, PreprocessedGraph)
    assert len(result.nodes) == 9
    assert len(result.links) == 10


def test_stage_assignment_for_known_types():
    result = preprocess(_alice_like_graph())
    stages = {n["id"]: n["stage"] for n in result.nodes}
    assert stages["doc1"] == "document"
    assert stages["c1"] == "chunk"
    assert stages["c2"] == "chunk"
    assert stages["alice"] == "entity"
    assert stages["bob"] == "entity"
    assert stages["nlp"] == "entity"
    assert stages["person"] == "type"
    assert stages["field"] == "type"
    assert stages["sum1"] == "summary"


def test_stage_falls_through_to_other_for_unknown_types():
    nodes_data = [("x1", {"type": "MysteryType"})]
    edges_data = []
    result = preprocess((nodes_data, edges_data))
    assert result.nodes[0]["stage"] == "other"


def test_visual_rank_uses_stamped_topological_rank():
    """Phase 1a stamps topological_rank in the pipeline. The preprocessor
    must use that real value when it's a positive integer."""
    result = preprocess(_alice_like_graph())
    by_id = {n["id"]: n for n in result.nodes}
    assert by_id["doc1"]["visual_rank"] == 1
    assert by_id["c1"]["visual_rank"] == 2
    assert by_id["alice"]["visual_rank"] == 3
    assert by_id["person"]["visual_rank"] == 4
    assert by_id["sum1"]["visual_rank"] == 5


def test_visual_rank_falls_back_when_topological_rank_zero_or_none():
    """Legacy graphs (pre-Phase-1a) have rank=0 or rank=None on every node.
    The preprocessor must produce a usable rank from the stage."""
    nodes_data = [
        ("d", {"type": "TextDocument", "topological_rank": 0}),
        ("c", {"type": "DocumentChunk"}),  # no rank at all
        ("e", {"type": "Entity", "topological_rank": None}),
    ]
    edges_data = [("d", "c", "contains", {}), ("c", "e", "contains", {})]
    result = preprocess((nodes_data, edges_data))
    by_id = {n["id"]: n for n in result.nodes}
    # Stage-order fallback: document=1, chunk=2, entity=3
    assert by_id["d"]["visual_rank"] == STAGE_ORDER.index("document") + 1
    assert by_id["c"]["visual_rank"] == STAGE_ORDER.index("chunk") + 1
    assert by_id["e"]["visual_rank"] == STAGE_ORDER.index("entity") + 1


def test_has_meaningful_topological_rank_flag():
    """The renderer reads this flag to decide whether to use real ranks
    or fall back to its own type-based scheme."""
    real = preprocess(_alice_like_graph())
    assert real.has_meaningful_topological_rank is True

    nodes_data = [("d", {"type": "TextDocument", "topological_rank": 0})]
    edges_data = []
    legacy = preprocess((nodes_data, edges_data))
    assert legacy.has_meaningful_topological_rank is False


def test_structural_edges_classified_correctly():
    result = preprocess(_alice_like_graph())
    by_relation = {
        (link["source"], link["target"], link["relation"]): link for link in result.links
    }

    for key in [
        ("doc1", "c1", "contains"),
        ("doc1", "c2", "contains"),
        ("c1", "alice", "contains"),
        ("alice", "person", "is_a"),
        ("c1", "sum1", "made_from"),
    ]:
        assert by_relation[key]["edge_class"] == "structural", f"{key} should be structural"

    assert by_relation[("alice", "bob", "knows")]["edge_class"] == "semantic"


def test_bundle_key_collapses_structural_edges_into_groups():
    """The Alice-like graph has 5 ``contains`` edges, but they fall into two
    bundles: doc->chunk (2 edges) and chunk->entity (3 edges).

    This proves the renderer can replace 5 lines with 2 ribbons."""
    result = preprocess(_alice_like_graph())
    contains_bundles = {k: v for k, v in result.bundles.items() if "|contains" in k}
    assert len(contains_bundles) == 2
    counts = sorted(contains_bundles.values())
    assert counts == [2, 3]


def test_provenance_present_only_when_fields_set():
    result = preprocess(_alice_like_graph())
    by_id = {n["id"]: n for n in result.nodes}
    # doc1 has no provenance fields in the fixture — section must be hidden
    assert "provenance" not in by_id["doc1"]
    # c1 has source_pipeline and source_task set
    assert by_id["c1"]["provenance"] == {
        "source_pipeline": "cognify_pipeline",
        "source_task": "extract_chunks_from_documents",
    }


def test_color_maps_have_expected_keys():
    result = preprocess(_alice_like_graph())
    assert set(result.color_maps.keys()) == {"task", "pipeline", "node_set", "user"}
    # pipeline color map should contain the one pipeline that's set
    assert "cognify_pipeline" in result.color_maps["pipeline"]
    # task color map should contain both tasks
    assert "extract_chunks_from_documents" in result.color_maps["task"]
    assert "extract_graph_from_data" in result.color_maps["task"]


def test_pipeline_stages_in_canonical_order():
    result = preprocess(_alice_like_graph())
    # Story-view spine: document, chunk, entity, type, summary
    assert result.pipeline_stages == ["document", "chunk", "entity", "type", "summary"]


def test_degree_count_matches_edge_count():
    result = preprocess(_alice_like_graph())
    by_id = {n["id"]: n for n in result.nodes}
    # doc1 -> c1, doc1 -> c2  => degree 2
    assert by_id["doc1"]["degree"] == 2
    # c1: doc1->c1, c1->alice, c1->bob, c1->sum1 => degree 4
    assert by_id["c1"]["degree"] == 4


def test_label_priority_marks_documents_and_types_always():
    result = preprocess(_alice_like_graph())
    by_id = {n["id"]: n for n in result.nodes}
    # Documents and entity-types are landmarks; always labeled in Key mode
    assert by_id["doc1"]["label_priority"] is True
    assert by_id["person"]["label_priority"] is True
    assert by_id["field"]["label_priority"] is True


def test_edge_class_counts_summed():
    result = preprocess(_alice_like_graph())
    # 5 contains + 3 is_a + 1 made_from = 9 structural, 1 knows = 1 semantic
    assert result.edge_classes["structural"] == 9
    assert result.edge_classes["semantic"] == 1


def test_handles_3tuple_edges_without_edge_info():
    """Some adapters may yield 3-tuple edges (no edge_info dict)."""
    nodes_data = [("a", {"type": "Entity"}), ("b", {"type": "Entity"})]
    edges_data = [("a", "b", "knows")]  # 3-tuple
    result = preprocess((nodes_data, edges_data))
    assert len(result.links) == 1
    assert result.links[0]["edge_class"] == "semantic"


def test_node_color_preserved_from_type_map():
    """The preprocessor's TYPE_COLOR_MAP drives node colors in both the
    canvas and the legend swatches. Pin the canonical four so a color
    palette change doesn't silently break the visual encoding."""
    result = preprocess(_alice_like_graph())
    by_id = {n["id"]: n for n in result.nodes}
    assert by_id["alice"]["color"] == "#6510F4"  # Entity
    assert by_id["person"]["color"] == "#D5C2FF"  # EntityType
    assert by_id["c1"]["color"] == "#0DFF00"  # DocumentChunk
    assert (
        by_id["doc1"]["color"] == "#A550FF"
    )  # TextDocument (was default gray before Phase 1 polish)


def test_ontology_valid_overrides_color():
    nodes_data = [("e", {"type": "Entity", "name": "X", "ontology_valid": True})]
    edges_data = []
    result = preprocess((nodes_data, edges_data))
    assert result.nodes[0]["color"] == "#D8D8D8"


def test_schema_graph_falls_back_to_type_graph_when_no_schema_nodes():
    result = preprocess(_alice_like_graph())
    assert "nodes" in result.schema_graph
    assert "links" in result.schema_graph
    # Type-graph fallback emits one GraphNodeType per distinct type
    type_node_ids = {
        n["id"] for n in result.schema_graph["nodes"] if n.get("type") == "GraphNodeType"
    }
    assert "type:TextDocument" in type_node_ids
    assert "type:DocumentChunk" in type_node_ids
    assert "type:Entity" in type_node_ids


def test_schema_graph_uses_schema_nodes_when_present():
    nodes_data = [
        (
            "users",
            {
                "type": "SchemaTable",
                "name": "users",
                "columns": '[{"name": "id", "type": "uuid"}]',
                "primary_key": "id",
            },
        ),
        (
            "posts",
            {
                "type": "SchemaTable",
                "name": "posts",
                "columns": '[{"name": "id", "type": "uuid"}]',
                "primary_key": "id",
            },
        ),
    ]
    edges_data = [("posts", "users", "has_relationship", {"relationship_name": "foreign_key"})]
    result = preprocess((nodes_data, edges_data))
    schema_node_types = {n.get("type") for n in result.schema_graph["nodes"]}
    assert "SchemaTable" in schema_node_types


def test_empty_graph_does_not_crash():
    result = preprocess(([], []))
    assert result.nodes == []
    assert result.links == []
    assert result.has_meaningful_topological_rank is False
    assert result.pipeline_stages == []


def test_provenance_index_indexes_only_nodes_with_provenance():
    result = preprocess(_alice_like_graph())
    # c1 has provenance; doc1 doesn't
    assert "c1" in result.provenance_index
    assert "doc1" not in result.provenance_index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
