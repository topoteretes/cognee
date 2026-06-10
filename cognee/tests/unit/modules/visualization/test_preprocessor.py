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
    OTHER_ENTITY_TYPES_LABEL,
    SCHEMA_MAX_ENTITY_TYPES,
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
    # Type-graph fallback emits one GraphNodeType per distinct semantic type.
    # Entity instances resolve to their EntityType via is_a, so "type:Entity"
    # is replaced by the resolved "type:Person" / "type:Field".
    type_node_ids = {
        n["id"] for n in result.schema_graph["nodes"] if n.get("type") == "GraphNodeType"
    }
    assert "type:TextDocument" in type_node_ids
    assert "type:DocumentChunk" in type_node_ids
    assert "type:Entity" not in type_node_ids
    assert "type:Person" in type_node_ids


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


def test_schema_type_nodes_resolve_semantic_types_via_is_a():
    """Entity instances collapse into their EntityType semantic types
    (Person/Field) via the is_a edge, so the literal "Entity" never appears."""
    result = preprocess(_alice_like_graph())
    type_nodes = {
        n["name"]: n for n in result.schema_graph["nodes"] if n["type"] == "GraphNodeType"
    }

    assert "Entity" not in type_nodes
    assert "Person" in type_nodes
    assert "Field" in type_nodes
    assert type_nodes["Person"]["instance_count"] == 2
    assert type_nodes["Field"]["instance_count"] == 1


def test_schema_type_nodes_carry_bounded_deterministic_samples():
    result = preprocess(_alice_like_graph())
    type_nodes = {
        n["name"]: n for n in result.schema_graph["nodes"] if n["type"] == "GraphNodeType"
    }

    person = type_nodes["Person"]
    # Alice has degree 3 (contains, is_a, knows), Bob has degree 3
    # (contains, is_a, knows). Tie on degree breaks to name order: Alice, Bob.
    assert person["samples"] == ["Alice", "Bob"]
    assert person["sample_size"] == 2
    assert type_nodes["Field"]["samples"] == ["NLP"]

    # Samples never exceed the per-type cap regardless of instance count.
    for node in type_nodes.values():
        assert node["sample_size"] <= 5
        assert len(node["samples"]) == node["sample_size"]


def test_schema_type_nodes_carry_full_relationship_distribution():
    result = preprocess(_alice_like_graph())
    type_nodes = {
        n["name"]: n for n in result.schema_graph["nodes"] if n["type"] == "GraphNodeType"
    }

    person_rels = {
        (r["relation"], r["to_type"]): r["count"] for r in type_nodes["Person"]["relationships"]
    }
    # Alice + Bob both is_a the EntityType node; alice knows bob (Person -> Person).
    assert person_rels[("is_a", "EntityType")] == 2
    assert person_rels[("knows", "Person")] == 1

    # DocumentChunk contains Person twice (alice, bob), Field once (nlp).
    chunk_rels = {
        (r["relation"], r["to_type"]): r["count"]
        for r in type_nodes["DocumentChunk"]["relationships"]
    }
    assert chunk_rels[("contains", "Person")] == 2
    assert chunk_rels[("contains", "Field")] == 1


def _many_entity_types_graph(num_types):
    """num_types semantic entity types with strictly descending instance counts:
    Type00 has num_types instances, Type01 has num_types-1, ... down to 1."""
    nodes_data = []
    edges_data = []
    for i in range(num_types):
        type_name = f"Type{i:02d}"
        type_id = f"etype{i}"
        nodes_data.append((type_id, {"type": "EntityType", "name": type_name}))
        for j in range(num_types - i):
            entity_id = f"e{i}_{j}"
            nodes_data.append((entity_id, {"type": "Entity", "name": f"{type_name}_inst{j}"}))
            edges_data.append((entity_id, type_id, "is_a", {}))
    return (nodes_data, edges_data)


def test_entity_type_long_tail_rolls_up_into_other_entities():
    """Beyond SCHEMA_MAX_ENTITY_TYPES, the tail of semantic entity types
    collapses into one rollup card so the Entity column stays bounded."""
    num_types = SCHEMA_MAX_ENTITY_TYPES + 3
    result = preprocess(_many_entity_types_graph(num_types))
    type_nodes = {
        n["name"]: n for n in result.schema_graph["nodes"] if n["type"] == "GraphNodeType"
    }

    semantic_cards = [name for name in type_nodes if name.startswith("Type")] + (
        [OTHER_ENTITY_TYPES_LABEL] if OTHER_ENTITY_TYPES_LABEL in type_nodes else []
    )
    assert len(semantic_cards) == SCHEMA_MAX_ENTITY_TYPES

    # Top types keep their own cards; the smallest types are rolled up.
    assert "Type00" in type_nodes
    assert f"Type{num_types - 1:02d}" not in type_nodes

    rollup = type_nodes[OTHER_ENTITY_TYPES_LABEL]
    assert rollup["rollup"] is True
    tail_size = num_types - (SCHEMA_MAX_ENTITY_TYPES - 1)
    assert len(rollup["rolled_up_types"]) == tail_size
    # Tail of descending counts ends at 1: tail_size + (tail_size-1) + ... + 1.
    assert rollup["instance_count"] == tail_size * (tail_size + 1) // 2
    # Rollup keeps the same rank as the kept entity-type cards (one column).
    assert rollup["rank"] == type_nodes["Type00"]["rank"]
    # The lead field announces the rollup.
    assert any(f["name"] == "entity types" for f in rollup["fields"])

    # Pair-relationship nodes never reference a rolled-up type name.
    rolled_names = {t["name"] for t in rollup["rolled_up_types"]}
    for node in result.schema_graph["nodes"]:
        if node["type"] == "GraphRelationshipType":
            assert node["source_type"] not in rolled_names
            assert node["target_type"] not in rolled_names

    # Instance drill-down still reaches the rolled-up instances.
    instances = result.schema_graph["instances_by_type"][OTHER_ENTITY_TYPES_LABEL]
    assert len(instances) == rollup["instance_count"]


def test_entity_types_under_cap_are_not_rolled_up():
    result = preprocess(_alice_like_graph())
    names = {n["name"] for n in result.schema_graph["nodes"] if n["type"] == "GraphNodeType"}
    assert OTHER_ENTITY_TYPES_LABEL not in names

    result_at_cap = preprocess(_many_entity_types_graph(SCHEMA_MAX_ENTITY_TYPES))
    names_at_cap = {
        n["name"] for n in result_at_cap.schema_graph["nodes"] if n["type"] == "GraphNodeType"
    }
    assert OTHER_ENTITY_TYPES_LABEL not in names_at_cap
    assert sum(1 for name in names_at_cap if name.startswith("Type")) == SCHEMA_MAX_ENTITY_TYPES


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


def test_operation_layer_maps_operations_to_present_types():
    """The transformation impact-layer emits operation nodes + typed links only
    for catalog operations that touch a type present in the graph, expanding
    'Entity' to the semantic entity types."""
    nodes = [
        ("d1", {"type": "TextDocument", "name": "a.txt"}),
        ("p1", {"type": "Entity", "name": "Carlos"}),
        ("t_person", {"type": "EntityType", "name": "Person"}),
    ]
    edges = [("p1", "t_person", "is_a", {})]
    schema = preprocess((nodes, edges)).schema_graph

    op_ids = {o["id"] for o in schema["operations"]}
    assert "op:cognify" in op_ids

    cognify_targets = {
        (link["target"], link["effect"])
        for link in schema["operation_links"]
        if link["source"] == "op:cognify"
    }
    assert ("type:TextDocument", "produces") in cognify_targets
    assert ("type:Person", "produces") in cognify_targets  # Entity → semantic type
    assert ("type:EntityType", "produces") in cognify_targets

    # An operation whose only targets are absent (Rule) is filtered out entirely.
    assert "op:coding_rule_associations" not in op_ids

    # Modify effects are surfaced (feedback weighting touches entity types).
    feedback_targets = {
        (link["target"], link["effect"])
        for link in schema["operation_links"]
        if link["source"] == "op:apply_feedback_weights"
    }
    assert ("type:Person", "modifies") in feedback_targets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
