from unittest.mock import MagicMock

import pytest

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.modules.engine.models import Entity
from cognee.modules.graph.utils.expand_with_nodes_and_edges import (
    attach_new_edges_to_data_points,
    construct_data_points_and_edges,
)
from cognee.shared.data_models import KnowledgeGraph, Node, Edge as KGEdge


def _make_chunk(importance_weight=0.5):
    from unittest.mock import MagicMock as MM

    chunk = MM()
    chunk.contains = None
    chunk.belongs_to_set = []
    chunk.importance_weight = importance_weight
    chunk._provenance_edges = []
    return chunk


def _make_graph(nodes, edges):
    return KnowledgeGraph(nodes=nodes, edges=edges)


def _construct_test_data_points(data_chunks, extracted_graphs):
    data_points_by_id, edges_by_identity = construct_data_points_and_edges(
        data_chunks,
        extracted_graphs,
    )
    attach_new_edges_to_data_points(data_points_by_id, edges_by_identity, set())
    return list(data_points_by_id.values())


def test_chunk_contains_populated():
    chunk = _make_chunk()
    graph = _make_graph(
        [Node(id="n1", name="Alice", type="Person", description="A person")],
        [],
    )
    _construct_test_data_points([chunk], [graph])

    assert chunk.contains is not None
    assert len(chunk.contains) == 1
    _, entity = chunk.contains[0]
    assert entity.name == "alice"


def test_entity_relations_populated_from_graph_edges():
    chunk = _make_chunk()
    graph = _make_graph(
        [
            Node(id="n1", name="Alice", type="Person", description="desc"),
            Node(id="n2", name="Bob", type="Person", description="desc"),
        ],
        [KGEdge(source_node_id="n1", target_node_id="n2", relationship_name="knows")],
    )
    data_points = _construct_test_data_points([chunk], [graph])

    alice = next(data_point for data_point in data_points if data_point.name == "alice")
    assert len(alice.relations) == 1
    edge_obj, target = alice.relations[0]
    assert target.name == "bob"
    assert edge_obj.relationship_type == "knows"
    assert edge_obj.edge_text is None


def test_existing_graph_edge_stays_out_of_datapoint_relations():
    chunk = _make_chunk()
    graph = _make_graph(
        [
            Node(id="n1", name="Alice", type="Person", description="desc"),
            Node(id="n2", name="Bob", type="Person", description="desc"),
        ],
        [KGEdge(source_node_id="n1", target_node_id="n2", relationship_name="knows")],
    )
    existing_edge_identity = EdgeIdentity(
        source_id=str(Entity.id_for("Alice")),
        target_id=str(Entity.id_for("Bob")),
        relationship_name="knows",
    )

    data_points_by_id, edges_by_identity = construct_data_points_and_edges([chunk], [graph])
    attach_new_edges_to_data_points(
        data_points_by_id,
        edges_by_identity,
        {existing_edge_identity},
    )

    alice = next(
        data_point for data_point in data_points_by_id.values() if data_point.name == "alice"
    )
    assert alice.relations == []
    assert chunk._provenance_edges == [
        (
            str(Entity.id_for("Alice")),
            str(Entity.id_for("Bob")),
            "knows",
            {"edge_text": None},
        )
    ]


def test_edge_evidence_uses_post_ontology_node_ids():
    from types import SimpleNamespace

    from cognee.modules.ontology.construct_data_points_and_edges_with_ontology import (
        construct_data_points_and_edges_with_ontology,
    )

    resolver = MagicMock()

    def get_subgraph(node_name, node_type):
        if node_type == "individuals" and node_name == "alice":
            return [], [], SimpleNamespace(name="Alicia", uri=None)
        if node_type == "individuals" and node_name == "bob":
            return [], [], SimpleNamespace(name="Robert", uri=None)
        return [], [], None

    resolver.get_subgraph.side_effect = get_subgraph
    chunk = _make_chunk()
    graph = _make_graph(
        [
            Node(id="Alice", name="Alice", type="Person", description="desc"),
            Node(id="Bob", name="Bob", type="Person", description="desc"),
        ],
        [KGEdge(source_node_id="Alice", target_node_id="Bob", relationship_name="knows")],
    )

    construct_data_points_and_edges_with_ontology([chunk], [graph], resolver)

    assert chunk._provenance_edges[0][:3] == (
        str(Entity.id_for("Alicia")),
        str(Entity.id_for("Robert")),
        "knows",
    )


def test_chunk_contains_edge_text_uses_per_chunk_description():
    chunk1, chunk2 = _make_chunk(), _make_chunk()
    graph1 = _make_graph(
        [Node(id="n1", name="Alice", type="Person", description="Alice founded Acme.")],
        [],
    )
    graph2 = _make_graph(
        [Node(id="n1", name="Alice", type="Person", description="Alice lives in Paris.")],
        [],
    )

    _construct_test_data_points([chunk1, chunk2], [graph1, graph2])

    first_edge, first_entity = chunk1.contains[0]
    second_edge, second_entity = chunk2.contains[0]

    assert first_entity is second_entity
    assert first_edge.edge_text == "Document chunk mentions alice: Alice founded Acme."
    assert second_edge.edge_text == "Document chunk mentions alice: Alice lives in Paris."


def test_blank_chunk_description_leaves_edge_text_none_before_storage():
    chunk = _make_chunk()
    graph = _make_graph(
        [Node(id="n1", name="Alice", type="Person", description="   ")],
        [],
    )

    _construct_test_data_points([chunk], [graph])

    edge_obj, _ = chunk.contains[0]
    assert edge_obj.edge_text is None


def test_entity_relation_preserves_llm_edge_description():
    chunk = _make_chunk()
    graph = _make_graph(
        [
            Node(id="n1", name="Alice", type="Person", description="desc"),
            Node(id="n2", name="Acme", type="Company", description="desc"),
        ],
        [
            KGEdge(
                source_node_id="n1",
                target_node_id="n2",
                relationship_name="works_at",
                description="Alice works at Acme.",
            )
        ],
    )

    data_points = _construct_test_data_points([chunk], [graph])

    alice = next(data_point for data_point in data_points if data_point.name == "alice")
    edge_obj, target = alice.relations[0]
    assert target.name == "acme"
    assert edge_obj.relationship_type == "works_at"
    assert edge_obj.edge_text == "Alice works at Acme."


def test_edge_model_does_not_default_edge_text_from_relationship_type():
    assert Edge(relationship_type="contains").edge_text is None


def test_construct_returns_data_point_and_edge_maps():
    chunk = _make_chunk()
    graph = _make_graph(
        [Node(id="n1", name="Thing", type="Object", description="a thing")],
        [],
    )
    data_points_by_id, edges_by_identity = construct_data_points_and_edges(
        [chunk],
        [graph],
    )

    assert isinstance(data_points_by_id, dict)
    assert isinstance(edges_by_identity, dict)


def test_empty_graph_skipped():
    chunk = _make_chunk()
    data_points = _construct_test_data_points([chunk], [None])

    assert chunk.contains is None
    assert data_points == []


def test_entity_deduplication_across_chunks():
    chunk1, chunk2 = _make_chunk(), _make_chunk()
    node = Node(id="n1", name="Alice", type="Person", description="desc")
    graph1 = _make_graph([node], [])
    graph2 = _make_graph([node], [])

    data_points = _construct_test_data_points(
        [chunk1, chunk2],
        [graph1, graph2],
    )

    alice_nodes = [data_point for data_point in data_points if data_point.name == "alice"]
    assert len(alice_nodes) == 1


def test_unique_entity_id_comes_from_name_not_llm_id():
    chunk = _make_chunk()
    graph = _make_graph(
        [Node(id="llm-local-id", name="Alice", type="Person", description="desc")],
        [],
    )

    data_points = _construct_test_data_points([chunk], [graph])

    alice = next(data_point for data_point in data_points if isinstance(data_point, Entity))
    assert alice.id == Entity.id_for("Alice")


def test_duplicate_names_remain_distinct_within_one_graph():
    chunk = _make_chunk()
    chunk.id = "chunk-1"
    graph = _make_graph(
        [
            Node(id="alice-b", name="Alice", type="Person", description="second"),
            Node(id="alice-a", name="Alice", type="Person", description="first"),
        ],
        [KGEdge(source_node_id="alice-a", target_node_id="alice-b", relationship_name="knows")],
    )

    data_points = _construct_test_data_points([chunk], [graph])

    alices = [data_point for data_point in data_points if isinstance(data_point, Entity)]
    first = next(node for node in alices if node.description == "first")
    second = next(node for node in alices if node.description == "second")
    assert len(alices) == 2
    assert first.id == Entity.id_for("Alice", "chunk-1", 1)
    assert second.id == Entity.id_for("Alice", "chunk-1", 2)
    assert [(edge.relationship_type, target.id) for edge, target in first.relations] == [
        ("knows", second.id)
    ]


def test_duplicate_extracted_node_ids_are_rejected_before_data_point_construction():
    chunk = _make_chunk()
    graph = _make_graph(
        [
            Node(id="duplicate", name="Alice", type="Person", description="first"),
            Node(id="duplicate", name="Bob", type="Person", description="second"),
        ],
        [],
    )

    with pytest.raises(ValueError, match="Duplicate node id in extracted graph: duplicate"):
        construct_data_points_and_edges([chunk], [graph])

    assert chunk.contains is None


def test_importance_weight_propagates_to_created_nodes():
    chunk = _make_chunk()
    chunk.importance_weight = 0.9
    graph = _make_graph(
        [Node(id="n1", name="Alice", type="Person", description="A person")],
        [],
    )

    data_points = _construct_test_data_points([chunk], [graph])

    alice = next(data_point for data_point in data_points if data_point.name == "alice")
    person = next(data_point for data_point in data_points if data_point.name == "person")
    _, contained_entity = chunk.contains[0]

    assert alice.importance_weight == 0.9
    assert person.importance_weight == 0.9
    assert contained_entity.importance_weight == 0.9


def test_default_importance_weight_propagates_to_created_nodes():
    chunk = _make_chunk()
    graph = _make_graph(
        [Node(id="n1", name="Alice", type="Person", description="A person")],
        [],
    )

    data_points = _construct_test_data_points([chunk], [graph])

    alice = next(data_point for data_point in data_points if data_point.name == "alice")
    person = next(data_point for data_point in data_points if data_point.name == "person")
    _, contained_entity = chunk.contains[0]

    assert alice.importance_weight == 0.5
    assert person.importance_weight == 0.5
    assert contained_entity.importance_weight == 0.5


def test_entity_name_does_not_replace_entity_with_same_extracted_id():
    chunk = _make_chunk()
    graph = _make_graph(
        [
            Node(id="alpha", name="Beta", type="Other", description="beta"),
            Node(id="other", name="alpha", type="Other", description="alpha"),
        ],
        [KGEdge(source_node_id="alpha", target_node_id="other", relationship_name="ref")],
    )

    data_points = _construct_test_data_points([chunk], [graph])

    beta = next(node for node in data_points if isinstance(node, Entity) and node.name == "beta")
    alpha = next(node for node in data_points if isinstance(node, Entity) and node.name == "alpha")

    assert beta.id != alpha.id
    assert beta.ontology_valid is False
    assert alpha.id == Entity.id_for("alpha")
    assert alpha.ontology_valid is False
    assert [(edge.relationship_type, target.name) for edge, target in beta.relations] == [
        ("ref", "alpha")
    ]
    assert [entity.name for _, entity in chunk.contains] == ["beta", "alpha"]
