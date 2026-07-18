from unittest.mock import MagicMock

import pytest

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models import Entity
from cognee.modules.engine.utils import generate_edge_name
from cognee.modules.graph.utils.expand_with_nodes_and_edges import (
    _create_edge_key,
    expand_with_nodes_and_edges,
)
from cognee.shared.data_models import KnowledgeGraph, Node, Edge as KGEdge


def _mock_resolver():
    resolver = MagicMock()
    resolver.get_subgraph.return_value = ([], [], None)
    return resolver


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


def test_chunk_contains_populated():
    chunk = _make_chunk()
    graph = _make_graph(
        [Node(id="n1", name="Alice", type="Person", description="A person")],
        [],
    )
    chunks, entity_nodes = expand_with_nodes_and_edges([chunk], [graph], _mock_resolver())

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
    _, entity_nodes = expand_with_nodes_and_edges([chunk], [graph], _mock_resolver())

    alice = next(e for e in entity_nodes if e.name == "alice")
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
    existing_edge_key = _create_edge_key(
        Entity.id_for("n1"),
        Entity.id_for("n2"),
        generate_edge_name("knows"),
    )

    _, entity_nodes = expand_with_nodes_and_edges(
        [chunk], [graph], _mock_resolver(), {existing_edge_key: True}
    )

    alice = next(entity for entity in entity_nodes if entity.name == "alice")
    assert alice.relations == []
    assert chunk._provenance_edges == [
        (
            Entity.id_for("n1"),
            Entity.id_for("n2"),
            "knows",
            {
                "relationship_name": "knows",
                "source_node_id": Entity.id_for("n1"),
                "target_node_id": Entity.id_for("n2"),
                "ontology_valid": False,
                "edge_text": None,
            },
        )
    ]


def test_edge_evidence_uses_post_ontology_node_ids():
    from types import SimpleNamespace

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

    expand_with_nodes_and_edges([chunk], [graph], resolver)

    assert chunk._provenance_edges[0][:3] == (
        Entity.id_for("Alicia"),
        Entity.id_for("Robert"),
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

    expand_with_nodes_and_edges([chunk1, chunk2], [graph1, graph2], _mock_resolver())

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

    expand_with_nodes_and_edges([chunk], [graph], _mock_resolver())

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

    _, entity_nodes = expand_with_nodes_and_edges([chunk], [graph], _mock_resolver())

    alice = next(e for e in entity_nodes if e.name == "alice")
    edge_obj, target = alice.relations[0]
    assert target.name == "acme"
    assert edge_obj.relationship_type == "works_at"
    assert edge_obj.edge_text == "Alice works at Acme."


def test_edge_model_does_not_default_edge_text_from_relationship_type():
    assert Edge(relationship_type="contains").edge_text is None


def test_returns_chunks_and_entity_nodes():
    chunk = _make_chunk()
    graph = _make_graph(
        [Node(id="n1", name="Thing", type="Object", description="a thing")],
        [],
    )
    result = expand_with_nodes_and_edges([chunk], [graph], _mock_resolver())

    assert isinstance(result, tuple) and len(result) == 2
    returned_chunks, entity_nodes = result
    assert returned_chunks is not None
    assert isinstance(entity_nodes, list)


def test_empty_graph_skipped():
    chunk = _make_chunk()
    chunks, entity_nodes = expand_with_nodes_and_edges([chunk], [None], _mock_resolver())

    assert chunk.contains is None
    assert entity_nodes == []


def test_entity_deduplication_across_chunks():
    chunk1, chunk2 = _make_chunk(), _make_chunk()
    node = Node(id="n1", name="Alice", type="Person", description="desc")
    graph1 = _make_graph([node], [])
    graph2 = _make_graph([node], [])

    _, entity_nodes = expand_with_nodes_and_edges(
        [chunk1, chunk2], [graph1, graph2], _mock_resolver()
    )

    alice_nodes = [e for e in entity_nodes if e.name == "alice"]
    assert len(alice_nodes) == 1


def test_importance_weight_propagates_to_created_nodes():
    chunk = _make_chunk()
    chunk.importance_weight = 0.9
    graph = _make_graph(
        [Node(id="n1", name="Alice", type="Person", description="A person")],
        [],
    )

    _, entity_nodes = expand_with_nodes_and_edges([chunk], [graph], _mock_resolver())

    alice = next(node for node in entity_nodes if node.name == "alice")
    person = next(node for node in entity_nodes if node.name == "person")
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

    _, entity_nodes = expand_with_nodes_and_edges([chunk], [graph], _mock_resolver())

    alice = next(node for node in entity_nodes if node.name == "alice")
    person = next(node for node in entity_nodes if node.name == "person")
    _, contained_entity = chunk.contains[0]

    assert alice.importance_weight == 0.5
    assert person.importance_weight == 0.5
    assert contained_entity.importance_weight == 0.5
