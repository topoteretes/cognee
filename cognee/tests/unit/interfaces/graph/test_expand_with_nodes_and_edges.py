from unittest.mock import MagicMock

import pytest

from cognee.modules.graph.utils.expand_with_nodes_and_edges import expand_with_nodes_and_edges
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
