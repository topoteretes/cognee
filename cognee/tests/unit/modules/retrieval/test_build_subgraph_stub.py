from unittest.mock import MagicMock

import pytest

from cognee.modules.retrieval.utils.used_graph_elements import (
    build_retrieved_subgraph,
    build_subgraph_stub_from_edges,
    supports_subgraph_search_type,
)
from cognee.modules.search.types.SearchType import SearchType


def _edge(
    source_id: str,
    target_id: str,
    *,
    relationship: str = "discovered",
    score: float = 0.83,
    source_name: str = "Marie Curie",
    target_name: str = "Radium",
    source_type: str = "Entity",
    target_type: str = "Entity",
):
    n1 = MagicMock()
    n1.id = source_id
    n1.attributes = {"name": source_name, "type": source_type, "vector_distance": [score]}
    n2 = MagicMock()
    n2.id = target_id
    n2.attributes = {"name": target_name, "type": target_type, "vector_distance": [score]}
    edge = MagicMock()
    edge.node1 = n1
    edge.node2 = n2
    edge.attributes = {
        "relationship_name": relationship,
        "vector_distance": [score],
    }
    return edge


def test_build_subgraph_stub_from_edges_populates_nodes_and_edges():
    edge = _edge("node-a", "node-b")
    subgraph = build_subgraph_stub_from_edges([edge])

    assert len(subgraph["nodes"]) == 2
    assert {node["id"] for node in subgraph["nodes"]} == {"node-a", "node-b"}
    assert subgraph["nodes"][0]["type"] == "Entity"
    assert subgraph["nodes"][0]["name"] == "Marie Curie"
    assert subgraph["edges"] == [
        {
            "source": "node-a",
            "target": "node-b",
            "relationship": "discovered",
            "score": 0.83,
        }
    ]


def test_build_subgraph_stub_empty_edges():
    assert build_subgraph_stub_from_edges([]) == {"nodes": [], "edges": []}


def test_build_retrieved_subgraph_batch_returns_one_stub_per_query():
    batch = [
        [_edge("a1", "b1")],
        [_edge("a2", "b2", source_name="Ada", target_name="Lovelace")],
    ]
    result = build_retrieved_subgraph(batch, SearchType.GRAPH_COMPLETION)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["edges"][0]["source"] == "a1"
    assert result[1]["nodes"][0]["name"] == "Ada"


def test_build_retrieved_subgraph_non_graph_search_type_returns_none():
    assert build_retrieved_subgraph([], SearchType.CHUNKS) is None


def test_build_subgraph_stub_edge_endpoints_exist_in_nodes():
    edge = _edge("node-a", "node-b")
    subgraph = build_subgraph_stub_from_edges([edge])
    node_ids = {node["id"] for node in subgraph["nodes"]}
    for edge_stub in subgraph["edges"]:
        assert edge_stub["source"] in node_ids
        assert edge_stub["target"] in node_ids


def test_build_retrieved_subgraph_empty_hits_returns_empty_stub_not_error():
    assert build_retrieved_subgraph([], SearchType.GRAPH_COMPLETION) == {
        "nodes": [],
        "edges": [],
    }


def test_supports_subgraph_search_type():
    assert supports_subgraph_search_type(SearchType.GRAPH_COMPLETION) is True
    assert supports_subgraph_search_type(SearchType.TEMPORAL) is True
    assert supports_subgraph_search_type(SearchType.RAG_COMPLETION) is False
