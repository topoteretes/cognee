import pytest
from unittest.mock import MagicMock

from cognee.modules.retrieval.utils.used_graph_elements import (
    extract_from_edges,
    extract_from_scored_results,
    extract_from_temporal_dict,
    is_edge_list,
)


def test_is_edge_list_empty_or_not_list():
    """is_edge_list returns False for empty list or non-list."""
    assert is_edge_list([]) is False
    assert is_edge_list(None) is False
    assert is_edge_list("not a list") is False


def test_is_edge_list_true_for_edge_like():
    """is_edge_list returns True for list of edge-like objects."""
    edge = MagicMock()
    edge.node1 = MagicMock()
    edge.node2 = MagicMock()
    edge.attributes = {}
    assert is_edge_list([edge]) is True


def test_is_edge_list_false_for_non_edge():
    """is_edge_list returns False when first element lacks node1/node2/attributes."""
    assert is_edge_list([object()]) is False
    obj = MagicMock()
    obj.node1 = None
    obj.node2 = MagicMock()
    obj.attributes = {}
    assert is_edge_list([obj]) is False


def test_extract_from_edges_empty():
    """extract_from_edges returns None for empty list."""
    assert extract_from_edges([]) is None


def test_extract_from_edges_non_edge_list_returns_none():
    """extract_from_edges with non-edge-like list still iterates; no node1/node2 so result empty -> None."""
    assert extract_from_edges([object()]) is None


def test_extract_from_edges_node_and_edge_ids():
    """extract_from_edges collects node_ids and edge_ids from edge_object_id."""
    n1 = MagicMock()
    n1.id = "node-1"
    n2 = MagicMock()
    n2.id = "node-2"
    edge = MagicMock()
    edge.node1 = n1
    edge.node2 = n2
    edge.attributes = {"edge_object_id": "edge-1"}
    result = extract_from_edges([edge])
    assert result is not None
    assert set(result["node_ids"]) == {"node-1", "node-2"}
    assert result["edge_ids"] == ["edge-1"]


def test_extract_from_edges_only_node_ids_when_no_edge_object_id():
    """extract_from_edges returns only node_ids when edge_object_id missing."""
    n1 = MagicMock()
    n1.id = "n1"
    n2 = MagicMock()
    n2.id = "n2"
    edge = MagicMock()
    edge.node1 = n1
    edge.node2 = n2
    edge.attributes = {}
    result = extract_from_edges([edge])
    assert result == {"node_ids": ["n1", "n2"]}


def test_extract_from_scored_results_empty():
    """extract_from_scored_results returns None for empty or no ids."""
    assert extract_from_scored_results([]) is None


def test_extract_from_scored_results_payload_id():
    """extract_from_scored_results uses payload['id'] when present."""
    r = MagicMock()
    r.payload = {"id": "chunk1"}
    result = extract_from_scored_results([r])
    assert result == {"node_ids": ["chunk1"]}


def test_extract_from_scored_results_fallback_to_id():
    """extract_from_scored_results uses .id when payload has no id."""
    r = MagicMock()
    r.payload = {}
    r.id = "result-id"
    result = extract_from_scored_results([r])
    assert result == {"node_ids": ["result-id"]}


def test_extract_from_scored_results_dedupes():
    """extract_from_scored_results deduplicates and sorts node_ids."""
    r1 = MagicMock()
    r1.payload = {"id": "b"}
    r2 = MagicMock()
    r2.payload = {"id": "a"}
    result = extract_from_scored_results([r1, r2])
    assert result == {"node_ids": ["a", "b"]}


def test_extract_from_temporal_dict_not_dict():
    """extract_from_temporal_dict returns None for non-dict."""
    assert extract_from_temporal_dict(None) is None
    assert extract_from_temporal_dict([]) is None


def test_extract_from_temporal_dict_triplets():
    """extract_from_temporal_dict uses extract_from_edges when triplets is edge list."""
    n1 = MagicMock()
    n1.id = "n1"
    n2 = MagicMock()
    n2.id = "n2"
    edge = MagicMock()
    edge.node1 = n1
    edge.node2 = n2
    edge.attributes = {"edge_object_id": "e1"}
    result = extract_from_temporal_dict({"triplets": [edge]})
    assert result is not None
    assert set(result["node_ids"]) == {"n1", "n2"}
    assert result["edge_ids"] == ["e1"]


def test_extract_from_temporal_dict_vector_search_results():
    """extract_from_temporal_dict uses extract_from_scored_results for vector_search_results."""
    r = MagicMock()
    r.payload = {"id": "event-1"}
    r.id = "event-1"
    result = extract_from_temporal_dict({"vector_search_results": [r]})
    assert result == {"node_ids": ["event-1"]}


def test_extract_from_temporal_dict_empty_returns_none():
    """extract_from_temporal_dict returns None when no triplets or vector_search_results."""
    assert extract_from_temporal_dict({}) is None
    assert extract_from_temporal_dict({"other": []}) is None
    assert extract_from_temporal_dict({"vector_search_results": []}) is None
