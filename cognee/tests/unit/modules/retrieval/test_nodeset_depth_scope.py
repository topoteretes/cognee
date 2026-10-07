"""Keyless regression coverage for NodeSet-scoped neighborhood retrieval."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    _get_top_triplet_importances,
    get_memory_fragment,
)
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch


class _Group:
    pass


@pytest.fixture
def graph_adapter():
    scoped_nodes = [
        ("seed", {"name": "seed"}),
        ("inside", {"name": "inside"}),
        ("unreachable", {"name": "unreachable"}),
    ]
    edges = [
        ("seed", "inside", "related", {}),
        ("seed", "excluded", "related", {}),
        ("excluded", "unreachable", "related", {}),
    ]
    return SimpleNamespace(
        get_nodeset_subgraph=AsyncMock(return_value=(scoped_nodes, edges)),
        get_neighborhood=AsyncMock(
            return_value=(scoped_nodes + [("excluded", {"name": "excluded"})], edges)
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", [None, NodeSet, _Group])
@pytest.mark.parametrize("operator", ["OR", "AND"])
async def test_depth_search_forwards_scope_before_traversal(graph_adapter, node_type, operator):
    async def scoped_subgraph(*, node_type, node_name, node_name_filter_operator):
        assert node_type is expected_type
        assert node_name == ["A", "B"]
        assert node_name_filter_operator == operator
        return graph_adapter.get_nodeset_subgraph.return_value

    expected_type = node_type or NodeSet
    graph_adapter.get_nodeset_subgraph.side_effect = scoped_subgraph
    fragment = await get_memory_fragment(
        graph_engine=graph_adapter,
        node_type=node_type,
        node_name=["A", "B"],
        node_name_filter_operator=operator,
        relevant_ids_to_filter=["seed"],
        neighborhood_depth=2,
        feedback_influence=0.0,
    )

    assert set(fragment.nodes) == {"seed", "inside"}
    assert {(edge.node1.id, edge.node2.id) for edge in fragment.edges} == {("seed", "inside")}


@pytest.mark.asyncio
@pytest.mark.parametrize("wide_search_limit", [None, 1])
async def test_scoped_depth_keeps_vector_seeds_without_a_wide_limit(
    graph_adapter, wide_search_limit
):
    search = NodeEdgeVectorSearch(vector_engine=AsyncMock())
    search.node_distances = {"Entity_name": [SimpleNamespace(id="seed", score=0.1)]}

    results = await _get_top_triplet_importances(
        memory_fragment=None,
        vector_search=search,
        properties_to_project=None,
        node_type=NodeSet,
        node_name=["A"],
        node_name_filter_operator="OR",
        triplet_distance_penalty=6.5,
        feedback_influence=0.0,
        wide_search_limit=wide_search_limit,
        top_k=5,
        graph_engine=graph_adapter,
        neighborhood_depth=2,
    )

    assert {(edge.node1.id, edge.node2.id) for edge in results} == {("seed", "inside")}


@pytest.mark.asyncio
async def test_unfiltered_depth_keeps_native_adapter_results(graph_adapter):
    fragment = await get_memory_fragment(
        graph_engine=graph_adapter,
        relevant_ids_to_filter=["seed"],
        neighborhood_depth=2,
        feedback_influence=0.0,
    )

    assert set(fragment.nodes) == {"seed", "inside", "excluded", "unreachable"}
    assert {(edge.node1.id, edge.node2.id) for edge in fragment.edges} == {
        ("seed", "inside"),
        ("seed", "excluded"),
        ("excluded", "unreachable"),
    }
