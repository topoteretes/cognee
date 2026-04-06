import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.graph.utils.expand_with_nodes_and_edges import _create_edge_key
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node

retrieve_module = importlib.import_module("cognee.modules.graph.utils.retrieve_existing_edges")


def _make_chunk(chunk_id: str):
    return MagicMock(id=chunk_id)


def _make_graph(source_id: str, target_id: str, relationship_name: str) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(id=source_id, name=source_id, type="Person", description="desc"),
            Node(id=target_id, name=target_id, type="Person", description="desc"),
        ],
        edges=[
            KGEdge(
                source_node_id=source_id,
                target_node_id=target_id,
                relationship_name=relationship_name,
            )
        ],
    )


@pytest.mark.asyncio
@patch.object(retrieve_module, "get_graph_engine", new_callable=AsyncMock)
async def test_retrieve_existing_edges_queries_graph_edges_from_all_chunks(mock_get_graph_engine):
    graph_engine = MagicMock()
    graph_engine.has_edges = AsyncMock(return_value=[])
    mock_get_graph_engine.return_value = graph_engine

    data_chunks = [_make_chunk("chunk-1"), _make_chunk("chunk-2")]
    chunk_graphs = [
        _make_graph("source-1", "target-1", "knows"),
        _make_graph("source-2", "target-2", "works_with"),
    ]

    await retrieve_module.retrieve_existing_edges(data_chunks, chunk_graphs)

    queried_edges = graph_engine.has_edges.await_args.args[0]

    assert ("source-1", "target-1", "knows") in queried_edges
    assert ("source-2", "target-2", "works_with") in queried_edges


@pytest.mark.asyncio
@patch.object(retrieve_module, "get_graph_engine", new_callable=AsyncMock)
async def test_retrieve_existing_edges_uses_same_key_format_as_expand(mock_get_graph_engine):
    graph_engine = MagicMock()
    graph_engine.has_edges = AsyncMock(return_value=[("source", "target", "works_with")])
    mock_get_graph_engine.return_value = graph_engine

    existing_edges_map = await retrieve_module.retrieve_existing_edges(
        [_make_chunk("chunk-1")],
        [_make_graph("source", "target", "works_with")],
    )

    assert existing_edges_map == {_create_edge_key("source", "target", "works_with"): True}
