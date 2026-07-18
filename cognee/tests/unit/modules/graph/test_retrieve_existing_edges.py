import importlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key
from cognee.modules.engine.models import Entity
from cognee.modules.engine.utils import generate_edge_name, generate_node_id
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
        _make_graph("Source 1", "Target 1", "Knows"),
        _make_graph("Source 2", "Target 2", "Works With"),
    ]

    await retrieve_module.retrieve_existing_edges(data_chunks, chunk_graphs)

    queried_edges = graph_engine.has_edges.await_args.args[0]

    assert (
        Entity.id_for("Source 1"),
        Entity.id_for("Target 1"),
        generate_edge_name("Knows"),
    ) in queried_edges
    assert (
        Entity.id_for("Source 2"),
        Entity.id_for("Target 2"),
        generate_edge_name("Works With"),
    ) in queried_edges


@pytest.mark.asyncio
@patch.object(retrieve_module, "get_graph_engine", new_callable=AsyncMock)
async def test_retrieve_existing_edges_uses_same_key_format_as_expand(mock_get_graph_engine):
    graph_engine = MagicMock()
    graph_engine.has_edges = AsyncMock(
        return_value=[
            (
                generate_node_id("Source Node"),
                generate_node_id("Target Node"),
                generate_edge_name("Works With"),
            )
        ]
    )
    mock_get_graph_engine.return_value = graph_engine

    existing_edges_map = await retrieve_module.retrieve_existing_edges(
        [_make_chunk("chunk-1")],
        [_make_graph("Source Node", "Target Node", "Works With")],
    )

    expected_key = _create_edge_key(
        generate_node_id("Source Node"),
        generate_node_id("Target Node"),
        generate_edge_name("Works With"),
    )

    assert existing_edges_map == {expected_key: True}


@pytest.mark.asyncio
@patch.object(retrieve_module, "graph_provenance_write_kwargs", new_callable=AsyncMock)
@patch.object(retrieve_module, "get_graph_engine", new_callable=AsyncMock)
async def test_repeated_existing_edge_gets_one_batched_source_ref_attach(
    mock_get_graph_engine,
    mock_graph_provenance_write_kwargs,
):
    source_id = str(Entity.id_for("Alice"))
    target_id = str(Entity.id_for("Acme"))
    relationship_name = generate_edge_name("Works At")
    existing_edge = (source_id, target_id, relationship_name)

    graph_engine = MagicMock()
    graph_engine.has_edges = AsyncMock(return_value=[existing_edge, existing_edge])
    graph_engine.attach_edge_source_refs = AsyncMock()
    graph_engine.add_edges = AsyncMock()
    mock_get_graph_engine.return_value = graph_engine

    dataset_id = uuid4()
    data_id = uuid4()
    pipeline_run_id = uuid4()
    source_ref_key = make_source_ref_key(dataset_id, data_id)
    mock_graph_provenance_write_kwargs.return_value = {
        "source_ref_key": source_ref_key,
        "pipeline_run_id": str(pipeline_run_id),
    }
    ctx = MagicMock()

    existing_edges_map = await retrieve_module.retrieve_existing_edges(
        [_make_chunk("chunk-1"), _make_chunk("chunk-2")],
        [
            _make_graph("Alice", "Acme", "Works At"),
            _make_graph("Alice", "Acme", "Works At"),
        ],
        ctx=ctx,
    )

    assert existing_edges_map == {_create_edge_key(source_id, target_id, relationship_name): True}
    mock_graph_provenance_write_kwargs.assert_awaited_once_with(graph_engine, ctx)
    graph_engine.attach_edge_source_refs.assert_awaited_once_with(
        [EdgeIdentity(source_id, target_id, relationship_name)],
        [source_ref_key],
        str(pipeline_run_id),
    )
    graph_engine.add_edges.assert_not_called()


@pytest.mark.asyncio
@patch.object(retrieve_module, "graph_provenance_write_kwargs", new_callable=AsyncMock)
@patch.object(retrieve_module, "get_graph_engine", new_callable=AsyncMock)
async def test_existing_edge_is_not_attached_on_relational_ledger_graph(
    mock_get_graph_engine,
    mock_graph_provenance_write_kwargs,
):
    existing_edge = (
        str(Entity.id_for("Alice")),
        str(Entity.id_for("Acme")),
        generate_edge_name("Works At"),
    )
    graph_engine = MagicMock()
    graph_engine.has_edges = AsyncMock(return_value=[existing_edge])
    graph_engine.attach_edge_source_refs = AsyncMock()
    mock_get_graph_engine.return_value = graph_engine
    mock_graph_provenance_write_kwargs.return_value = {
        "source_ref_key": None,
        "pipeline_run_id": None,
    }

    await retrieve_module.retrieve_existing_edges(
        [_make_chunk("chunk-1")],
        [_make_graph("Alice", "Acme", "Works At")],
        ctx=MagicMock(),
    )

    graph_engine.attach_edge_source_refs.assert_not_called()


@pytest.mark.asyncio
@patch.object(retrieve_module, "graph_provenance_write_kwargs", new_callable=AsyncMock)
@patch.object(retrieve_module, "get_graph_engine", new_callable=AsyncMock)
async def test_boolean_has_edges_results_are_mapped_back_to_edge_identities(
    mock_get_graph_engine,
    mock_graph_provenance_write_kwargs,
):
    semantic_edge = (
        Entity.id_for("Alice"),
        Entity.id_for("Acme"),
        generate_edge_name("Works At"),
    )
    graph_engine = MagicMock()

    async def boolean_results(candidate_edges):
        return [edge == semantic_edge for edge in candidate_edges]

    graph_engine.has_edges = AsyncMock(side_effect=boolean_results)
    graph_engine.attach_edge_source_refs = AsyncMock()
    mock_get_graph_engine.return_value = graph_engine
    source_ref_key = make_source_ref_key(uuid4(), uuid4())
    mock_graph_provenance_write_kwargs.return_value = {
        "source_ref_key": source_ref_key,
        "pipeline_run_id": None,
    }

    existing_edges_map = await retrieve_module.retrieve_existing_edges(
        [_make_chunk("chunk-1")],
        [_make_graph("Alice", "Acme", "Works At")],
        ctx=MagicMock(),
    )

    normalized_edge = EdgeIdentity(*(str(value) for value in semantic_edge))
    assert existing_edges_map == {
        _create_edge_key(
            normalized_edge.source_id,
            normalized_edge.target_id,
            normalized_edge.relationship_name,
        ): True
    }
    graph_engine.attach_edge_source_refs.assert_awaited_once_with(
        [normalized_edge], [source_ref_key], None
    )
