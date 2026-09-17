import importlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key
from cognee.modules.engine.models import Entity

existing_edges_module = importlib.import_module(
    "cognee.modules.graph.utils.retrieve_existing_edges"
)


@pytest.mark.asyncio
@patch.object(existing_edges_module, "get_graph_engine", new_callable=AsyncMock)
async def test_find_existing_edge_identities_returns_edge_identity_objects(mock_get_graph_engine):
    graph_engine = MagicMock()
    edge_identity = EdgeIdentity(
        source_id=str(Entity.id_for("Alice")),
        target_id=str(Entity.id_for("Bob")),
        relationship_name="knows",
    )
    edge_tuple = (
        edge_identity.source_id,
        edge_identity.target_id,
        edge_identity.relationship_name,
    )
    graph_engine.has_edges = AsyncMock(return_value=[edge_tuple])
    mock_get_graph_engine.return_value = graph_engine

    existing_edge_identities = await existing_edges_module.find_existing_edge_identities(
        {edge_identity}
    )

    graph_engine.has_edges.assert_awaited_once_with([edge_tuple])
    assert existing_edge_identities == {edge_identity}


@pytest.mark.asyncio
@patch.object(existing_edges_module, "graph_provenance_write_kwargs", new_callable=AsyncMock)
@patch.object(existing_edges_module, "get_graph_engine", new_callable=AsyncMock)
async def test_repeated_existing_edge_gets_one_batched_source_ref_attach(
    mock_get_graph_engine,
    mock_graph_provenance_write_kwargs,
):
    edge_identity = EdgeIdentity(
        source_id=str(Entity.id_for("Alice")),
        target_id=str(Entity.id_for("Acme")),
        relationship_name="works_at",
    )
    existing_edge = (
        edge_identity.source_id,
        edge_identity.target_id,
        edge_identity.relationship_name,
    )

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

    existing_edge_identities = await existing_edges_module.find_existing_edge_identities(
        [edge_identity],
        ctx=ctx,
    )

    assert existing_edge_identities == {edge_identity}
    mock_graph_provenance_write_kwargs.assert_awaited_once_with(graph_engine, ctx)
    graph_engine.attach_edge_source_refs.assert_awaited_once_with(
        [edge_identity],
        [source_ref_key],
        str(pipeline_run_id),
    )
    graph_engine.add_edges.assert_not_called()


@pytest.mark.asyncio
@patch.object(existing_edges_module, "graph_provenance_write_kwargs", new_callable=AsyncMock)
@patch.object(existing_edges_module, "get_graph_engine", new_callable=AsyncMock)
async def test_existing_edge_is_not_attached_on_relational_ledger_graph(
    mock_get_graph_engine,
    mock_graph_provenance_write_kwargs,
):
    edge_identity = EdgeIdentity(
        source_id=str(Entity.id_for("Alice")),
        target_id=str(Entity.id_for("Acme")),
        relationship_name="works_at",
    )
    graph_engine = MagicMock()
    graph_engine.has_edges = AsyncMock(
        return_value=[
            (
                edge_identity.source_id,
                edge_identity.target_id,
                edge_identity.relationship_name,
            )
        ]
    )
    graph_engine.attach_edge_source_refs = AsyncMock()
    mock_get_graph_engine.return_value = graph_engine
    mock_graph_provenance_write_kwargs.return_value = {
        "source_ref_key": None,
        "pipeline_run_id": None,
    }

    await existing_edges_module.find_existing_edge_identities(
        [edge_identity],
        ctx=MagicMock(),
    )

    graph_engine.attach_edge_source_refs.assert_not_called()


@pytest.mark.asyncio
@patch.object(existing_edges_module, "graph_provenance_write_kwargs", new_callable=AsyncMock)
@patch.object(existing_edges_module, "get_graph_engine", new_callable=AsyncMock)
async def test_no_ctx_skips_source_ref_attach(
    mock_get_graph_engine,
    mock_graph_provenance_write_kwargs,
):
    edge_identity = EdgeIdentity(
        source_id=str(Entity.id_for("Alice")),
        target_id=str(Entity.id_for("Acme")),
        relationship_name="works_at",
    )
    graph_engine = MagicMock()
    graph_engine.has_edges = AsyncMock(
        return_value=[
            (
                edge_identity.source_id,
                edge_identity.target_id,
                edge_identity.relationship_name,
            )
        ]
    )
    graph_engine.attach_edge_source_refs = AsyncMock()
    mock_get_graph_engine.return_value = graph_engine

    existing_edge_identities = await existing_edges_module.find_existing_edge_identities(
        [edge_identity]
    )

    assert existing_edge_identities == {edge_identity}
    mock_graph_provenance_write_kwargs.assert_not_called()
    graph_engine.attach_edge_source_refs.assert_not_called()
