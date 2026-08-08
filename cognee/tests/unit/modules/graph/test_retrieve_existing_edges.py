import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.infrastructure.databases.provenance import EdgeIdentity
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
