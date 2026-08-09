from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface


@pytest.mark.asyncio
async def test_default_edge_type_counts_use_current_graph_state():
    """The fallback counts persisted edges, not the batch being written."""
    graph = SimpleNamespace(
        get_graph_data=AsyncMock(
            return_value=(
                [],
                [
                    ("a", "b", "depends_on", {}),
                    ("c", "d", "depends_on", {}),
                    ("e", "f", "contains", {}),
                ],
            )
        )
    )

    assert await GraphDBInterface.get_edge_type_counts(graph, ["depends_on", "missing"]) == {
        "depends_on": 2,
        "missing": 0,
    }


@pytest.mark.asyncio
async def test_default_edge_type_counts_returns_empty_mapping_without_requested_names():
    """An empty request avoids retrieving graph data and has no implicit relationship types."""
    graph = SimpleNamespace(get_graph_data=AsyncMock())

    assert await GraphDBInterface.get_edge_type_counts(graph, []) == {}
    graph.get_graph_data.assert_not_awaited()
