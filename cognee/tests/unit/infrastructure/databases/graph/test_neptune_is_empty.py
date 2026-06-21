from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB


def test_neptune_adapter_implements_is_empty():
    """Regression test: NeptuneGraphDB must implement the abstract is_empty.

    GraphDBInterface declares is_empty as an @abstractmethod, and ladybug,
    neo4j and postgres implement it. Neptune did not, so it stayed abstract
    and could not be instantiated at all
    (``TypeError: Can't instantiate abstract class NeptuneGraphDB``).
    """
    assert "is_empty" not in NeptuneGraphDB.__abstractmethods__


@pytest.mark.asyncio
async def test_is_empty_true_when_no_nodes():
    stub = SimpleNamespace(
        _GRAPH_NODE_LABEL="COGNEE_NODE",
        query=AsyncMock(return_value=[{"node_count": 0}]),
    )
    assert await NeptuneGraphDB.is_empty(stub) is True


@pytest.mark.asyncio
async def test_is_empty_false_when_nodes_exist():
    stub = SimpleNamespace(
        _GRAPH_NODE_LABEL="COGNEE_NODE",
        query=AsyncMock(return_value=[{"node_count": 7}]),
    )
    assert await NeptuneGraphDB.is_empty(stub) is False


@pytest.mark.asyncio
async def test_is_empty_true_when_result_empty():
    """A query returning no rows must not raise IndexError; treat as empty."""
    stub = SimpleNamespace(
        _GRAPH_NODE_LABEL="COGNEE_NODE",
        query=AsyncMock(return_value=[]),
    )
    assert await NeptuneGraphDB.is_empty(stub) is True


@pytest.mark.asyncio
async def test_is_empty_wraps_query_error():
    """A failing query is reported instead of bubbling up the raw driver error."""
    stub = SimpleNamespace(
        _GRAPH_NODE_LABEL="COGNEE_NODE",
        query=AsyncMock(side_effect=RuntimeError("boom")),
    )
    with pytest.raises(Exception, match="Failed to check if graph is empty"):
        await NeptuneGraphDB.is_empty(stub)
