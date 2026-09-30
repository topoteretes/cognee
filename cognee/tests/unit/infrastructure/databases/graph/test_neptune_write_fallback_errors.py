"""A Neptune write that failed must not return as if it succeeded.

add_nodes, add_edges and delete_nodes try a bulk query first and fall back to one
item at a time. The fallback caught every per-item failure and moved on, so when
Neptune was unreachable the bulk write failed, every individual write failed, and
the method still returned normally having written nothing. add_node, add_edge and
delete_node called directly raise for the same failure. has_edge answered False for
any failure, which reads as "there is no such edge".

The other graph adapters - ladybug, neo4j, postgres, turso - raise or have no
handler on all four.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB


def _node(node_id):
    return SimpleNamespace(id=node_id, model_dump=lambda: {"id": node_id})


def _stub(**overrides):
    stub = SimpleNamespace(
        _GRAPH_NODE_LABEL="COGNEE_NODE",
        _serialize_properties=NeptuneGraphDB._serialize_properties,
        query=AsyncMock(side_effect=ConnectionError("neptune unreachable")),
        add_node=AsyncMock(side_effect=RuntimeError("Failed to add node")),
        add_edge=AsyncMock(side_effect=RuntimeError("Failed to add edge")),
        delete_node=AsyncMock(side_effect=RuntimeError("Failed to delete node")),
    )
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


@pytest.mark.asyncio
async def test_add_nodes_raises_when_bulk_and_every_fallback_fail():
    with pytest.raises(RuntimeError, match="2 of 2 nodes"):
        await NeptuneGraphDB.add_nodes(_stub(), [_node("a"), _node("b")])


@pytest.mark.asyncio
async def test_add_nodes_fallback_still_succeeds_when_individual_writes_do():
    """The fallback is the design and stays: a failed bulk query salvaged one by one."""
    stub = _stub(add_node=AsyncMock(return_value=None))

    await NeptuneGraphDB.add_nodes(stub, [_node("a"), _node("b")])

    assert stub.add_node.await_count == 2


@pytest.mark.asyncio
async def test_delete_nodes_raises_when_bulk_and_every_fallback_fail():
    with pytest.raises(RuntimeError, match="2 of 2 nodes"):
        await NeptuneGraphDB.delete_nodes(_stub(), ["a", "b"])


@pytest.mark.asyncio
async def test_add_edges_raises_when_bulk_and_every_fallback_fail():
    edges = [("a", "b", "RELATES_TO", {}), ("b", "c", "RELATES_TO", {})]

    with pytest.raises(RuntimeError, match="2 of 2 edges"):
        await NeptuneGraphDB.add_edges(_stub(), edges)


@pytest.mark.asyncio
async def test_has_edge_raises_instead_of_answering_false():
    """False means the edge is not there. An unreachable database must not say that."""
    with pytest.raises(RuntimeError, match="Failed to check edge existence"):
        await NeptuneGraphDB.has_edge(_stub(), "a", "b", "RELATES_TO")


@pytest.mark.asyncio
async def test_has_edge_still_false_when_the_query_finds_nothing():
    stub = _stub(query=AsyncMock(return_value=[]))

    assert await NeptuneGraphDB.has_edge(stub, "a", "b", "RELATES_TO") is False
