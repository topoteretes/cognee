"""NeptuneGraphDB.get_nodeset_subgraph must accept node_name_filter_operator.

CogneeGraph and TripletRetriever always pass ``node_name_filter_operator`` (the
interface declares it), but Neptune's override did not take it, so every
``node_name``-scoped search on Neptune raised ``TypeError``. Graph retrievers
swallowed it and returned an empty graph; TRIPLET_COMPLETION raised it.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB

ROWS = [
    {
        "rawNodes": [{"id": 1, "properties": {"name": "a"}}],
        "rawRels": [{"source_id": 1, "target_id": 1, "type": "rel", "properties": {}}],
    }
]


class NodeSet:
    pass


def _stub():
    return SimpleNamespace(_GRAPH_NODE_LABEL="COGNEE_NODE", query=AsyncMock(return_value=ROWS))


def test_signature_matches_interface():
    expected = inspect.signature(GraphDBInterface.get_nodeset_subgraph).parameters
    actual = inspect.signature(NeptuneGraphDB.get_nodeset_subgraph).parameters
    assert actual["node_name_filter_operator"].default == "OR"
    assert list(actual) == list(expected)


@pytest.mark.asyncio
async def test_or_returns_subgraph():
    stub = _stub()
    nodes, edges = await NeptuneGraphDB.get_nodeset_subgraph(
        stub, node_type=NodeSet, node_name=["a", "b"], node_name_filter_operator="OR"
    )
    assert nodes == [(1, {"name": "a"})]
    assert edges == [(1, 1, "rel", {})]
    query = stub.query.await_args.args[0]
    assert "matched_count" not in query


@pytest.mark.asyncio
async def test_and_keeps_only_neighbours_of_every_named_node():
    stub = _stub()
    await NeptuneGraphDB.get_nodeset_subgraph(
        stub, node_type=NodeSet, node_name=["a", "b"], node_name_filter_operator="AND"
    )
    query, params = stub.query.await_args.args
    assert "WHERE matched_count = size(primary)" in query
    assert params == {"names": ["a", "b"], "type": "NodeSet"}
