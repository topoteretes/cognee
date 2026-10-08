"""NodeSet query contract tests without an external Neptune service."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognee.modules.engine.models.node_set import NodeSet

try:
    from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB
except ModuleNotFoundError as error:
    if error.name != "botocore":
        raise
    # These tests bypass AWS client initialization, so Config only needs to be importable.
    config_module = ModuleType("botocore.config")
    config_module.Config = MagicMock()
    sys.modules["botocore.config"] = config_module
    try:
        from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB
    finally:
        del sys.modules["botocore.config"]


@pytest.fixture
def adapter():
    graph = NeptuneGraphDB.__new__(NeptuneGraphDB)
    graph.query = AsyncMock(return_value=[])
    return graph


@pytest.mark.asyncio
@pytest.mark.parametrize("operator", [None, "OR", "AND"])
async def test_nodeset_query_selects_union_or_common_neighbors(adapter, operator):
    options = {} if operator is None else {"node_name_filter_operator": operator}
    names = ["A", "B`' WITH 1 AS injected"]

    await adapter.get_nodeset_subgraph(NodeSet, names, **options)

    query, params = adapter.query.call_args.args
    query = " ".join(query.split())
    assert params == {"names": names, "type": "NodeSet"}
    assert names[1] not in query
    assert "n.name = wantedName AND n.type = $type" in query
    assert "OPTIONAL MATCH (p)-[r]-(nbr:COGNEE_NODE)" in query
    assert "primary + nbrs AS nodelist" in query
    if operator == "AND":
        assert "count(DISTINCT p) AS matched_count" in query
        assert "CASE WHEN matched_count = size(primary) THEN nbr ELSE null END" in query
    else:
        assert "matched_count" not in query


@pytest.mark.asyncio
async def test_nodeset_subgraph_keeps_stored_edge_direction(adapter):
    adapter.query.return_value = [
        {
            "rawNodes": [
                {"id": "set-a", "properties": {"name": "A", "type": "NodeSet"}},
                {"id": "entity", "properties": {"name": "Entity", "type": "Entity"}},
            ],
            "rawRels": [
                {
                    "source_id": "entity",
                    "target_id": "set-a",
                    "type": "belongs_to",
                    "properties": {"weight": 1},
                }
            ],
        }
    ]

    nodes, edges = await adapter.get_nodeset_subgraph(NodeSet, ["A"])

    assert nodes == [
        ("set-a", {"name": "A", "type": "NodeSet"}),
        ("entity", {"name": "Entity", "type": "Entity"}),
    ]
    assert edges == [("entity", "set-a", "belongs_to", {"weight": 1})]
    query = " ".join(adapter.query.call_args.args[0].split())
    assert "source_id: id(startNode(r))" in query
    assert "target_id: id(endNode(r))" in query


@pytest.mark.asyncio
@pytest.mark.parametrize("operator", ["OR", "AND"])
async def test_nodeset_subgraph_keeps_primary_nodes_without_edges(adapter, operator):
    adapter.query.return_value = [
        {
            "rawNodes": [{"id": "set-a", "properties": {"name": "A", "type": "NodeSet"}}],
            "rawRels": [],
        }
    ]

    result = await adapter.get_nodeset_subgraph(NodeSet, ["A"], node_name_filter_operator=operator)

    assert result == ([("set-a", {"name": "A", "type": "NodeSet"})], [])
    query = " ".join(adapter.query.call_args.args[0].split())
    assert "OPTIONAL MATCH (a:COGNEE_NODE)-[r]->(b:COGNEE_NODE)" in query
    assert "WHERE a IN nodes AND b IN nodes" in query


@pytest.mark.asyncio
@pytest.mark.parametrize("operator", ["OR", "AND"])
async def test_empty_nodeset_names_return_empty_without_query(adapter, operator):
    result = await adapter.get_nodeset_subgraph(NodeSet, [], node_name_filter_operator=operator)

    assert result == ([], [])
    adapter.query.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operator", ["OR", "AND"])
@pytest.mark.parametrize("rows", [[], [{"rawNodes": [], "rawRels": []}]])
async def test_unknown_nodeset_names_return_empty(adapter, operator, rows):
    adapter.query.return_value = rows

    assert await adapter.get_nodeset_subgraph(
        NodeSet, ["missing"], node_name_filter_operator=operator
    ) == ([], [])


@pytest.mark.asyncio
@pytest.mark.parametrize("operator", ["XOR", "or"])
async def test_invalid_nodeset_operator_is_rejected_before_query(adapter, operator):
    with pytest.raises(ValueError, match="node_name_filter_operator"):
        await adapter.get_nodeset_subgraph(NodeSet, ["A"], node_name_filter_operator=operator)

    adapter.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_nodeset_query_error_retains_its_cause(adapter):
    error = RuntimeError("Neptune unavailable")
    adapter.query.side_effect = error

    with pytest.raises(RuntimeError, match="Failed to get nodeset subgraph") as captured:
        await adapter.get_nodeset_subgraph(NodeSet, ["A"], node_name_filter_operator="AND")

    assert captured.value.__cause__ is error
