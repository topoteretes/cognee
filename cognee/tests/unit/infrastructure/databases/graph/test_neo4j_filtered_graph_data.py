from unittest.mock import AsyncMock

import pytest

pytest.importorskip("neo4j")

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


def _adapter() -> Neo4jAdapter:
    return Neo4jAdapter(
        "bolt://unused",
        graph_database_allow_anonymous=True,
        driver=object(),
    )


@pytest.mark.asyncio
async def test_filtered_graph_edges_fall_back_to_returned_endpoint_ids():
    adapter = _adapter()
    adapter.query = AsyncMock(
        side_effect=[
            [
                {"id": "source", "properties": {"id": "source", "type": "CodeSymbol"}},
                {"id": "target", "properties": {"id": "target", "type": "CodeSymbol"}},
            ],
            [
                {
                    "source": "source",
                    "target": "target",
                    "type": "calls",
                    "properties": {},
                }
            ],
        ]
    )

    _nodes, edges = await adapter.get_filtered_graph_data([{"type": ["CodeSymbol"]}])

    assert edges == [("source", "target", "calls", {})]
    edge_query = adapter.query.await_args_list[1].args[0]
    assert "m.id AS target" in edge_query


@pytest.mark.asyncio
async def test_filtered_graph_data_rejects_attribute_outside_whitelist():
    adapter = _adapter()
    adapter.query = AsyncMock(return_value=[])

    with pytest.raises(ValueError, match="Invalid filter attribute: 'properties'"):
        await adapter.get_filtered_graph_data([{"properties": ["hidden"]}])

    adapter.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_filtered_graph_data_binds_values_instead_of_interpolating():
    """A value containing a quote used to be wrapped in bare quotes and produce
    invalid Cypher. Values now travel as bound parameters."""
    adapter = _adapter()
    adapter.query = AsyncMock(side_effect=[[], []])

    await adapter.get_filtered_graph_data([{"name": ["it's quoted"]}])

    node_query, node_params = adapter.query.await_args_list[0].args
    assert "it's quoted" not in node_query
    assert node_params == {"filter_name": ["it's quoted"]}


@pytest.mark.asyncio
async def test_filtered_graph_data_edge_clause_covers_both_aliases_per_attribute():
    """The edge query must carry one clause per attribute per alias, each bound to
    the same parameter as the node query."""
    adapter = _adapter()
    adapter.query = AsyncMock(side_effect=[[], []])

    await adapter.get_filtered_graph_data([{"name": ["Section n.1"], "type": ["Doc"]}])

    node_query, node_params = adapter.query.await_args_list[0].args
    edge_query, edge_params = adapter.query.await_args_list[1].args

    assert node_query.count(" IN $") == 2
    assert edge_query.count(" IN $") == 4

    for attribute in ("name", "type"):
        assert f"n.{attribute} IN $filter_{attribute}" in node_query
        assert f"n.{attribute} IN $filter_{attribute}" in edge_query
        assert f"m.{attribute} IN $filter_{attribute}" in edge_query

    assert node_params == edge_params == {"filter_name": ["Section n.1"], "filter_type": ["Doc"]}
