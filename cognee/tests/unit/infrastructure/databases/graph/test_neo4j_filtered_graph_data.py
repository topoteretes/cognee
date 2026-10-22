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
    assert node_params == {"filter_0_name": ["it's quoted"]}


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
        assert f"n.{attribute} IN $filter_0_{attribute}" in node_query
        assert f"n.{attribute} IN $filter_0_{attribute}" in edge_query
        assert f"m.{attribute} IN $filter_0_{attribute}" in edge_query

    assert (
        node_params == edge_params == {"filter_0_name": ["Section n.1"], "filter_0_type": ["Doc"]}
    )


@pytest.mark.asyncio
async def test_filtered_graph_data_ands_every_filter_dict():
    adapter = _adapter()
    adapter.query = AsyncMock(side_effect=[[], []])

    await adapter.get_filtered_graph_data([{"type": ["Entity"]}, {"name": ["Alice"]}])

    node_query, node_params = adapter.query.await_args_list[0].args
    assert "n.type IN $filter_0_type" in node_query
    assert "n.name IN $filter_1_name" in node_query
    assert node_params == {"filter_0_type": ["Entity"], "filter_1_name": ["Alice"]}


@pytest.mark.asyncio
async def test_filtered_graph_data_repeated_attribute_keeps_both_params():
    """Two dicts filtering the same attribute must not collide on one parameter name."""
    adapter = _adapter()
    adapter.query = AsyncMock(side_effect=[[], []])

    await adapter.get_filtered_graph_data([{"type": ["Entity"]}, {"type": ["Chunk"]}])

    node_query, node_params = adapter.query.await_args_list[0].args
    assert node_query.count(" IN $") == 2
    assert node_params == {"filter_0_type": ["Entity"], "filter_1_type": ["Chunk"]}


@pytest.mark.asyncio
async def test_filtered_graph_data_empty_value_list_matches_nothing():
    """An empty value list is bound as an empty list, which Cypher treats as a
    match-nothing IN, the same semantics turso and postgres_demo implement."""
    adapter = _adapter()
    adapter.query = AsyncMock(side_effect=[[], []])

    await adapter.get_filtered_graph_data([{"type": []}])

    node_query, node_params = adapter.query.await_args_list[0].args
    assert "n.type IN $filter_0_type" in node_query
    assert node_params == {"filter_0_type": []}


@pytest.mark.asyncio
async def test_filtered_graph_data_binds_non_string_values_unquoted():
    adapter = _adapter()
    adapter.query = AsyncMock(side_effect=[[], []])

    await adapter.get_filtered_graph_data([{"id": [1, 2]}])

    node_query, node_params = adapter.query.await_args_list[0].args
    assert "n.id IN $filter_0_id" in node_query
    assert node_params == {"filter_0_id": [1, 2]}


@pytest.mark.asyncio
async def test_filtered_graph_data_no_filters_issues_no_query():
    adapter = _adapter()
    adapter.query = AsyncMock(return_value=[])

    assert await adapter.get_filtered_graph_data([]) == ([], [])
    assert await adapter.get_filtered_graph_data([{}]) == ([], [])

    adapter.query.assert_not_awaited()
