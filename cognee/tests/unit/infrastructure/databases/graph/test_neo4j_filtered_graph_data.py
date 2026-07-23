from unittest.mock import AsyncMock

import pytest

pytest.importorskip("neo4j")

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


@pytest.mark.asyncio
async def test_filtered_graph_edges_fall_back_to_returned_endpoint_ids():
    adapter = Neo4jAdapter(
        "bolt://unused",
        graph_database_allow_anonymous=True,
        driver=object(),
    )
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


def _adapter_with_empty_results() -> Neo4jAdapter:
    adapter = Neo4jAdapter(
        "bolt://unused",
        graph_database_allow_anonymous=True,
        driver=object(),
    )
    adapter.query = AsyncMock(side_effect=[[], []])
    return adapter


@pytest.mark.asyncio
async def test_filtered_graph_data_binds_values_as_parameters():
    """Filter values must be bound as query parameters, not interpolated into
    the Cypher text — a value containing a quote previously broke the query
    (and left it open to injection)."""
    adapter = _adapter_with_empty_results()

    await adapter.get_filtered_graph_data([{"type": ["Entity", "it's quoted"]}])

    node_call = adapter.query.await_args_list[0]
    node_query = node_call.args[0]
    assert "$filter_values_0" in node_query
    assert "it's quoted" not in node_query
    assert node_call.args[1] == {"filter_values_0": ["Entity", "it's quoted"]}


@pytest.mark.asyncio
async def test_filtered_graph_data_builds_edge_clause_per_alias():
    """The edge clause must be built explicitly for both aliases — the previous
    where_clause.replace("n.", "m.") corrupted values containing the literal
    "n." substring."""
    adapter = _adapter_with_empty_results()

    await adapter.get_filtered_graph_data([{"type": ["Section n.1"]}])

    edge_call = adapter.query.await_args_list[1]
    edge_query = edge_call.args[0]
    assert "n.type IN $filter_values_0" in edge_query
    assert "m.type IN $filter_values_0" in edge_query
    assert edge_call.args[1] == {"filter_values_0": ["Section n.1"]}


@pytest.mark.asyncio
async def test_filtered_graph_data_rejects_unknown_attribute():
    """Attribute names are validated against a whitelist, mirroring the
    postgres/turso adapters."""
    adapter = _adapter_with_empty_results()

    with pytest.raises(ValueError, match="Invalid filter attribute"):
        await adapter.get_filtered_graph_data([{"type = 'x' OR 1=1 //": ["Entity"]}])

    adapter.query.assert_not_awaited()
