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
