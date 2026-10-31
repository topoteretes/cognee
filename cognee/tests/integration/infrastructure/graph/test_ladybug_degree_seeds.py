"""Execute seed selection against Ladybug, including graphs without edges."""

from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_degree_seeds_include_isolated_nodes_and_rank_both_endpoints(tmp_path):
    module = pytest.importorskip("cognee.infrastructure.databases.graph.ladybug.adapter")
    adapter = module.LadybugAdapter(
        str(tmp_path / "degree_seeds"),
        kuzu_num_threads=2,
        kuzu_buffer_pool_size=64 * 1024 * 1024,
        kuzu_max_db_size=256 * 1024 * 1024,
    )
    adapter.get_graph_data = AsyncMock(side_effect=AssertionError("unexpected full graph read"))
    try:
        assert await adapter.get_top_degree_node_ids(5) == []

        for node_id in ["hub", "incoming", "outgoing", "isolated"]:
            await adapter.query("CREATE (:Node {id: $id})", {"id": node_id})

        assert set(await adapter.get_top_degree_node_ids(10)) == {
            "hub",
            "incoming",
            "outgoing",
            "isolated",
        }
        assert len(await adapter.get_top_degree_node_ids(2)) == 2

        for source, target in [("incoming", "hub"), ("hub", "outgoing")]:
            await adapter.query(
                "MATCH (a:Node), (b:Node) WHERE a.id = $source AND b.id = $target "
                "CREATE (a)-[:EDGE {relationship_name: 'rel'}]->(b)",
                {"source": source, "target": target},
            )

        assert await adapter.get_top_degree_node_ids(1) == ["hub"]
        seeds = await adapter.get_top_degree_node_ids(10)
        assert seeds[0] == "hub"
        assert set(seeds) == {"hub", "incoming", "outgoing", "isolated"}
        adapter.get_graph_data.assert_not_awaited()
    finally:
        await adapter.close()
