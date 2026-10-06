from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


def _metrics_adapter(num_nodes: int, num_edges: int) -> Neo4jAdapter:
    """Adapter wired so every metric helper finds the key it expects."""
    adapter = object.__new__(Neo4jAdapter)
    adapter.drop_graph = AsyncMock()
    adapter.project_entire_graph = AsyncMock()

    async def fake_query(query, params=None):
        # The GDS metric queries embed count(...) subexpressions, so their own
        # projection aliases must be matched before the bare count() checks.
        if "AS num_connected_components" in query:
            return [{"num_connected_components": 1}]
        if "AS size" in query:
            return [{"size": num_nodes}]
        if "YIELD distance" in query:  # shortest paths (optional metrics only)
            return []
        if "adapter_loop_count" in query:  # self-loops (optional metrics only)
            return [{"adapter_loop_count": 0}]
        if "avg_clustering" in query:
            return [{"avg_clustering": 0.25}]
        if "count(n)" in query:
            return [{"count": num_nodes}]
        if "count(r)" in query:
            return [{"count": num_edges}]
        raise AssertionError(f"unexpected query: {query}")

    adapter.query = AsyncMock(side_effect=fake_query)
    return adapter


def _queries(adapter: Neo4jAdapter) -> list[str]:
    return [call.args[0] for call in adapter.query.await_args_list]


@pytest.mark.asyncio
async def test_graph_counts_use_aggregation_and_nothing_else():
    """Regression for the graph-summary OOM (#4832): counting by collect(n) /
    collect([n, r, m]) and taking len() materialized the whole graph inside one
    transaction. get_graph_counts must issue only the two count queries — no
    collect, no GDS projection, no component metrics."""
    adapter = _metrics_adapter(num_nodes=7, num_edges=11)

    assert await adapter.get_graph_counts() == (7, 11)

    queries = _queries(adapter)
    assert len(queries) == 2
    assert not any("collect(" in query for query in queries)
    adapter.drop_graph.assert_not_awaited()
    adapter.project_entire_graph.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_metrics_compute_components_and_density_from_the_counts():
    """get_graph_metrics computes components on every call, like the other
    adapters, and derives edge density from its own counts instead of a separate
    unlabelled count query that could disagree with them."""
    adapter = _metrics_adapter(num_nodes=7, num_edges=11)

    metrics = await adapter.get_graph_metrics(include_optional=False)

    assert metrics["num_nodes"] == 7
    assert metrics["num_edges"] == 11
    assert metrics["mean_degree"] == pytest.approx(2 * 11 / 7)
    assert metrics["edge_density"] == pytest.approx(11 / (7 * 6))
    assert metrics["num_connected_components"] == 1
    assert metrics["sizes_of_connected_components"] == [7]
    adapter.drop_graph.assert_awaited_once()
    adapter.project_entire_graph.assert_awaited_once()
    assert not any("collect(" in query for query in _queries(adapter))


@pytest.mark.asyncio
async def test_graph_metrics_full_path_still_projects():
    adapter = _metrics_adapter(num_nodes=7, num_edges=11)

    metrics = await adapter.get_graph_metrics(include_optional=True)

    adapter.drop_graph.assert_awaited_once()
    adapter.project_entire_graph.assert_awaited_once()
    assert metrics["num_connected_components"] == 1
    assert metrics["num_selfloops"] == 0
    assert metrics["avg_clustering"] == 0.25


@pytest.mark.asyncio
async def test_graph_metrics_handles_empty_graph():
    adapter = _metrics_adapter(num_nodes=0, num_edges=0)

    metrics = await adapter.get_graph_metrics(include_optional=False)

    assert metrics["num_nodes"] == 0
    assert metrics["num_edges"] == 0
    assert metrics["mean_degree"] is None
    assert metrics["edge_density"] == 0
