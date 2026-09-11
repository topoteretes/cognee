from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


def _metrics_adapter(num_nodes: int, num_edges: int) -> Neo4jAdapter:
    """Adapter wired so every metric helper finds the key it expects."""
    adapter = object.__new__(Neo4jAdapter)
    adapter.get_model_independent_graph_data = AsyncMock()
    adapter.drop_graph = AsyncMock()
    adapter.project_entire_graph = AsyncMock()

    async def fake_query(query, params=None):
        # The GDS metric queries embed count(...) subexpressions, so their own
        # projection aliases must be matched before the bare count() checks.
        if "AS edge_density" in query:
            return [{"edge_density": 0.5}]
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


@pytest.mark.asyncio
async def test_graph_metrics_counts_without_materializing_the_graph():
    """Regression for the graph-summary OOM: get_graph_metrics used
    get_model_independent_graph_data() — collect(n) / collect([n, r, m]) — only to
    take len() of the results, materializing the entire graph (all nodes and edges
    with all properties) inside one transaction. On large graphs that exhausts the
    Neo4j transaction memory pool. It must count with aggregation queries instead.
    """
    adapter = _metrics_adapter(num_nodes=7, num_edges=11)

    metrics = await adapter.get_graph_metrics(include_optional=False)

    adapter.get_model_independent_graph_data.assert_not_awaited()
    assert metrics["num_nodes"] == 7
    assert metrics["num_edges"] == 11
    assert metrics["mean_degree"] == pytest.approx(2 * 11 / 7)
    count_queries = [c.args[0] for c in adapter.query.await_args_list if "count(" in c.args[0]]
    assert any("count(n)" in q for q in count_queries)
    assert any("count(r)" in q for q in count_queries)


@pytest.mark.asyncio
async def test_graph_metrics_count_only_path_skips_the_gds_projection():
    """Regression for the remaining summary-path failure: the count-only path
    (include_optional=False, what GET /datasets/graph-summary uses) used to run
    drop_graph + project_entire_graph unconditionally for the component metrics.
    On LLM-extracted graphs with thousands of relationship types — some carrying
    non-identifier characters — the interpolated projection query fails, so the
    summary endpoint could not compute or cache its result at all. The projection
    and component metrics are now gated on include_optional; component fields are
    None on the count-only path (the summary endpoint stores only the counts)."""
    adapter = _metrics_adapter(num_nodes=7, num_edges=11)

    metrics = await adapter.get_graph_metrics(include_optional=False)

    adapter.drop_graph.assert_not_awaited()
    adapter.project_entire_graph.assert_not_awaited()
    assert metrics["num_connected_components"] is None
    assert metrics["sizes_of_connected_components"] is None
    assert metrics["edge_density"] == 0.5


@pytest.mark.asyncio
async def test_graph_metrics_full_path_still_projects():
    adapter = _metrics_adapter(num_nodes=7, num_edges=11)

    metrics = await adapter.get_graph_metrics(include_optional=True)

    adapter.drop_graph.assert_awaited_once()
    adapter.project_entire_graph.assert_awaited_once()
    assert metrics["num_connected_components"] == 1


@pytest.mark.asyncio
async def test_graph_metrics_handles_empty_graph():
    adapter = _metrics_adapter(num_nodes=0, num_edges=0)

    metrics = await adapter.get_graph_metrics(include_optional=False)

    adapter.get_model_independent_graph_data.assert_not_awaited()
    adapter.drop_graph.assert_not_awaited()
    assert metrics["num_nodes"] == 0
    assert metrics["num_edges"] == 0
    assert metrics["mean_degree"] is None
