from cognee.tests.tasks.descriptive_metrics.metrics_test_utils import create_disconnected_test_graph
from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
from cognee.infrastructure.databases.graph import get_graph_engine
import cognee
import asyncio
import pytest


async def get_neo4j_metrics(include_optional=True):
    create_graph_engine.cache_clear()
    cognee.config.set_graph_database_provider("neo4j")
    graph_engine = await get_graph_engine()
    await graph_engine.delete_graph()
    await create_disconnected_test_graph()
    neo4j_graph_metrics = await graph_engine.get_graph_metrics(include_optional=include_optional)
    return neo4j_graph_metrics


@pytest.mark.asyncio
async def test_neo4j_metrics():
    neo4j_metrics = await get_neo4j_metrics(include_optional=True)
    assert neo4j_metrics["num_nodes"] == 9, f"Expected 9 nodes, got {neo4j_metrics['num_nodes']}"
    assert neo4j_metrics["num_edges"] == 9, f"Expected 9 edges, got {neo4j_metrics['num_edges']}"
    assert neo4j_metrics["mean_degree"] == 2, (
        f"Expected mean degree is 2, got {neo4j_metrics['mean_degree']}"
    )
    assert neo4j_metrics["edge_density"] == 0.125, (
        f"Expected edge density is 0.125, got {neo4j_metrics['edge_density']}"
    )
    assert neo4j_metrics["num_connected_components"] == 2, (
        f"Expected 2 connected components, got {neo4j_metrics['num_connected_components']}"
    )
    assert neo4j_metrics["sizes_of_connected_components"] == [5, 4], (
        f"Expected connected components of size [5, 4], got {neo4j_metrics['sizes_of_connected_components']}"
    )
    assert neo4j_metrics["num_selfloops"] == 1, (
        f"Expected 1 self-loop, got {neo4j_metrics['num_selfloops']}"
    )


if __name__ == "__main__":
    asyncio.run(test_neo4j_metrics())
