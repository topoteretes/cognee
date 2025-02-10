from cognee.tests.tasks.descriptive_metrics.metrics_test_utils import create_disconnected_test_graph
from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
from cognee.infrastructure.databases.graph import get_graph_engine
import cognee
import asyncio


async def get_networkx_metrics(include_optional=True):
    create_graph_engine.cache_clear()
    cognee.config.set_graph_database_provider("networkx")
    graph_engine = await get_graph_engine()
    await graph_engine.delete_graph()
    await create_disconnected_test_graph()
    networkx_graph_metrics = await graph_engine.get_graph_metrics(include_optional=include_optional)
    return networkx_graph_metrics


async def assert_networkx_metrics():
    networkx_metrics = await get_networkx_metrics(include_optional=True)
    assert networkx_metrics["num_nodes"] == 9, (
        f"Expected 9 nodes, got {networkx_metrics['num_nodes']}"
    )
    assert networkx_metrics["num_edges"] == 9, (
        f"Expected 9 edges, got {networkx_metrics['num_edges']}"
    )
    assert networkx_metrics["mean_degree"] == 2, (
        f"Expected mean degree is 2, got {networkx_metrics['mean_degree']}"
    )
    assert networkx_metrics["edge_density"] == 0.125, (
        f"Expected edge density is 0.125, got {networkx_metrics['edge_density']}"
    )
    assert networkx_metrics["num_connected_components"] == 2, (
        f"Expected 2 connected components, got {networkx_metrics['num_connected_components']}"
    )
    assert networkx_metrics["sizes_of_connected_components"] == [5, 4], (
        f"Expected connected components of size [5, 4], got {networkx_metrics['sizes_of_connected_components']}"
    )
    assert networkx_metrics["num_selfloops"] == 1, (
        f"Expected 1 self-loop, got {networkx_metrics['num_selfloops']}"
    )
    assert networkx_metrics["diameter"] is None, (
        f"Diameter should be None for disconnected graphs, got {networkx_metrics['diameter']}"
    )
    assert networkx_metrics["avg_shortest_path_length"] is None, (
        f"Average shortest path should be None for disconnected graphs, got {networkx_metrics['avg_shortest_path_length']}"
    )
    assert networkx_metrics["avg_clustering"] == 0, (
        f"Expected 0 average clustering, got {networkx_metrics['avg_clustering']}"
    )


if __name__ == "__main__":
    asyncio.run(assert_networkx_metrics())
