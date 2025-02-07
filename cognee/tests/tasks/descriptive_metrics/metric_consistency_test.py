from cognee.tests.tasks.descriptive_metrics.networkx_metrics_test import get_networkx_metrics
from cognee.tests.tasks.descriptive_metrics.neo4j_metrics_test import get_neo4j_metrics
import asyncio


async def check_graph_metrics_consistency_across_adapters():
    neo4j_metrics = await get_neo4j_metrics(include_optional=False)
    networkx_metrics = await get_networkx_metrics(include_optional=False)
    assert networkx_metrics == neo4j_metrics


if __name__ == "__main__":
    asyncio.run(check_graph_metrics_consistency_across_adapters())
