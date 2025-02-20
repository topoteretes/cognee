from cognee.tests.tasks.descriptive_metrics.metrics_test_utils import get_metrics

import asyncio


async def check_graph_metrics_consistency_across_adapters(include_optional=False):
    neo4j_metrics = await get_metrics(provider="neo4j", include_optional=include_optional)
    networkx_metrics = await get_metrics(provider="networkx", include_optional=include_optional)

    diff_keys = set(neo4j_metrics.keys()).symmetric_difference(set(networkx_metrics.keys()))
    if diff_keys:
        raise AssertionError(f"Metrics dictionaries have different keys: {diff_keys}")

    for key, neo4j_value in neo4j_metrics.items():
        assert networkx_metrics[key] == neo4j_value, (
            f"Difference in '{key}': got {neo4j_value} with neo4j and {networkx_metrics[key]} with networkx"
        )


if __name__ == "__main__":
    asyncio.run(check_graph_metrics_consistency_across_adapters(include_optional=True))
    asyncio.run(check_graph_metrics_consistency_across_adapters(include_optional=False))
