from cognee.tests.tasks.descriptive_metrics.metrics_test_utils import get_metrics

import asyncio


async def check_graph_metrics_consistency_across_adapters(include_optional=False):
    # NetworkX has been moved to the community package
    # This test now only uses neo4j and kuzu for consistency checks
    neo4j_metrics = await get_metrics(provider="neo4j", include_optional=include_optional)
    kuzu_metrics = await get_metrics(provider="kuzu", include_optional=include_optional)

    diff_keys = set(neo4j_metrics.keys()).symmetric_difference(set(kuzu_metrics.keys()))
    if diff_keys:
        raise AssertionError(f"Metrics dictionaries have different keys: {diff_keys}")

    for key, neo4j_value in neo4j_metrics.items():
        assert kuzu_metrics[key] == neo4j_value, (
            f"Difference in '{key}': got {neo4j_value} with neo4j and {kuzu_metrics[key]} with kuzu"
        )


if __name__ == "__main__":
    asyncio.run(check_graph_metrics_consistency_across_adapters(include_optional=True))
    asyncio.run(check_graph_metrics_consistency_across_adapters(include_optional=False))
