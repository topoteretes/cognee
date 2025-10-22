import asyncio


async def main():
    from cognee.tests.tasks.descriptive_metrics.metrics_test_utils import assert_metrics

    await assert_metrics(provider="neo4j", include_optional=False)
    await assert_metrics(provider="neo4j", include_optional=True)


if __name__ == "__main__":
    asyncio.run(main())
