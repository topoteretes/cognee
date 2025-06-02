import cognee
import asyncio
from cognee.shared.logging_utils import get_logger, ERROR
from cognee.modules.metrics.operations import get_pipeline_run_metrics

from cognee.api.v1.search import SearchType

text_1_to_cognify = """Germany is located next to the Netherlands."""

text_2_to_cognify = (
    """Skoda is a czech car manufacturer which is the part of the Volkswagen group"""
)

text_3_not_to_cognify = "This text should not be cognified with cognee which is an AI memory engine"


async def main():
    await cognee.prune.prune_data()

    await cognee.prune.prune_system(metadata=True)

    text_1 = await cognee.add(text_1_to_cognify)

    await cognee.cognify(datapoints=text_1.packets)

    text_2 = await cognee.add(text_2_to_cognify)

    await cognee.cognify(datapoints=text_2.packets)

    await cognee.add(text_3_not_to_cognify)

    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text="What is cognee?"
    )
    print(search_results)


if __name__ == "__main__":
    logger = get_logger(level=ERROR)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
