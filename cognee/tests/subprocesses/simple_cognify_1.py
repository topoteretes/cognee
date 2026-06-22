import asyncio

import cognee
from cognee.api.v1.search import SearchType
from cognee.shared.logging_utils import INFO, setup_logging
from common import configure_cognee_for_subprocess


async def main():
    configure_cognee_for_subprocess(cognee)

    await cognee.cognify(datasets=["first_cognify_dataset"])

    query_text = (
        "Tell me what is in the context. Additionally write out 'FIRST_COGNIFY' before your answer"
    )
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=query_text,
        datasets=["first_cognify_dataset"],
    )

    print("Search results:")
    for result_text in search_results:
        print(result_text)


if __name__ == "__main__":
    setup_logging(log_level=INFO)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
