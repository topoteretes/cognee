import asyncio

from common import configure_cognee_for_subprocess

import cognee
from cognee.shared.logging_utils import INFO, setup_logging
from cognee.tests.utils.ci_search_type import completion_or_chunks


async def main():
    configure_cognee_for_subprocess(cognee)

    await cognee.cognify(datasets=["first_cognify_dataset"])

    query_text = (
        "Tell me what is in the context. Additionally write out 'FIRST_COGNIFY' before your answer"
    )
    search_results = await cognee.search(
        query_type=completion_or_chunks(),
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
