import os
import asyncio
import pathlib
import logging

import cognee
from cognee.api.v1.search import SearchType
from cognee.shared.utils import setup_logging

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"


async def main():
    # Create a clean slate for cognee -- reset data and system state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # cognee knowledge graph will be created based on the text
    # and description of these files
    mp3_file_path = os.path.join(
        pathlib.Path(__file__).parent.parent.parent,
        ".data/multimedia/text_to_speech.mp3",
    )
    png_file_path = os.path.join(
        pathlib.Path(__file__).parent.parent.parent,
        ".data/multimedia/example.png",
    )

    # Add the files, and make it available for cognify
    await cognee.add([mp3_file_path, png_file_path])

    # Use LLMs and cognee to create knowledge graph
    await cognee.cognify()

    # Query cognee for summaries of the data in the multimedia files
    search_results = await cognee.search(
        SearchType.SUMMARIES,
        query_text="What is in the multimedia files?",
    )

    # Display search results
    for result_text in search_results:
        print(result_text)


if __name__ == "__main__":
    setup_logging(logging.ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
