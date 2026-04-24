import asyncio
import os
import pathlib

import cognee
from cognee.api.v1.search import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

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
        pathlib.Path(__file__).parent,
        "data/text_to_speech.mp3",
    )
    png_file_path = os.path.join(
        pathlib.Path(__file__).parent,
        "data/example.png",
    )

    # Add the files, and make it available for cognify
    await cognee.add([mp3_file_path, png_file_path])

    # Use LLMs and cognee to create knowledge graph
    await cognee.cognify()

    # Query cognee for summaries of the data in the multimedia files
    search_results = await cognee.search(
        query_type=SearchType.SUMMARIES,
        query_text="What is in the multimedia files?",
    )

    # Display search results
    print(search_results)


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
