import asyncio
import os
import pathlib

import cognee
from cognee import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"
#
# Optional richer image ingestion (both default off, see `.env.template`):
#   IMAGE_EXTRACTION_ENABLED — extraction-oriented transcription prompt (entities/values/relations)
#   IMAGE_OCR_ENABLED         — append local OCR text; needs pip install "cognee[rapidocr]"


async def main():
    # Create a clean slate for cognee -- reset data and system state
    await cognee.forget(everything=True)

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

    # Remember the files and create knowledge graph memory
    await cognee.remember([mp3_file_path, png_file_path], self_improvement=False)

    # Query cognee for summaries of the data in the multimedia files
    search_results = await cognee.recall(
        query_type=SearchType.SUMMARIES,
        query_text="What is in the multimedia files?",
    )

    # Display search results
    for result_text in search_results:
        print(result_text)


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
