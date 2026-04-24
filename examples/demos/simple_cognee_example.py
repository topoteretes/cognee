import asyncio

import cognee
from cognee.api.v1.search import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"


async def main():
    # Prune data/system, add conversation, cognify.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """

    await cognee.add(text)

    await cognee.cognify()

    query_text = "Tell me about NLP"
    print(f"Searching cognee for insights with query: '{query_text}'")
    # Query cognee for insights on the added text
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=query_text
    )

    print(search_results)


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
