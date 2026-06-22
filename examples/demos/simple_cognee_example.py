import asyncio

import cognee
from cognee import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"


async def main():
    # Start clean, then remember knowledge with the v1.0 memory API.
    await cognee.forget(everything=True)
    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """

    await cognee.remember(text, self_improvement=False)

    query_text = "Tell me about NLP"
    print(f"Searching cognee for insights with query: '{query_text}'")
    search_results = await cognee.recall(
        query_type=SearchType.GRAPH_COMPLETION, query_text=query_text
    )

    for result_text in search_results:
        print(result_text)


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
