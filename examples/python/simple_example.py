import cognee
import asyncio
from cognee.api.v1.search import SearchType

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"
# 3. (Optional) To minimize setup effort, set `VECTOR_DB_PROVIDER="lancedb"` in `.env".

async def main():
    # Create a clean slate for cognee -- reset data and system state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # cognee knowledge graph will be created based on this text
    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """

    # Add the text, and make it available for cognify
    await cognee.add(text)

    # Use LLMs and cognee to create knowledge graph
    await cognee.cognify()

    # Query cognee for insights on the added text
    search_results = await cognee.search(
        SearchType.INSIGHTS, query_text='Tell me about NLP'
    )

    # Display search results
    for result_text in search_results:
        print(result_text)

if __name__ == '__main__':
    asyncio.run(main())
