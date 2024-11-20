import asyncio

import cognee
from cognee.api.v1.search import SearchType

# Prerequisites:
# 1. Copy `.env.template` and rename it to `.env`.
# 2. Add your OpenAI API key to the `.env` file in the `LLM_API_KEY` field:
#    LLM_API_KEY = "your_key_here"


async def main():
    # Create a clean slate for cognee -- reset data and system state
    print("Resetting cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("Data reset complete.\n")

    # cognee knowledge graph will be created based on this text
    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """
    
    print("Adding text to cognee:")
    print(text.strip())  
    # Add the text, and make it available for cognify
    await cognee.add(text)
    print("Text added successfully.\n")

    
    print("Running cognify to create knowledge graph...")
    # Use LLMs and cognee to create knowledge graph
    await cognee.cognify()
    print("Cognify process complete.\n")

    
    query_text = 'Tell me about NLP'
    print(f"Searching cognee for insights with query: '{query_text}'")
    # Query cognee for insights on the added text
    search_results = await cognee.search(
        SearchType.INSIGHTS, query_text=query_text
    )
    
    print("Search results:")
    # Display results
    for result_text in search_results:
        print(result_text)


if __name__ == '__main__':
    asyncio.run(main())
