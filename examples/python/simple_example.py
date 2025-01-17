import asyncio
import cognee
import logging
from cognee.api.v1.search import SearchType
from cognee.shared.utils import setup_logging

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

    print("Running cognify to create knowledge graph...\n")
    print("Cognify process steps:")
    print("1. Classifying the document: Determining the type and category of the input text.")
    print(
        "2. Checking permissions: Ensuring the user has the necessary rights to process the text."
    )
    print(
        "3. Extracting text chunks: Breaking down the text into sentences or phrases for analysis."
    )
    print("4. Adding data points: Storing the extracted chunks for processing.")
    print(
        "5. Generating knowledge graph: Extracting entities and relationships to form a knowledge graph."
    )
    print("6. Summarizing text: Creating concise summaries of the content for quick insights.\n")

    # Use LLMs and cognee to create knowledge graph
    await cognee.cognify()
    print("Cognify process complete.\n")

    query_text = "Tell me about NLP"
    print(f"Searching cognee for insights with query: '{query_text}'")
    # Query cognee for insights on the added text
    search_results = await cognee.search(SearchType.INSIGHTS, query_text=query_text)

    print("Search results:")
    # Display results
    for result_text in search_results:
        print(result_text)

    # Example output:
    # ({'id': UUID('bc338a39-64d6-549a-acec-da60846dd90d'), 'updated_at': datetime.datetime(2024, 11, 21, 12, 23, 1, 211808, tzinfo=datetime.timezone.utc), 'name': 'natural language processing', 'description': 'An interdisciplinary subfield of computer science and information retrieval.'}, {'relationship_name': 'is_a_subfield_of', 'source_node_id': UUID('bc338a39-64d6-549a-acec-da60846dd90d'), 'target_node_id': UUID('6218dbab-eb6a-5759-a864-b3419755ffe0'), 'updated_at': datetime.datetime(2024, 11, 21, 12, 23, 15, 473137, tzinfo=datetime.timezone.utc)}, {'id': UUID('6218dbab-eb6a-5759-a864-b3419755ffe0'), 'updated_at': datetime.datetime(2024, 11, 21, 12, 23, 1, 211808, tzinfo=datetime.timezone.utc), 'name': 'computer science', 'description': 'The study of computation and information processing.'})
    # (...)
    # It represents nodes and relationships in the knowledge graph:
    # - The first element is the source node (e.g., 'natural language processing').
    # - The second element is the relationship between nodes (e.g., 'is_a_subfield_of').
    # - The third element is the target node (e.g., 'computer science').


if __name__ == "__main__":
    setup_logging(logging.ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
