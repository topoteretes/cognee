import asyncio
from pprint import pprint

import cognee
from cognee.api.v1.search import SearchType
from cognee.shared.logging_utils import setup_logging, INFO

text_1 = "Cognee is an AI memory platform that turns raw data into knowledge graphs for agents."
text_2 = "Redis can be used as a cache vector store for fast semantic search over embeddings."
text_3 = "The cache triplet retriever reads from the cache collection instead of the main vector DB."


async def main(knowledge_graph_creation: bool, evaluation: bool):
    """Run the Redis cache evaluation pipeline.

    - knowledge_graph_creation: if True, prunes data and system, adds text, and cognifies (builds KG).
    - evaluation: if True, runs the retriever search and prints results.
    """
    if knowledge_graph_creation:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        text_list = [text_1, text_2, text_3]
        for text in text_list:
            await cognee.add(text)
            print(f"Added text: {text[:35]}...")
        await cognee.cognify()
        print("Knowledge graph created.")

    if evaluation:
        search_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION, query_text="What is Cognee and how does the cache retriever work?"
        )
        pprint(search_results)


if __name__ == "__main__":
    logger = setup_logging(log_level=INFO)

    knowledge_graph_creation = True
    evaluation = True

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(knowledge_graph_creation, evaluation))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
