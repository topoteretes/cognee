import asyncio
import logging
from typing import Dict

from cognee.modules.retrieval.summaries_retriever import SummariesRetriever
from cognee.tasks.summarization import query_summaries
from examples.python.dynamic_steps_example import main as setup_main


async def compare_summaries_search(query: str = "Who has experience in data science?") -> Dict:
    """Compares old and new summaries search implementations."""
    # Get results from both implementations
    old_results = await query_summaries(query)
    retriever = SummariesRetriever()
    new_results = await retriever.get_completion(query)

    # Compare results
    lengths_match = len(old_results) == len(new_results)

    if lengths_match:
        print("Results length match")
        matches = [old == new for old, new in zip(old_results, new_results)]
        if all(matches):
            print("All entries match")
        else:
            print(f"Differences found at indices: {[i for i, m in enumerate(matches) if not m]}")
    else:
        matches = []
        print(f"Results length mismatch: {len(old_results)} vs {len(new_results)}")

    return {
        "old_results": old_results,
        "new_results": new_results,
        "lengths_match": lengths_match,
        "element_matches": matches,
    }


async def main():
    """Main function to run comparisons."""
    # Setup cognee using the example setup
    steps_to_enable = {
        "prune_data": True,
        "prune_system": True,
        "add_text": True,
        "cognify": True,
        "retriever": False,
    }
    await setup_main(steps_to_enable)

    # Run summaries comparison
    await compare_summaries_search()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main())
