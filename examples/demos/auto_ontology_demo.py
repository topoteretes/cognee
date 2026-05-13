import asyncio
import sys
from pathlib import Path
from pprint import pprint

import cognee
from cognee.api.v1.search import SearchType
from cognee.shared.logging_utils import setup_logging, ERROR

# Change this to swap the dataset ingested by the workflow below.
# Options come from _comparison_datasets.py:
#   "cvs", "recipes", "scientific_abstracts", "historical_events"
DATASET_KEY = "recipes"

sys.path.insert(0, str(Path(__file__).parent))

from _comparison_datasets import DATASETS  # noqa: E402


async def main(enable_steps):
    # Step 1: Reset data and system state
    if enable_steps.get("prune_data"):
        await cognee.prune.prune_data()
        print("Data pruned.")

    if enable_steps.get("prune_system"):
        await cognee.prune.prune_system(metadata=True)
        print("System pruned.")

    # Step 2: Add text
    if enable_steps.get("add_text"):
        text_list = DATASETS[DATASET_KEY]["texts"]
        for text in text_list:
            await cognee.add(text)
            print(f"Added text: {text[:35]}...")

    # Step 3: Create knowledge graph
    if enable_steps.get("cognify"):
        await cognee.cognify()
        print("Knowledge graph created.")

    # Step 4: Query insights
    if enable_steps.get("retriever"):
        search_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION,
            query_text="List all contactinfos in the dataset with the corresponding candidate names.",
            top_k=50,
        )
        pprint(search_results)


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)

    rebuild_kg = True
    retrieve = True
    steps_to_enable = {
        "prune_data": rebuild_kg,
        "prune_system": rebuild_kg,
        "add_text": rebuild_kg,
        "cognify": rebuild_kg,
        "graph_metrics": rebuild_kg,
        "retriever": retrieve,
    }

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(steps_to_enable))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
