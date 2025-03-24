from gitdb.fun import chunk_size

import cognee
import asyncio
import logging
import os

from cognee.api.v1.search import SearchType
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.shared.utils import setup_logging


async def main():
    # Step 1: Reset data and system state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    scientific_papers_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scientific_papers/"
    )

    # Step 2: Add text
    await cognee.add(scientific_papers_dir)

    # Step 3: Create knowledge graph

    ontology_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ontology_input_example/enriched_medical_ontology_with_classes.owl",
    )

    pipeline_run = await cognee.cognify(ontology_file_path=ontology_path)
    print("Knowledge with ontology created.")

    # Step 4: Calculate descriptive metrics
    await cognee.get_pipeline_run_metrics(pipeline_run, include_optional=True)
    print("Descriptive graph metrics saved to database.")

    # Step 5: Query insights
    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What are the main benefits of chocolate?",
    )
    print(search_results)

    await visualize_graph()


if __name__ == "__main__":
    setup_logging(logging.INFO)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
