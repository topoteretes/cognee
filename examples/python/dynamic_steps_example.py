import cognee
import asyncio


from cognee.shared.logging_utils import get_logger, ERROR
from cognee.modules.metrics.operations import get_pipeline_run_metrics
from cognee.modules.engine.models.Entity import Entity
from cognee.api.v1.search import SearchType

job_1 = """
   Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
"""


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
        text_list = [job_1]
        for text in text_list:
            await cognee.add(text)
            print(f"Added text: {text[:35]}...")

    # Step 3: Create knowledge graph
    if enable_steps.get("cognify"):
        pipeline_run = await cognee.cognify()
        print("Knowledge graph created.")

    # Step 4: Calculate descriptive metrics
    if enable_steps.get("graph_metrics"):
        await get_pipeline_run_metrics(pipeline_run, include_optional=True)
        print("Descriptive graph metrics saved to database.")

    # Step 5: Query insights
    if enable_steps.get("retriever"):
        search_results = await cognee.search(
            query_type=SearchType.GRAPH_COMPLETION,
            query_text="What is computer science?",
            node_type=Entity,
            node_name=["computer science"],
        )
        print(search_results)


if __name__ == "__main__":
    logger = get_logger(level=ERROR)

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
