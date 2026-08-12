import asyncio
from pathlib import Path

import cognee
from cognee import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging

DATA_DIR = Path(__file__).parent / "dynamic_steps_resume_analysis_hr_example_data"
job_1 = (DATA_DIR / "cv_1.txt").read_text()
job_2 = (DATA_DIR / "cv_2.txt").read_text()
job_3 = (DATA_DIR / "cv_3.txt").read_text()
job_4 = (DATA_DIR / "cv_4.txt").read_text()
job_5 = (DATA_DIR / "cv_5.txt").read_text()


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
        text_list = [job_1, job_2, job_3, job_4, job_5]
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
            query_type=SearchType.GRAPH_COMPLETION, query_text="Who has experience in design tools?"
        )
        print(search_results)


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
