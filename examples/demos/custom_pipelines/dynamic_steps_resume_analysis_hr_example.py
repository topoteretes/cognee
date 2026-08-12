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

    # Step 2: Remember the CVs (ingest + build the knowledge graph)
    if enable_steps.get("remember"):
        text_list = [job_1, job_2, job_3, job_4, job_5]
        for text in text_list:
            print(f"Remembering text: {text[:35]}...")
        await cognee.remember(text_list, self_improvement=False)
        print("Knowledge graph created.")

    # Step 3: Query insights
    if enable_steps.get("retriever"):
        results = await cognee.recall(
            query_type=SearchType.GRAPH_COMPLETION, query_text="Who has experience in design tools?"
        )
        print([result.text for result in results])


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)

    rebuild_kg = True
    retrieve = True
    steps_to_enable = {
        "prune_data": rebuild_kg,
        "prune_system": rebuild_kg,
        "remember": rebuild_kg,
        "retriever": retrieve,
    }

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(steps_to_enable))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
