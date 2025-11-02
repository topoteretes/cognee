import asyncio
import cognee
from cognee import memify
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.sentiment_analysis.sentiment_analysis import run_sentiment_analysis
from cognee.tasks.sentiment_analysis.enrichment import enrichment_task
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def main():
    # Step 1: Reset Cognee data
    logger.info('Resetting Cognee data...')
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    logger.info("Data reset complete.\n")

    # Step 2: Add sample content
    text = "Cognee turns documents into AI memory."
    await cognee.add(text)

    # Step 3: Build the knowledge graph
    logger.info("Cognifying the content...")
    await cognee.cognify()
    logger.info("Cognify complete.\n")

    # Step 4: Define queries to test
    queries = [
        "What does Cognee do?",
        "How does Cognee store data?",
        "Are you even listening to what I am asking?",
        "This is good, this was what I was asking for"
    ]

    all_results = {}

    for q in queries:
        results = await cognee.search(
            query_text=q,
            save_interaction=True,  # Save interactions for analysis
        )
        all_results[q] = results

    # Step 5: Create your extraction and enrichment tasks
    extraction_task = Task(run_sentiment_analysis)
    enrichment = Task(enrichment_task)

    # Step 6: Run memify pipeline — this executes both tasks
    logger.info("Running sentiment analysis pipeline via memify...")
    sentiment_data_points = await memify(
        extraction_tasks=[extraction_task],
        enrichment_tasks=[enrichment],
    )
    logger.info("Memify pipeline complete.\n")
    logger.info(sentiment_data_points)


if __name__ == '__main__':
    asyncio.run(main())
