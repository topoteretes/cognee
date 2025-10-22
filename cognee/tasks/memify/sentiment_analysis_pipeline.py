from cognee import memify
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.sentiment_analysis.sentiment_analysis import sentiment_analysis_task
from cognee.tasks.sentiment_analysis.enrichment import enrichment_task 
import asyncio
async def main():
 
    extraction_task = Task(sentiment_analysis_task)  
    enrichment_task = Task(enrichment_task)  

    await memify(
        extraction_tasks=[extraction_task],
        enrichment_tasks=[enrichment_task],
    )

if __name__ == "__main__":
    asyncio.run(main())
