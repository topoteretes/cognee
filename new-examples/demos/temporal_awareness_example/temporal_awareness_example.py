import asyncio
import os
import cognee
from cognee.shared.logging_utils import setup_logging, INFO
from cognee.api.v1.search import SearchType
from pathlib import Path

with open(
    os.path.join(Path(__file__).resolve().parent, "data/biography_1.txt"), "r", encoding="utf-8"
) as f:
    biography_1 = f.read()

with open(
    os.path.join(Path(__file__).resolve().parent, "data/biography_2.txt"), "r", encoding="utf-8"
) as f:
    biography_2 = f.read()


async def main():
    # Step 1: Reset data and system state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Step 2: Add text and create knwoledge graph
    await cognee.add([biography_1, biography_2])
    await cognee.cognify(temporal_cognify=True)

    queries = [
        "What happened before 1980?",
        "What happened after 2010?",
        "What happened between 2000 and 2006?",
        "What happened between 1903 and 1995, I am interested in the Selected Works of Arnulf Øverland Ole Peter Arnulf Øverland?",
        "Who is Attaphol Buspakom Attaphol Buspakom?",
        "Who was Arnulf Øverland?",
    ]

    # Step 3: Query insights
    for query_text in queries:
        search_results = await cognee.search(
            query_type=SearchType.TEMPORAL,
            query_text=query_text,
            top_k=15,
        )
        print(f"Query: {query_text}")
        print(f"Results: {search_results}\n")


if __name__ == "__main__":
    logger = setup_logging(log_level=INFO)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
