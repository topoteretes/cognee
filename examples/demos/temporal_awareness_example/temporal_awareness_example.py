import asyncio
from pathlib import Path

import cognee
from cognee import SearchType
from cognee.shared.logging_utils import INFO, setup_logging

data_dir = Path(__file__).resolve().parent / "data"
try:
    biography_1 = (data_dir / "biography_1.txt").read_text(encoding="utf-8")
    biography_2 = (data_dir / "biography_2.txt").read_text(encoding="utf-8")
except (FileNotFoundError, OSError) as exc:
    raise RuntimeError(f"Missing demo data file under: {data_dir}") from exc


async def main():
    # Step 1: Reset data and system state
    await cognee.forget(everything=True)

    # Step 2: Remember text and create temporal knowledge graph memory
    await cognee.remember(
        [biography_1, biography_2],
        temporal_cognify=True,
        self_improvement=False,
    )

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
        search_results = await cognee.recall(
            query_type=SearchType.TEMPORAL,
            query_text=query_text,
            top_k=15,
        )
        print(f"Query: {query_text}")
        print(f"Results: {search_results}\n")


if __name__ == "__main__":
    logger = setup_logging(log_level=INFO)
    asyncio.run(main())
