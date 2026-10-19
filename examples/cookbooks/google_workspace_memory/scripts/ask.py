"""Ask one question across your calendar, email and Drive.

The default question prepares your coming client meetings: for each client you meet in the
next week, which orders are late or at risk, why, and what to tell them. Today's date goes
into the question, so "next week" and "late" mean what they should. Pass --ask for your own,
and --sample to ask the sample's dataset `google_workspace_memory_sample`.

Run alone: uv run python examples/cookbooks/google_workspace_memory/scripts/ask.py [--ask "question"] [--sample]
"""

import argparse
import asyncio
import os
from datetime import datetime

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.modules.engine.operations.setup import setup
from cognee.modules.search.types import SearchType

DATASET = "google_workspace_memory"  # the same in every script
SAMPLE_DATASET = "google_workspace_memory_sample"  # the sample never mixes with your data
DEFAULT_QUESTION = (
    "Which client meetings do I have in the next seven days? For each of those clients, "
    "which orders are late or at risk of missing the promised delivery date, why, and "
    "what should I tell them in the meeting?"
)


async def ask(question: str | None = None, sample: bool = False) -> None:
    question = question or DEFAULT_QUESTION
    dataset = SAMPLE_DATASET if sample else DATASET
    await setup()  # a fresh install has no database yet
    if dataset not in {dataset.name for dataset in await cognee.datasets.list_datasets()}:
        raise SystemExit(
            f"[ask] The dataset {dataset} does not exist yet. Run the ingest scripts first."
        )
    print(f"[ask] {question}")
    answer = await cognee.recall(
        f"Today is {datetime.now().astimezone().date().isoformat()}. {question}",
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=[dataset],
    )
    if not answer:
        raise SystemExit("[ask] Nothing found. Run the ingest scripts first.")
    print(f"[ask] Answer:\n\n{answer[0].text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ask", help="your own question")
    parser.add_argument("--sample", action="store_true", help="ask the sample's dataset")
    args = parser.parse_args()
    asyncio.run(ask(args.ask, args.sample))
