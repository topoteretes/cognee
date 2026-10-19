"""Ask one question across your calendar, email and Drive.

The default question prepares your coming client meetings: for each client you meet in the
next week, which orders are late or at risk, why, and what to tell them. Today's date goes
into the question, so "next week" and "late" mean what they should. Pass --ask for your own.

Run alone: uv run python examples/cookbooks/google_workspace_memory/scripts/ask.py [--ask "question"]
"""

import argparse
import asyncio
import os
from datetime import datetime

os.environ.setdefault("LOG_LEVEL", "ERROR")  # quiet cognee's logs; set before importing it

import cognee
from cognee.modules.search.types import SearchType

DATASET = "google_workspace_memory"  # the same in every script
DEFAULT_QUESTION = (
    "Which client meetings do I have in the next seven days? For each of those clients, "
    "which orders are late or at risk of missing the promised delivery date, why, and "
    "what should I tell them in the meeting?"
)


async def ask(question: str | None = None) -> None:
    question = question or DEFAULT_QUESTION
    print(f"[ask] {question}")
    answer = await cognee.recall(
        f"Today is {datetime.now().astimezone().date().isoformat()}. {question}",
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=[DATASET],
    )
    if not answer:
        raise SystemExit("[ask] Nothing found. Run the ingest scripts first.")
    print(f"[ask] Answer:\n\n{answer[0].text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ask", help="your own question")
    asyncio.run(ask(parser.parse_args().ask))
