"""Ask the company brain a question that may need several sources to answer.

The answer comes from the whole graph, across every source ingest.py remembered, so one
answer can join a person from the database with their ticket and a fix from the docs.

Run alone: uv run python examples/cookbooks/company_brain/company_qa/scripts/ask.py "question"
"""

import argparse
import asyncio
import os

# One local store for these scripts, the API server and MCP (see README.md).
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

import cognee
from cognee.shared.logging_utils import ERROR, setup_logging

DATASET = "company_brain"  # the same in every script


async def ask(question: str) -> None:
    results = await cognee.recall(question, datasets=[DATASET])
    answer = str(results[0].text) if results else "Nothing found. Run ingest.py first."
    print(f"[ask] Q: {question}\n[ask] A: {answer}")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="+", help="what to ask")
    asyncio.run(ask(" ".join(parser.parse_args().question)))
