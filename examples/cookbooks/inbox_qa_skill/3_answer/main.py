"""Step 3: answer a question from everything the earlier steps remembered.

Run alone: uv run python examples/cookbooks/inbox_qa_skill/3_answer/main.py "Your question"
"""

import asyncio
import sys

import cognee
from cognee.shared.logging_utils import ERROR, setup_logging

DATASET = "inbox_qa_skill"  # the same in every step
DEFAULT_QUESTION = "What does the newest email ask of me, and what do my notes say about it?"


async def main(question: str) -> None:
    results = await cognee.recall(question, datasets=[DATASET])
    answer = str(results[0].text) if results else "Nothing found. Run the ingest steps first."
    print(f"[answer] Q: {question}\n[answer] A: {answer}")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main(" ".join(sys.argv[1:]) or DEFAULT_QUESTION))
