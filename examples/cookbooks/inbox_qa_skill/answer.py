"""Step 3: answer a question from everything the earlier steps remembered.

Run alone: uv run python examples/cookbooks/inbox_qa_skill/answer.py "Your question"
"""

import asyncio
import sys

from common import DATASET

import cognee

DEFAULT_QUESTION = "What does the newest email ask of me, and what do my notes say about it?"


async def run(question: str = DEFAULT_QUESTION) -> str:
    results = await cognee.recall(question, datasets=[DATASET])
    answer = str(results[0].text) if results else "Nothing found. Run the ingest steps first."
    print(f"[answer] Q: {question}\n[answer] A: {answer}")
    return answer


if __name__ == "__main__":
    asyncio.run(run(" ".join(sys.argv[1:]) or DEFAULT_QUESTION))
