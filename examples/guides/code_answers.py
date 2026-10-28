"""Ask an indexed code dataset a question and inspect the exact evidence.

Run after add()/cognify() has indexed the repository:
    uv run python examples/guides/code_answers.py my_code_dataset "Da li pauza blokira ručni sync?"

Requires a configured LLM for question planning and the answer. An explicit
--terms skips the planning call. --context previews evidence without an LLM;
use English terms or exact identifiers for that deterministic preview.
This example never ingests, deletes or changes a repository.
"""

import argparse
import asyncio
import json

import cognee
from cognee import SearchType


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("question")
    parser.add_argument("--terms", help="Explicit code search terms, e.g. sync_repositories")
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--session-id")
    args = parser.parse_args()
    results = await cognee.recall(
        args.question,
        query_type=SearchType.HYBRID_COMPLETION,
        datasets=[args.dataset],
        scope=["graph", "code"],
        code_query={"operation": "ask", "question": args.terms} if args.terms else None,
        only_context=args.context,
        session_id=args.session_id,
        include_references=True,
    )
    print(
        json.dumps(
            [entry.model_dump(mode="json") for entry in results], indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
