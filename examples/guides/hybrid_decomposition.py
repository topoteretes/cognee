"""Answer a multi-part question with context-primed hybrid decomposition.

Requires a configured LLM and an already ingested dataset.
Run: uv run python examples/guides/hybrid_decomposition.py --dataset my_dataset
"""

import argparse
import asyncio

import cognee
from cognee import SearchType


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--question", default="What did Acme order, and when is delivery due?")
    args = parser.parse_args()
    result = await cognee.recall(
        query_text=args.question,
        datasets=[args.dataset],
        query_type=SearchType.HYBRID_COMPLETION_DECOMPOSITION,
        include_references=True,
        retriever_specific_config={
            "max_subqueries": 7,
            "include_external_metadata": True,
            "external_metadata_keys": ["source_id", "title"],
            # Optional: "decomposition_system_prompt": "Your use-case instructions..."
        },
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
