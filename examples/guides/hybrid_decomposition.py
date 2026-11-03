"""Inspect subquestions or retrieve with context-primed hybrid decomposition.

Requires a configured LLM. Retrieval also requires an already ingested dataset.
Run: uv run python examples/guides/hybrid_decomposition.py --dataset my_dataset
Plan only: uv run python examples/guides/hybrid_decomposition.py --plan-only
"""

import argparse
import asyncio

import cognee
from cognee import SearchType
from cognee.modules.retrieval.hybrid_decomposition_retriever import HybridDecompositionRetriever


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--question", default="What did Acme order, and when is delivery due?")
    args = parser.parse_args()
    if args.plan_only:
        retriever = HybridDecompositionRetriever(max_subqueries=7)
        # Optional caller-supplied context. This method performs no retrieval.
        subqueries = await retriever.decompose_query(args.question, context="")
        for query in subqueries:
            print(query)
        return
    if not args.dataset:
        parser.error("--dataset is required for retrieval")
    result = await cognee.recall(
        query_text=args.question,
        datasets=[args.dataset],
        query_type=SearchType.HYBRID_COMPLETION_DECOMPOSITION,
        only_context=True,  # Retrieves and decomposes, but skips the final answer.
        include_references=True,
        retriever_specific_config={
            "max_subqueries": 7,
            "chunks_top_k": 10,
            "entities_top_k": 10,
            "facts_top_k": 10,
            "include_external_metadata": True,
            "external_metadata_keys": ["source_id", "title"],
            # Optional: "decomposition_system_prompt": "Your use-case instructions..."
        },
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
