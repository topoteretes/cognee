"""Answer a multi-part question with context-primed hybrid decomposition.

A three-part question (what Acme ordered, when it is delivered, where the supplier is based) is
recalled with only_context=True twice under the same per-lane limits of one passage per search:
once with HYBRID_COMPLETION, whose single search returns one passage, and once with
HYBRID_COMPLETION_DECOMPOSITION, which splits the question into focused subqueries after a first
retrieval pass and merges the passages each one returns. The printed contexts show which parts
each covered. The question is then answered with decomposition, with the passages' source_id and
title metadata in the context and an Evidence block citing them.

Requires: LLM_API_KEY.
Run: uv run python examples/guides/hybrid_decomposition.py
"""

import asyncio

import cognee
from cognee import SearchType
from cognee.tasks.ingestion.data_item import DataItem

QUERY = "What did Acme order from Northwind, when is delivery due, and where is Northwind based?"


def document(source_id, title, text):
    return DataItem(text, external_metadata={"source_id": source_id, "title": title})


DOCUMENTS = [
    document("po-1042", "Acme order", "Acme ordered 40 pumps from Northwind on March 3."),
    document("mail-77", "Acme delivery", "Northwind will deliver Acme's pumps on April 15."),
    document("po-1043", "Globex order", "Globex ordered 12 fans from Northwind on March 5."),
    document("mail-81", "Globex delivery", "Northwind will deliver Globex's fans on March 28."),
    document("po-1044", "Initech order", "Initech ordered 300 valves from Contoso on March 9."),
    document("memo-15", "Initech approval", "Sam Lee approved Initech's valve order."),
    document("note-3", "Northwind profile", "Northwind is headquartered in Rotterdam."),
    document("note-4", "Contoso profile", "Contoso is headquartered in Lyon."),
]

# Passages only, one per search, so the printed contexts are easy to compare: a single search
# returns one passage, a decomposed one up to one per subquery.
LIMITS = {"chunks_top_k": 1, "entities_top_k": 0, "facts_top_k": 0}


async def main():
    await cognee.remember(DOCUMENTS, self_improvement=False)

    context_single_search = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.HYBRID_COMPLETION,
        only_context=True,
        retriever_specific_config=LIMITS,
    )
    print(context_single_search)

    context_decomposed = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.HYBRID_COMPLETION_DECOMPOSITION,
        only_context=True,
        retriever_specific_config={**LIMITS, "max_subqueries": 3},
    )
    print(context_decomposed)

    answer = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.HYBRID_COMPLETION_DECOMPOSITION,
        include_references=True,
        retriever_specific_config={
            **LIMITS,
            "max_subqueries": 3,
            "include_external_metadata": True,
            "external_metadata_keys": ["source_id", "title"],
            # Optional: "decomposition_system_prompt": "Your use-case instructions..."
        },
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
