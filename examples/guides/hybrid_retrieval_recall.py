import asyncio

import cognee
from cognee import SearchType

QUERY = "What did Alice and Bob work on together?"


async def main():
    await cognee.remember(
        [
            "Alice and Bob were PhD students in Berlin from 2021 to 2024.",
            "Alice and Bob worked on a paper together in 2023.",
            "Alice joined Cognee as a backend engineer in 2025.",
            "Bob joined Cognee as a data scientist in 2026.",
            "Alice and Bob worked together on some sections of the documentation of Cognee in 2026.",
            "New sections of the documentation are available since July 2026.",
        ],
        self_improvement=False,
    )

    context_passage_focused = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.HYBRID_COMPLETION,
        only_context=True,
        retriever_specific_config={
            "chunks_top_k": 5,
            "entities_top_k": 0,
            "facts_top_k": 0,
        },
    )
    print(context_passage_focused)

    context_entity_focused = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.HYBRID_COMPLETION,
        only_context=True,
        retriever_specific_config={
            "chunks_top_k": 0,
            "entities_top_k": 5,
            "max_edges_per_entity": 5,
            "facts_top_k": 0,
        },
    )
    print(context_entity_focused)

    context_fact_focused = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.HYBRID_COMPLETION,
        only_context=True,
        retriever_specific_config={
            "chunks_top_k": 0,
            "entities_top_k": 5,
            "max_edges_per_entity": 0,
            "facts_top_k": 5,
        },
    )
    print(context_fact_focused)

    context_balanced = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.HYBRID_COMPLETION,
        only_context=True,
        retriever_specific_config={
            "chunks_top_k": 2,
            "entities_top_k": 2,
            "max_edges_per_entity": 3,
            "facts_top_k": 2,
        },
    )
    print(context_balanced)

    answer = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.HYBRID_COMPLETION,
        retriever_specific_config={
            "chunks_top_k": 2,
            "entities_top_k": 2,
            "max_edges_per_entity": 3,
            "facts_top_k": 2,
        },
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
