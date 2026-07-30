"""Global context index: a world-summary prelude for retrieval.

``improve(..., build_global_context_index=True)`` builds a compact summary of the whole
dataset — a world summary plus the areas it covers. Retrieval can then prepend that
summary to the context it assembles, which helps when an answer needs the shape of the
dataset rather than the few chunks nearest the query.

This guide asks for the *context* rather than an answer (``only_context=True``) so you
can see the prelude appear when the flag is on.

For a longer run with a multi-day conversation and several queries, see
``advanced_guides/global_context_index_smoke_demo.py``.
"""

import asyncio

import cognee
from cognee import SearchType

DATASET = "global_context_index_guide"

CONVERSATION = [
    "[2026-04-02 09:14] User: Could you suggest three dates for our team syncs this month?",
    "[2026-04-02 09:18] Assistant: April 5, 12, and 19 look clear — 10:00 for the first two.",
    "[2026-04-02 09:20] User: That works. Please put all three on the calendar.",
    "[2026-04-07 11:09] User: Cancel Meeting 3, and move Meeting 2 from April 12 to April 9.",
    "[2026-04-07 11:10] Assistant: Done. Meeting 3 is cancelled; Meeting 2 is now April 9 at 10:00.",
]

QUERY = "When is the second meeting?"

RETRIEVER_CONFIG = {"include_global_context_index": True, "global_context_index_top_k": 3}


async def context_for(include_global_context: bool) -> str:
    config = dict(RETRIEVER_CONFIG, include_global_context_index=include_global_context)
    results = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        only_context=True,
        retriever_specific_config=config,
    )
    if not results:
        return "(empty)"
    return results[0] if isinstance(results[0], str) else str(results[0])


async def main() -> None:
    # Prune data and system metadata before running, only if we want "fresh" state.
    await cognee.forget(everything=True)

    await cognee.remember(CONVERSATION, dataset_name=DATASET, self_improvement=False)

    # The index is built during improve(); it is off by default.
    await cognee.improve(dataset=DATASET, build_global_context_index=True)

    print(f"Query: {QUERY}")
    print("\n--- context WITHOUT the global context index ---")
    print(await context_for(include_global_context=False))
    print("\n--- context WITH the global context index ---")
    print("(look for the 'World summary:' / 'Relevant areas:' prelude)")
    print(await context_for(include_global_context=True))


if __name__ == "__main__":
    asyncio.run(main())
