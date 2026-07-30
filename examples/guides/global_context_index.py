"""Global context index: a dataset-level summary prepended to retrieval context.

``improve(..., build_global_context_index=True)`` builds a compact summary of the whole
dataset — a world summary plus the areas it covers. When
``include_global_context_index`` is on, the retriever **prepends that summary to the
context it hands the LLM**, so answers can draw on the shape of the whole dataset
instead of only the chunks nearest the query.

Note where the summary lands: it is retriever *input*, not output. A normal
``recall()`` returns an answer that was informed by the summary, not one containing
it. This guide passes ``only_context=True`` to skip generation and print the assembled
context instead, so you can diff the two directly — the extra block at the top of the
second output is the prelude.

For a longer run with a multi-day conversation and several queries, see
``examples/advanced_guides/global_context_index_smoke_demo.py``.
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


async def main() -> None:
    # Prune data and system metadata before running, only if we want "fresh" state.
    await cognee.forget(everything=True)

    await cognee.remember(CONVERSATION, dataset_name=DATASET, self_improvement=False)

    # The index is built during improve(); it is off by default.
    await cognee.improve(dataset=DATASET, build_global_context_index=True)

    print(f"Query: {QUERY}")

    # only_context=True skips generation and returns the assembled context instead,
    # so `text` holds what the retriever would have handed the LLM.
    print("\nCONTEXT WITHOUT the global context index")
    without_index = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        only_context=True,
        retriever_specific_config={
            "include_global_context_index": False,
            "global_context_index_top_k": 3,
        },
    )
    print(without_index[0].text)

    print("\nCONTEXT WITH the global context index")
    with_index = await cognee.recall(
        query_text=QUERY,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        only_context=True,
        retriever_specific_config={
            "include_global_context_index": True,
            "global_context_index_top_k": 3,
        },
    )
    print(with_index[0].text)


if __name__ == "__main__":
    asyncio.run(main())
