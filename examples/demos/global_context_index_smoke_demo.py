"""
Smoke demo for the global context index.

Usage:
    uv run python examples/demos/global_context_index_smoke_demo.py

Requires:
    LLM_API_KEY set in .env or environment.
"""

import asyncio

import cognee
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import ERROR, setup_logging


DATASET = "global_context_index_smoke_demo"

CONVERSATION_TURNS = [
    "[2026-04-02 09:14] User: Could you suggest three dates for our team syncs this month? One per week would be perfect.",
    "[2026-04-02 09:15] Assistant: I can help with that. April 5, 12, and 19 look clear. Do you generally prefer morning or afternoon slots for these?",
    "[2026-04-02 09:17] User: Let's go with mornings. Everyone seems to have more energy then.",
    "[2026-04-02 09:18] Assistant: How about 10:00 for the first two and 11:00 for the last one? I'll block off 60 minutes for each.",
    "[2026-04-02 09:20] User: That works perfectly. Please put all three on the calendar.",
    "[2026-04-02 09:21] Assistant: Done. I've scheduled Meeting 1, 2, and 3. You and the team should receive the invites shortly.",
    "[2026-04-07 11:05] User: The first meeting went well. However, I need to adjust the remaining two. Can we cancel one and move the other to an earlier date?",
    "[2026-04-07 11:07] Assistant: Of course. Which session should I remove, and what's the new target date for the remaining one?",
    "[2026-04-07 11:09] User: Cancel Meeting 3. For Meeting 2, let's move it from April 12 up to April 9, keeping the same time.",
    "[2026-04-07 11:10] Assistant: Got it. Meeting 3 is cancelled, and Meeting 2 is now set for Thursday, April 9, at 10:00.",
    "[2026-04-07 11:12] User: Thanks. Can you just give me a quick summary of where we stand with all three meetings now?",
    "[2026-04-07 11:13] Assistant: Certainly. Meeting 1 was completed on April 5. Meeting 2 is rescheduled for April 9 at 10:00. Meeting 3 has been officially cancelled.",
]

SMOKE_QUERIES = [
    "When is the first meeting?",
    "When is the second meeting?",
    "When is the third meeting?",
]

COMPARISON_QUERY = SMOKE_QUERIES[1]
WORLD_SUMMARY_HEADER = "World summary:"
RELEVANT_AREAS_HEADER = "Relevant areas:"


async def _search_context(query: str, include_global_context: bool) -> str:
    results = await cognee.search(
        query_text=query,
        query_type=SearchType.GRAPH_COMPLETION,
        datasets=[DATASET],
        only_context=True,
        retriever_specific_config={
            "include_global_context_index": include_global_context,
            "global_context_index_top_k": 3,
        },
    )
    if not results:
        return ""
    first = results[0]
    return first if isinstance(first, str) else str(first)


async def _ask_meeting_questions() -> None:
    print("\nMeeting question answers")
    for index, query in enumerate(SMOKE_QUERIES, start=1):
        results = await cognee.search(
            query_text=query,
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=[DATASET],
            retriever_specific_config={
                "include_global_context_index": True,
                "global_context_index_top_k": 3,
            },
        )
        answer = results[0] if results else ""
        if not isinstance(answer, str):
            answer = str(answer)
        print(f"\nQ{index}: {query}")
        print(f"A: {answer or '(empty)'}")


def _has_global_context_prelude(context: str) -> bool:
    return WORLD_SUMMARY_HEADER in context or RELEVANT_AREAS_HEADER in context


async def main() -> None:
    print(f"Dataset: {DATASET}")
    print("Clearing existing data...")
    await cognee.forget(everything=True)

    print("Ingesting conversation with remember()...")
    await cognee.remember(
        CONVERSATION_TURNS,
        dataset_name=DATASET,
        self_improvement=False,
    )

    print("Running improve() with global context indexing enabled...")
    await cognee.improve(dataset=DATASET, build_global_context_index=True)

    print(f"\nContext comparison for: {COMPARISON_QUERY}")
    off_context = await _search_context(COMPARISON_QUERY, include_global_context=False)
    on_context = await _search_context(COMPARISON_QUERY, include_global_context=True)

    print("\n--- Context WITHOUT global context index ---")
    print(off_context or "(empty)")
    print("\n--- Context WITH global context index ---")
    print(on_context or "(empty)")

    if not _has_global_context_prelude(on_context):
        print("\nGlobal context smoke status: FAILED (prelude missing)")
        return

    await _ask_meeting_questions()
    print("\nGlobal context smoke status: PASSED")


if __name__ == "__main__":
    setup_logging(log_level=ERROR)
    asyncio.run(main())
