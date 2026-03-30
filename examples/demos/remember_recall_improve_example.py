"""
V2 Memory-Oriented API: remember, recall, improve, forget, status.

Full lifecycle: ingest data with session tracking, query within that
session (LLM sees what was just ingested), bridge session into the
permanent graph, query the enriched graph.

Usage:
    uv run python examples/demos/remember_recall_improve_example.py

Requires:
    LLM_API_KEY set in .env or environment.
"""

import asyncio
import cognee


SAMPLE_TEXT = (
    "Albert Einstein developed the theory of general relativity, "
    "which describes gravity as the curvature of spacetime caused by mass and energy. "
    "He published this work in 1915 while working at the University of Berlin. "
    "Marie Curie was the first woman to win a Nobel Prize and remains the only person "
    "to win Nobel Prizes in two different sciences: physics and chemistry. "
    "She conducted pioneering research on radioactivity at the Sorbonne in Paris. "
    "The Sorbonne, formally known as the University of Paris, has been a center of "
    "academic excellence since the 13th century. Albert Einstein gave several lectures "
    "there during his visits to France."
)

SESSION = "demo_session"


async def main():
    await cognee.forget(everything=True)

    # Step 1: Remember with session -- ingest + build graph + init session
    print("--- Step 1: remember(session_id) ---")
    await cognee.remember(SAMPLE_TEXT, dataset_name="scientists", session_id=SESSION)
    print("  Data ingested and session initialized.")

    # Step 2: Check status
    print("\n--- Step 2: status() ---")
    statuses = await cognee.status(datasets=["scientists"])
    for s in statuses:
        print(f"  {s.dataset_name}: {s.item_count} items, cognify={s.cognify_pipeline_status}")

    # Step 3: Recall in the same session -- LLM sees "user just provided data"
    print("\n--- Step 3: recall(session_id) ---")
    answer = await cognee.recall(
        "What did the user just tell me about Einstein?",
        datasets=["scientists"],
        session_id=SESSION,
    )
    print(f"  Answer: {answer}")

    # Step 4: Follow-up in the same session
    print("\n--- Step 4: recall() follow-up ---")
    answer = await cognee.recall(
        "Who else was mentioned and what did they do?",
        datasets=["scientists"],
        session_id=SESSION,
    )
    print(f"  Answer: {answer}")

    # Step 5: Improve -- bridge session feedback + Q&A into permanent graph
    print("\n--- Step 5: improve(session_ids) ---")
    await cognee.improve(dataset="scientists", session_ids=[SESSION])
    print("  Session bridged into permanent graph.")

    # Step 6: Recall without session -- graph is enriched with session content
    print("\n--- Step 6: recall() after improve ---")
    answer = await cognee.recall(
        "What contributions did these scientists make?",
        datasets=["scientists"],
    )
    print(f"  Answer: {answer}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
