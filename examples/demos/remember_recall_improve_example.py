"""
V2 Memory-Oriented API: remember, recall, improve, forget, status.

Full lifecycle: ingest data with session tracking, check per-item
processing status, query within that session, bridge session into the
permanent graph, verify freshness via content hashes, and clean up.

Usage:
    uv run python examples/demos/remember_recall_improve_example.py

Requires:
    LLM_API_KEY set in .env or environment.
"""

import asyncio
from datetime import datetime, timezone

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

    # Record the time before ingestion for the `since` filter
    before_ingest = datetime.now(timezone.utc)

    # Step 1: Remember -- ingest + build graph + init session
    print("--- Step 1: remember(session_id) ---")
    await cognee.remember(SAMPLE_TEXT, dataset_name="scientists", session_id=SESSION)
    print("  Data ingested and session initialized.")

    # Step 2: Dataset-level status (aggregates)
    print("\n--- Step 2: status() -- dataset aggregates ---")
    statuses = await cognee.status(datasets=["scientists"])
    for s in statuses:
        print(f"  {s.dataset_name}: {s.item_count} items, cognify={s.cognify_pipeline_status}")

    # Step 3: Per-item status -- see each file's processing state and content hash
    print("\n--- Step 3: status(items=True) -- per-source detail ---")
    items = await cognee.status(datasets=["scientists"], items=True)
    for item in items:
        print(f"  {item.name}: {item.status} (hash={item.content_hash[:12]}...)")
        if item.error:
            print(f"    error: {item.error}")

    # Step 4: Filter by time -- only items ingested since we started
    print("\n--- Step 4: status(items=True, since=...) -- time-filtered ---")
    recent = await cognee.status(datasets=["scientists"], items=True, since=before_ingest)
    print(f"  {len(recent)} item(s) ingested since {before_ingest.isoformat()}")

    # Step 5: Recall in the same session
    print("\n--- Step 5: recall(session_id) ---")
    answer = await cognee.recall(
        "What did the user just tell me about Einstein?",
        datasets=["scientists"],
        session_id=SESSION,
    )
    print(f"  Answer: {answer}")

    # Step 6: Follow-up in the same session
    print("\n--- Step 6: recall() follow-up ---")
    answer = await cognee.recall(
        "Who else was mentioned and what did they do?",
        datasets=["scientists"],
        session_id=SESSION,
    )
    print(f"  Answer: {answer}")

    # Step 7: Improve -- bridge session feedback into permanent graph
    print("\n--- Step 7: improve(session_ids) ---")
    await cognee.improve(dataset="scientists", session_ids=[SESSION])
    print("  Session bridged into permanent graph.")

    # Step 8: Recall without session -- graph is enriched
    print("\n--- Step 8: recall() after improve ---")
    answer = await cognee.recall(
        "What contributions did these scientists make?",
        datasets=["scientists"],
    )
    print(f"  Answer: {answer}")

    # Step 9: Freshness check -- compare search result hashes against status
    print("\n--- Step 9: freshness check via source_content_hash ---")
    current_hashes = {item.content_hash for item in items if item.status == "completed"}
    print(f"  Current source hashes: {current_hashes}")
    print("  (At retrieval time, compare node.source_content_hash against these)")

    # Step 10: Clean up with forget
    print("\n--- Step 10: forget(dataset) ---")
    result = await cognee.forget(dataset="scientists")
    print(f"  {result}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
