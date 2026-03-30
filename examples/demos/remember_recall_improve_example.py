"""
V2 Memory-Oriented API: remember, recall, improve, forget, status.

Demonstrates two memory patterns:
  1. Permanent memory -- remember() without session_id ingests data
     directly into the knowledge graph.
  2. Session memory -- remember() with session_id tracks a conversation.
     improve() then syncs session Q&A and feedback into the permanent graph.

Also shows per-source tracking (status with items/since) and freshness
checking via source_content_hash on graph nodes.

Usage:
    uv run python examples/demos/remember_recall_improve_example.py

Requires:
    LLM_API_KEY set in .env or environment.
"""

import asyncio
from datetime import datetime, timezone

import cognee


BACKGROUND_TEXT = (
    "The Sorbonne, formally known as the University of Paris, has been a center of "
    "academic excellence since the 13th century. It has hosted many notable scientists "
    "and thinkers throughout its history."
)

SCIENTISTS_TEXT = (
    "Albert Einstein developed the theory of general relativity, "
    "which describes gravity as the curvature of spacetime caused by mass and energy. "
    "He published this work in 1915 while working at the University of Berlin. "
    "Marie Curie was the first woman to win a Nobel Prize and remains the only person "
    "to win Nobel Prizes in two different sciences: physics and chemistry. "
    "She conducted pioneering research on radioactivity at the Sorbonne in Paris. "
    "Albert Einstein gave several lectures there during his visits to France."
)

SESSION = "demo_session"


async def main():
    # Ensure database tables exist (creates them on first run or after deletion)
    from cognee.infrastructure.databases.relational.create_db_and_tables import (
        create_db_and_tables,
    )

    await create_db_and_tables()

    # Enable filesystem-based session caching (required for session_id and improve)
    import os

    os.environ["CACHING"] = "true"
    os.environ["CACHE_BACKEND"] = "fs"

    # Clear cached config so the new env vars take effect
    from cognee.infrastructure.databases.cache.config import get_cache_config

    get_cache_config.cache_clear()

    await cognee.forget(everything=True)

    # Record the time before ingestion for the `since` filter
    before_ingest = datetime.now(timezone.utc)

    # ----------------------------------------------------------------
    # Part 1: Permanent memory -- remember() without session
    # ----------------------------------------------------------------

    # Ingest background knowledge directly into the graph.
    # No session tracking -- this is persistent, shared context.
    print("--- Step 1: remember() -- permanent memory (no session) ---")
    await cognee.remember(BACKGROUND_TEXT, dataset_name="institutions")
    print("  Background data ingested into permanent graph.")

    # Check per-item status of what we just ingested
    print("\n--- Step 2: status(items=True) -- per-source detail ---")
    items = await cognee.status(datasets=["institutions"], items=True)
    for item in items:
        print(f"  {item.name}: {item.status} (hash={item.content_hash[:12]}...)")

    # Query the permanent graph -- no session context needed
    print("\n--- Step 3: recall() -- query permanent memory ---")
    answer = await cognee.recall(
        "What is the Sorbonne?",
        datasets=["institutions"],
    )
    print(f"  Answer: {answer}")

    # ----------------------------------------------------------------
    # Part 2: Session memory -- remember() with session_id
    # ----------------------------------------------------------------

    # Ingest data with session tracking. The session records what the
    # user provided, so subsequent recall() calls in the same session
    # give the LLM awareness of the conversation history.
    print("\n--- Step 4: remember(session_id) -- session memory ---")
    await cognee.remember(
        SCIENTISTS_TEXT,
        dataset_name="scientists",
        session_id=SESSION,
    )
    print("  Data ingested and session initialized.")

    # Dataset-level status (aggregates across both datasets)
    print("\n--- Step 5: status() -- dataset aggregates ---")
    statuses = await cognee.status()
    for s in statuses:
        print(f"  {s.dataset_name}: {s.item_count} items, cognify={s.cognify_pipeline_status}")

    # Time-filtered status -- only items since we started
    print("\n--- Step 6: status(since=...) -- time-filtered ---")
    recent = await cognee.status(items=True, since=before_ingest)
    print(f"  {len(recent)} item(s) ingested since {before_ingest.isoformat()}")

    # Recall within the session -- LLM sees "user just provided data"
    print("\n--- Step 7: recall(session_id) -- session-aware query ---")
    answer = await cognee.recall(
        "What did the user just tell me about Einstein?",
        datasets=["scientists"],
        session_id=SESSION,
    )
    print(f"  Answer: {answer}")

    # Follow-up in the same session
    print("\n--- Step 8: recall(session_id) -- follow-up ---")
    answer = await cognee.recall(
        "Who else was mentioned and what did they do?",
        datasets=["scientists"],
        session_id=SESSION,
    )
    print(f"  Answer: {answer}")

    # ----------------------------------------------------------------
    # Part 3: Sync session memory to permanent graph via improve()
    # ----------------------------------------------------------------

    # improve() with session_ids does three things:
    #   1. Applies feedback weights from session entries to graph nodes
    #   2. Persists session Q&A text into the permanent graph
    #   3. Runs default enrichment (triplet embeddings)
    print("\n--- Step 9: improve(session_ids) -- sync session to permanent ---")
    await cognee.improve(dataset="scientists", session_ids=[SESSION])
    print("  Session Q&A and feedback synced to permanent graph.")

    # Now recall without session -- the graph contains both the original
    # data AND the session Q&A that was bridged in by improve()
    print("\n--- Step 10: recall() -- query enriched permanent graph ---")
    answer = await cognee.recall(
        "What contributions did these scientists make?",
        datasets=["scientists"],
    )
    print(f"  Answer: {answer}")

    # ----------------------------------------------------------------
    # Part 4: Freshness check and cleanup
    # ----------------------------------------------------------------

    # Freshness: graph nodes carry source_content_hash from their source
    # document. Compare against status() to verify nodes are current.
    print("\n--- Step 11: freshness check via source_content_hash ---")
    all_items = await cognee.status(items=True)
    current_hashes = {item.content_hash for item in all_items if item.status == "completed"}
    print(f"  Current source hashes: {current_hashes}")
    print("  (At retrieval time, compare node.source_content_hash against these)")

    # Clean up
    print("\n--- Step 12: forget(everything) ---")
    result = await cognee.forget(everything=True)
    print(f"  {result}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
