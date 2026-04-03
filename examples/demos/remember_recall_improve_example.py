"""
V2 Memory-Oriented API: remember, recall, improve, forget, status.

Demonstrates two memory patterns:
  1. Permanent memory -- remember() without session_id ingests data
     directly into the knowledge graph.
  2. Session memory -- remember() with session_id stores data in the
     session cache only. improve() syncs session content into the
     permanent graph.

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
from cognee import session

PERMANENT_TEXT = (
    "Albert Einstein developed the theory of general relativity, "
    "which describes gravity as the curvature of spacetime caused by mass and energy. "
    "He published this work in 1915 while working at the University of Berlin. "
    "Marie Curie was the first woman to win a Nobel Prize and remains the only person "
    "to win Nobel Prizes in two different sciences: physics and chemistry. "
    "She conducted pioneering research on radioactivity at the Sorbonne in Paris."
)

SESSION_TEXT_1 = (
    "The Sorbonne, formally known as the University of Paris, has been a center of "
    "academic excellence since the 13th century. Albert Einstein gave several lectures "
    "there during his visits to France."
)

SESSION_TEXT_2 = (
    "Niels Bohr proposed the atomic model with quantized electron orbits in 1913. "
    "He worked closely with Einstein on quantum mechanics debates throughout the 1920s."
)

DATASET = "scientists"
SESSION = "demo_session"


async def main():
    from cognee.infrastructure.databases.relational.create_db_and_tables import (
        create_db_and_tables,
    )

    await create_db_and_tables()

    # Enable filesystem-based session caching (required for session_id and improve)
    import os

    os.environ["CACHING"] = "true"
    os.environ["CACHE_BACKEND"] = "fs"

    from cognee.infrastructure.databases.cache.config import get_cache_config

    get_cache_config.cache_clear()

    await cognee.forget(everything=True)

    before_ingest = datetime.now(timezone.utc)

    # ----------------------------------------------------------------
    # Part 1: Permanent memory -- remember() without session
    # ----------------------------------------------------------------

    # Ingest data directly into the knowledge graph.
    print("--- Step 1: remember() -- permanent memory ---")
    await cognee.remember(PERMANENT_TEXT, dataset_name=DATASET)
    print("  Data ingested into permanent graph.")

    cognee.remember( "blalba", session_id='123')


    cognee.improve(dataset=DATASET)

    # Per-item status
    print("\n--- Step 2: status(items=True) -- per-source detail ---")
    items = await cognee.status(datasets=[DATASET], items=True)
    for item in items:
        print(f"  {item.name}: {item.status} (hash={item.content_hash[:12]}...)")

    # Query the permanent graph
    print("\n--- Step 3: recall() -- query permanent memory ---")
    answer = await cognee.recall(
        "What is the theory of general relativity?",
        datasets=[DATASET],
    )
    print(f"  Answer: {answer}")

    # ----------------------------------------------------------------
    # Part 2: Session memory -- remember() with session_id
    # ----------------------------------------------------------------

    # Store data in the session cache only. No add/cognify runs.
    # Multiple calls accumulate entries in the same session.
    print("\n--- Step 4: remember(session_id) -- session memory (entry 1) ---")
    await cognee.remember(SESSION_TEXT_1, session_id=SESSION)
    print("  Stored in session cache.")

    print("\n--- Step 5: remember(session_id) -- session memory (entry 2) ---")
    await cognee.remember(SESSION_TEXT_2, session_id=SESSION)
    print("  Stored in session cache.")

    # Status still shows only the permanent data -- session data is not in the graph yet
    print("\n--- Step 6: status() -- only permanent data visible ---")
    statuses = await cognee.status()
    for s in statuses:
        print(f"  {s.dataset_name}: {s.item_count} items, cognify={s.cognify_pipeline_status}")

    # Recall with session_id queries the permanent graph but the LLM also
    # sees the session conversation history as context
    print("\n--- Step 7: recall(session_id) -- session-aware query ---")
    answer = await cognee.recall(
        "What did the user mention about the Sorbonne?",
        datasets=[DATASET],
        session_id=SESSION,
    )
    print(f"  Answer: {answer}")

    print("\n--- Step 8: recall(session_id) -- follow-up ---")
    answer = await cognee.recall(
        "Who else was mentioned and what did they work on?",
        datasets=[DATASET],
        session_id=SESSION,
    )
    print(f"  Answer: {answer}")

    # ----------------------------------------------------------------
    # Part 3: Sync session memory to permanent graph via improve()
    # ----------------------------------------------------------------

    # improve() reads session entries, runs add + cognify on them,
    # persisting the session content into the permanent graph
    print("\n--- Step 9: improve(session_ids) -- sync session to permanent ---")
    await cognee.improve(dataset=DATASET, session_ids=[SESSION])
    print("  Session content synced to permanent graph.")

    # Now the graph contains both the original data and the session content
    print("\n--- Step 10: recall() -- query enriched permanent graph ---")
    answer = await cognee.recall(
        "What contributions did Einstein and Bohr make?",
        datasets=[DATASET],
    )
    print(f"  Answer: {answer}")

    # ----------------------------------------------------------------
    # Part 4: Freshness check and cleanup
    # ----------------------------------------------------------------

    print("\n--- Step 11: status(since=...) -- time-filtered ---")
    recent = await cognee.status(items=True, since=before_ingest)
    print(f"  {len(recent)} item(s) ingested since {before_ingest.isoformat()}")

    print("\n--- Step 12: freshness check via source_content_hash ---")
    all_items = await cognee.status(items=True)
    current_hashes = {item.content_hash for item in all_items if item.status == "completed"}
    print(f"  Current source hashes: {current_hashes}")
    print("  (At retrieval time, compare node.source_content_hash against these)")

    print("\n--- Step 13: forget(everything) ---")
    result = await cognee.forget(everything=True)
    print(f"  {result}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
