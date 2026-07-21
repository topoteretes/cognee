# ruff: noqa: E402
"""
Tutorial: Migrate from mem0 to Cognee (using Mem0Source)

This tutorial demonstrates how to bring **mem0** memories into Cognee using
the built-in ``Mem0Source`` importer. The importer reads a mem0 export and
translates each memory into a :class:`COGXMemory` record, which lands in the
Cognee knowledge graph.

What this tutorial covers
-------------------------
- Obtaining or preparing a mem0 dump (platform export / ``get_all()`` output).
- Importing with **preserve** mode (memories map directly to graph nodes with
  zero LLM calls — fast, no token cost).
- Importing with **re-derive** mode (cognee re-runs its extraction pipeline on
  the raw memory text — you get entity/relation extraction in addition to the
  raw memories).
- Querying the migrated memories with ``cognee.recall()`` to prove the data
  landed correctly.

Usage::

    uv run python examples/tutorials/migrate_from_mem0_tutorial.py

Requires:
    ``LLM_API_KEY`` set in ``.env`` or environment (required for recall, and
    for ``re-derive`` mode).

Reference
---------
- ``cognee/modules/migration/sources/mem0.py`` — the Mem0Source adapter.
- ``cognee/modules/migration/cogx.py`` — the COGXMemory record definition.
- ``examples/demos/remember_recall_improve_example.py`` — the v1.0 memory API.
"""

import asyncio
import os
from pathlib import Path

# Set up caching and environment BEFORE importing cognee.
# Cognee reads env-backed settings at import time, so values assigned later
# may not override defaults or ``.env``.
os.environ["CACHING"] = "true"
os.environ["CACHE_BACKEND"] = "fs"

import cognee
from cognee.modules.migration.cogx import COGXMemory
from cognee.modules.migration.sources.mem0 import Mem0Source

DATASET = "mem0_migration_tutorial"
SAMPLE_EXPORT = Path(__file__).parent / "data" / "mem0_sample_export.json"


async def main():
    """Run the mem0 migration tutorial end-to-end."""

    from cognee.infrastructure.databases.relational.create_db_and_tables import (
        create_db_and_tables,
    )

    await create_db_and_tables()

    # Clear cached config so the env vars above take effect.
    from cognee.infrastructure.databases.cache.config import get_cache_config

    get_cache_config.cache_clear()

    # ------------------------------------------------------------------
    # Step 1: Start clean
    # ------------------------------------------------------------------
    print("--- Step 1: forget(everything=True) ---")
    await cognee.forget(everything=True)
    print("  All existing data cleared.\n")

    # ------------------------------------------------------------------
    # Step 2: Import with preserve mode (zero LLM calls)
    # ------------------------------------------------------------------
    # preserve mode maps each memory straight into a graph node — no LLM
    # extraction is run.  This is fast and cost-free; useful when you
    # want to keep the external system's memories exactly as they are.
    print("--- Step 2: remember(Mem0Source, mode='preserve') ---")
    await cognee.remember(
        Mem0Source(SAMPLE_EXPORT, mode="preserve"),
        dataset_name=DATASET,
    )
    print("  Mem0 export ingested in preserve mode.\n")

    # ------------------------------------------------------------------
    # Step 3: Recall — prove the memories are queryable
    # ------------------------------------------------------------------
    print("--- Step 3: recall() — query migrated memories ---")
    answer = await cognee.recall(
        "Who developed the theory of general relativity and when?",
        datasets=[DATASET],
    )
    print(f"  Answer: {answer}\n")

    print("--- Step 4: recall() — query across multiple facts ---")
    answer = await cognee.recall(
        "Which scientists won Nobel Prizes?",
        datasets=[DATASET],
    )
    print(f"  Answer: {answer}\n")

    print("--- Step 5: recall() — query by category ---")
    answer = await cognee.recall(
        "What scientific discoveries were made in the early 20th century?",
        datasets=[DATASET],
    )
    print(f"  Answer: {answer}\n")

    # ------------------------------------------------------------------
    # Step 6: Import the same data with re-derive mode
    # ------------------------------------------------------------------
    # re-derive mode makes cognee re-run its extraction pipeline (cognify)
    # on the raw memory text.  This costs LLM tokens but produces richer
    # entities, relationships, and facts on top of the source memories.
    DATASET_REDERIVE = "mem0_migration_tutorial_rederive"

    print("--- Step 6: remember(Mem0Source, mode='re-derive') ---")
    await cognee.remember(
        Mem0Source(SAMPLE_EXPORT, mode="re-derive"),
        dataset_name=DATASET_REDERIVE,
    )
    print("  Mem0 export ingested in re-derive mode (cognify ran).\n")

    print("--- Step 7: recall() — query re-derived dataset ---")
    answer = await cognee.recall(
        "Where did Marie Curie conduct her research?",
        datasets=[DATASET_REDERIVE],
    )
    print(f"  Answer: {answer}\n")

    # ------------------------------------------------------------------
    # Step 8: Using an inline payload (no file needed)
    # ------------------------------------------------------------------
    # Mem0Source also accepts already-parsed Python lists/dicts — useful
    # when you fetch memories from the mem0 live API and want to pipe them
    # straight into Cognee without touching the filesystem.
    print("--- Step 8: remember() with inline Python dict ---")
    inline_data = {
        "results": [
            {
                "id": "inline-001",
                "memory": "Alan Turing proposed the Turing Test in 1950, laying the philosophical foundations of artificial intelligence.",
                "user_id": "user-inline",
                "categories": ["computer_science", "ai"],
                "created_at": "2024-06-01T12:00:00Z",
            }
        ]
    }
    await cognee.remember(
        Mem0Source(inline_data, mode="preserve"),
        dataset_name=DATASET,
    )
    print("  Inline payload ingested.\n")

    print("--- Step 9: recall() — query inline-imported memory ---")
    answer = await cognee.recall(
        "What did Alan Turing propose?",
        datasets=[DATASET],
    )
    print(f"  Answer: {answer}\n")

    # ------------------------------------------------------------------
    # Step 10: Manual inspection — see the COGXMemory records
    # ------------------------------------------------------------------
    # The raw COGXMemory records are available when you iterate over the
    # source's records() generator directly.  Each one represents a mem0
    # memory mapped into the standard COGX exchange format.
    print("--- Step 10: inspect Mem0Source.records() ---")
    source = Mem0Source(SAMPLE_EXPORT, mode="preserve")
    record_count = 0
    async for record in source.records():
        assert isinstance(record, COGXMemory), (
            f"Expected COGXMemory, got {type(record).__name__}"
        )
        record_count += 1
        print(
            f"  [{record.external_id}] {record.content[:80]}..."
            if len(record.content) > 80
            else f"  [{record.external_id}] {record.content}"
        )
    print(f"  Total records yielded: {record_count}\n")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print("--- Step 11: forget(everything=True) ---")
    result = await cognee.forget(everything=True)
    print(f"  {result}")

    print("\nDone.  Mem0 memories have been migrated and are queryable via cognee.recall().")


if __name__ == "__main__":
    asyncio.run(main())
