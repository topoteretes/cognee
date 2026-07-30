"""
Migrate from mem0 to Cognee using Mem0Source.

Demonstrates how to import a mem0 export (platform JSON or OSS dump)
into Cognee's knowledge graph and query the migrated memory.

Three import modes:
  - ``"preserve"``  — map mem0 memories straight to graph nodes (no LLM).
  - ``"re-derive"`` — ingest raw text and run Cognee's own extraction.
  - ``"hybrid"``    — preserve the graph *and* cognify raw content.

Usage:
    uv run python examples/tutorials/migrate_from_mem0_tutorial.py

Requires:
    LLM_API_KEY set in .env (needed for re-derive mode and recall).

The sample mem0 export lives at ``examples/tutorials/data/mem0_export.json``.
"""

import asyncio
from pathlib import Path

import cognee
from cognee.migration import Mem0Source
from cognee.shared.logging_utils import ERROR, setup_logging

DATA_FILE = Path(__file__).parent / "data" / "mem0_export.json"


async def main():
    from cognee.infrastructure.databases.relational.create_db_and_tables import (
        create_db_and_tables,
    )

    await create_db_and_tables()

    await cognee.forget(everything=True)

    # ------------------------------------------------------------------
    # Step 1: Inspect the mem0 export
    # ------------------------------------------------------------------
    print("Step 1 — Mem0 export file:", DATA_FILE)

    # ------------------------------------------------------------------
    # Step 2: Import with preserve mode (no LLM, fastest)
    # To enable recall over the imported data, in addition to initial remember,
    # you need to call an additional cognify() (calls LLM)
    # ------------------------------------------------------------------
    print("\nStep 2 — Importing mem0 memories (mode=preserve) …")
    source = Mem0Source(DATA_FILE, mode="preserve")
    result = await cognee.remember(source)
    await cognee.cognify()
    print("  ", result)

    # ------------------------------------------------------------------
    # Step 3: Query the migrated memory
    # ------------------------------------------------------------------
    print("\nStep 3 — Recall: what database does the project use?")
    answers = await cognee.recall("What database does the project use?")
    for answer in answers:
        print("  ", answer)

    print("\nStep 4 — Recall: who is Alice's manager?")
    answers = await cognee.recall("Who is Alice's manager?")
    for answer in answers:
        print("  ", answer)

    # ------------------------------------------------------------------
    # Step 5: Re-import with re-derive mode (runs cognify)
    # ------------------------------------------------------------------
    await cognee.forget(everything=True)
    print("\nStep 5 — Re-importing with mode=re-derive (runs cognify) …")
    source = Mem0Source(DATA_FILE, mode="re-derive")
    result = await cognee.remember(source)
    print("  ", result)

    # ------------------------------------------------------------------
    # Step 6: Query again after re-derive
    # ------------------------------------------------------------------
    print("\nStep 6 — Recall after re-derive: what are the Q2 OKRs?")
    answers = await cognee.recall("What are the Q2 OKRs?")
    for answer in answers:
        print("  ", answer)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print("\nStep 7 — Cleanup")
    await cognee.forget(everything=True)
    print("  Done.")


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
