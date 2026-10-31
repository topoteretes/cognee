"""
Migrate from Letta (MemGPT) and Zep / Graphiti to Cognee.

Demonstrates how to import a Letta agent file (``.af`` JSON) and a Zep /
Graphiti graph export into Cognee's knowledge graph, then query the
migrated memory with ``cognee.recall``.

Letta import (``re-derive`` mode by default):
    - Core memory blocks  -> DataItems (re-derived by Cognee's extraction)
    - Message history      -> DataItems (episodes)
    - Archival passages   -> DataItems (documents)

Zep import (``hybrid`` mode by default):
    - Episodes             -> DataItems (cognified) + COGX episodes
    - Entity nodes         -> Cognee Entity DataPoints (preserved)
    - Fact edges           -> Cognee relationship edges (preserved, with
      bi-temporal ``valid_at`` / ``invalid_at`` properties)

Concept mapping (COGX record kinds):
    Letta concept             COGX record kind
    -----------------------    ----------------------
    core memory block          COGXMemoryBlock
    message history            COGXEpisode
    archival memory passage    COGXDocument

    Zep concept                COGX record kind
    -----------------------    ----------------------
    episode                    COGXEpisode
    entity node                COGXEntity
    fact / relation edge       COGXFact

Usage:
    uv run python examples/demos/ingestion_and_migration/migrate_from_letta_and_zep/migrate_from_letta_and_zep.py

Requires:
    LLM_API_KEY set in .env (needed for re-derive mode and recall).

Sample dumps live at ``examples/demos/ingestion_and_migration/migrate_from_letta_and_zep/data/``.
"""

import asyncio
from pathlib import Path

import cognee
from cognee.migration import LettaSource, ZepSource
from cognee.shared.logging_utils import ERROR, setup_logging

LETTA_FILE = Path(__file__).parent / "data" / "sample_letta_dump.json"
ZEP_FILE = Path(__file__).parent / "data" / "sample_zep_dump.json"


async def main():
    from cognee.infrastructure.databases.relational.create_db_and_tables import (
        create_db_and_tables,
    )

    await create_db_and_tables()
    await cognee.forget(everything=True)

    # ==================================================================
    # Part A — Migrate a Letta (MemGPT) agent file
    # ==================================================================

    # ------------------------------------------------------------------
    # Step 1: Inspect the Letta dump
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Part A — Letta (MemGPT) migration")
    print("=" * 60)
    print("\nStep 1 — Letta agent file:", LETTA_FILE)

    # ------------------------------------------------------------------
    # Step 2: Import with re-derive mode (default for LettaSource)
    # ------------------------------------------------------------------
    # ``re-derive`` ingests all memory blocks, messages, and archival
    # passages as DataItems, then runs Cognee's own extraction (cognify)
    # to build a fresh knowledge graph. This is the safest mode when
    # you want Cognee to re-derive entities and relationships from
    # raw text rather than trusting the source's graph structure.
    print("\nStep 2 — Importing Letta agent (mode=re-derive) …")
    letta_source = LettaSource(LETTA_FILE, mode="re-derive")
    result = await cognee.remember(letta_source)
    print("  ", result)

    # ------------------------------------------------------------------
    # Step 3: Query the migrated Letta memory
    # ------------------------------------------------------------------
    print("\nStep 3 — Recall: what editor does Alice prefer?")
    answers = await cognee.recall("What editor does Alice prefer?")
    for answer in answers:
        print("  ", answer)

    print("\nStep 4 — Recall: who is Alice's manager?")
    answers = await cognee.recall("Who is Alice's manager?")
    for answer in answers:
        print("  ", answer)

    # ==================================================================
    # Part B — Migrate a Zep / Graphiti graph export
    # ==================================================================

    # ------------------------------------------------------------------
    # Step 5: Reset and inspect the Zep dump
    # ------------------------------------------------------------------
    await cognee.forget(everything=True)
    print("\n" + "=" * 60)
    print("Part B — Zep / Graphiti migration")
    print("=" * 60)
    print("\nStep 5 — Zep export file:", ZEP_FILE)

    # ------------------------------------------------------------------
    # Step 6: Import with hybrid mode (default for ZepSource)
    # ------------------------------------------------------------------
    # ``hybrid`` preserves the Zep graph's entities and facts directly
    # as Cognee DataPoints (zero LLM calls for the graph layer) *and*
    # ingests episode content as DataItems that go through cognify,
    # so Cognee also builds its own understanding of the raw text.
    #
    # Other modes:
    #   ``preserve`` — map everything to graph nodes/edges (no LLM at all)
    #   ``re-derive`` — discard the source graph, ingest all text fresh
    print("\nStep 6 — Importing Zep graph (mode=hybrid) …")
    zep_source = ZepSource(ZEP_FILE, mode="hybrid")
    result = await cognee.remember(zep_source)
    print("  ", result)

    # ------------------------------------------------------------------
    # Step 7: Query the migrated Zep memory
    # ------------------------------------------------------------------
    print("\nStep 7 — Recall: what database does Alice use?")
    answers = await cognee.recall("What database does Alice use?")
    for answer in answers:
        print("  ", answer)

    print("\nStep 8 — Recall: who manages Alice?")
    answers = await cognee.recall("Who manages Alice?")
    for answer in answers:
        print("  ", answer)

    print("\nStep 9 — Recall: what was decided in the project kickoff?")
    answers = await cognee.recall("What was decided in the project kickoff?")
    for answer in answers:
        print("  ", answer)

    # ==================================================================
    # Part C — Both sources together (combined memory)
    # ==================================================================

    # ------------------------------------------------------------------
    # Step 10: Import both sources into the same graph
    # ------------------------------------------------------------------
    await cognee.forget(everything=True)
    print("\n" + "=" * 60)
    print("Part C — Combined migration (Letta + Zep)")
    print("=" * 60)
    print("\nStep 10 — Importing both sources …")

    letta_source = LettaSource(LETTA_FILE, mode="re-derive")
    zep_source = ZepSource(ZEP_FILE, mode="hybrid")

    result_letta = await cognee.remember(letta_source)
    print("  Letta:", result_letta)
    result_zep = await cognee.remember(zep_source)
    print("  Zep:  ", result_zep)

    # ------------------------------------------------------------------
    # Step 11: Query the combined memory
    # ------------------------------------------------------------------
    print("\nStep 11 — Recall (combined): what technologies does Alice use?")
    answers = await cognee.recall("What technologies does Alice use?")
    for answer in answers:
        print("  ", answer)

    print("\nStep 12 — Recall (combined): what are the Q2 OKRs?")
    answers = await cognee.recall("What are the Q2 OKRs?")
    for answer in answers:
        print("  ", answer)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print("\nStep 13 — Cleanup")
    await cognee.forget(everything=True)
    print("  Done.")


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    asyncio.run(main())
