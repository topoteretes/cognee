"""
Tutorial: migrate memories from Letta/MemGPT and Zep to Cognee.

Import a Letta agent file and a Zep graph export with their respective
sources, then query the migrated content with cognee.recall().

This tutorial uses "preserve" mode by default, which maps the source
data directly into Cognee's graph without re-running entity extraction.
See the import modes table at the bottom for alternatives.

Usage:
    uv run python examples/tutorials/migrate_from_letta_and_zep_tutorial.py

Requires:
    LLM_API_KEY set in .env or environment (or run with mocks, see below).

COGX format reference:
    cognee/modules/migration/COGX.md
"""

import asyncio
import json
from pathlib import Path

import cognee
from cognee.migration import LettaSource, ZepSource

SCRIPT_DIR = Path(__file__).parent
LETTA_DUMP = SCRIPT_DIR / "data" / "sample_letta_agent.json"
ZEP_DUMP = SCRIPT_DIR / "data" / "sample_zep_graph.json"


def print_step(number: int, title: str) -> None:
    print(f"\n--- Step {number}: {title} ---")


async def main():
    # Start clean so results are reproducible
    await cognee.forget(everything=True)

    # ==================================================================
    # Part 1: Import from Letta/MemGPT
    # ==================================================================
    #
    # LettaSource reads a .af agent file (JSON) and maps it to COGX:
    #
    #   Letta concept        ->  COGX record kind
    #   -----------------------------------------------
    #   Core memory block    ->  COGXMemoryBlock
    #   Message history      ->  COGXEpisode (with turns)
    #   Archival passage     ->  COGXDocument
    #
    # The sample file contains a research assistant agent with two
    # core memory blocks, a short conversation, and two archival
    # passages about neuroscience.

    print("=" * 60)
    print("Part 1: Importing from Letta/MemGPT")
    print("=" * 60)

    print_step(1, "Load and inspect the Letta agent file")
    letta_data = json.loads(LETTA_DUMP.read_text(encoding="utf-8"))
    print(f"  Agent name: {letta_data['name']}")
    print(f"  Core memory blocks: {len(letta_data.get('core_memory', []))}")
    print(f"  Messages: {len(letta_data.get('messages', []))}")
    print(f"  Archival passages: {len(letta_data.get('archival_memory', []))}")

    print_step(2, "Import with LettaSource (mode=preserve)")
    result = await cognee.remember(
        LettaSource(LETTA_DUMP, mode="preserve"),
        dataset_name="letta_import",
    )
    print(f"  Status: {result.status}")
    print(f"  Items processed: {result.items_processed}")

    print_step(3, "Recall migrated Letta content")
    answer = await cognee.recall(
        "What do you know about Hopfield networks and recurrent neural networks?",
        datasets=["letta_import"],
    )
    print(f"  Result: {answer}")

    # ==================================================================
    # Part 2: Import from Zep
    # ==================================================================
    #
    # ZepSource reads a Zep/Graphiti JSON export and maps it to COGX:
    #
    #   Zep concept          ->  COGX record kind
    #   -----------------------------------------------
    #   Episode              ->  COGXEpisode
    #   Entity node          ->  COGXEntity
    #   Fact/edge            ->  COGXFact (with valid_at/invalid_at)
    #
    # The Zep format carries bi-temporal validity timestamps on facts,
    # which are preserved as edge properties in Cognee's graph.
    #
    # The sample file contains a small project management graph with
    # two episodes, five entities, and four facts.

    print("\n" + "=" * 60)
    print("Part 2: Importing from Zep")
    print("=" * 60)

    print_step(4, "Load and inspect the Zep graph export")
    zep_data = json.loads(ZEP_DUMP.read_text(encoding="utf-8"))
    print(f"  Episodes: {len(zep_data.get('episodes', []))}")
    print(f"  Entities: {len(zep_data.get('entities', []))}")
    print(f"  Facts: {len(zep_data.get('facts', []))}")

    print_step(5, "Import with ZepSource (mode=preserve)")
    result = await cognee.remember(
        ZepSource(ZEP_DUMP, mode="preserve"),
        dataset_name="zep_import",
    )
    print(f"  Status: {result.status}")
    print(f"  Items processed: {result.items_processed}")

    print_step(6, "Recall migrated Zep content")
    answer = await cognee.recall(
        "Who is working on the analytics platform and what database are they using?",
        datasets=["zep_import"],
    )
    print(f"  Result: {answer}")

    # ==================================================================
    # Cleanup
    # ==================================================================

    print_step(7, "Clean up")
    await cognee.forget(everything=True)
    print("  All data removed.")

    # ------------------------------------------------------------------
    # Import modes reference
    # ------------------------------------------------------------------
    print(
        """
Import modes (applies to both LettaSource and ZepSource):

  preserve   - Map source records directly into the graph. No LLM calls.
               Best for restoring backups or trusting the source graph.

  re-derive  - Ingest raw content (episodes, memory blocks, passages)
               and run Cognee's extraction pipeline. Costs LLM tokens.
               The source's derived graph is rendered as text digests.

  hybrid     - Both: preserve the source graph AND cognify raw content.
               Costs LLM tokens but gives the richest result.

Example:
    await cognee.remember(ZepSource("export.json", mode="hybrid"), ...)
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
