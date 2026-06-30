"""Migrating from Graphiti to Cognee.

If you have a Graphiti knowledge graph and want to move it into Cognee,
this tutorial walks you through exactly that. The GraphitiSource importer
already handles the heavy lifting — you just point it at your export and
choose an import mode.

Graphiti exports three types of records, and here's how they land in Cognee:

  episodes  ->  COGXEpisode   raw ingested text, stored verbatim
  entities  ->  COGXEntity    named nodes in the knowledge graph
  facts     ->  COGXFact      edges between nodes, with valid_at/invalid_at
                              timestamps that track when each fact was true

GraphitiSource takes care of the mapping, so once the import finishes you
can query the data immediately with cognee.recall().

COGX standard: https://docs.cognee.ai/cogx

Choosing an import mode
-----------------------
  preserve   Writes entities and facts straight into the graph as-is.
             No LLM calls, no token cost. Fastest option, and the graph
             structure stays exactly as it was in Graphiti.

  re-derive  Ignores the existing graph and re-runs Cognee's extraction
             pipeline on the raw episode text. Costs LLM tokens, but you
             get a graph that's fully rebuilt to Cognee's model.

  hybrid     Does both: preserve the Graphiti graph and cognify the
             episode text. This is the default for GraphitiSource, since
             Graphiti keeps both the derived graph and the source episodes.

One thing to know upfront
-------------------------
GraphitiSource reads from a file/dict dump — it's not a live connection
to a running Graphiti instance. If you need to pull directly from a live
Graphiti setup, that's tracked separately.

Running this tutorial:
    uv run python examples/tutorials/migrate_from_graphiti_tutorial.py

You'll need:
    LLM_API_KEY in your .env file or environment.

Relevant source files:
    cognee/modules/migration/sources/zep.py       (GraphitiSource lives here)
    cognee/modules/migration/import_source.py
    cognee/modules/migration/sources/base.py      (MemorySource + import modes)
"""

import asyncio

import cognee
from cognee.modules.migration.sources.zep import GraphitiSource

# ---------------------------------------------------------------------------
# Sample Graphiti export
# ---------------------------------------------------------------------------
# In a real migration, you'd load this from a JSON file — typically a Cypher
# dump of EntityNode, EpisodicNode, and RELATES_TO records from your Neo4j
# database. GraphitiSource handles minor key-name differences gracefully,
# so "nodes" or "entities", "edges" or "facts", "episode_body" or "content"
# all work fine.

GRAPHITI_DUMP = {
    "episodes": [
        {
            "uuid": "ep-001",
            "name": "Turing biography excerpt",
            "content": (
                "Alan Turing was a British mathematician and computer scientist. "
                "He formalized the concept of computation with the Turing machine in 1936 "
                "and made foundational contributions to artificial intelligence."
            ),
            "created_at": "2024-01-10T09:00:00Z",
            "valid_at": "2024-01-10T09:00:00Z",
        },
        {
            "uuid": "ep-002",
            "name": "von Neumann biography excerpt",
            "content": (
                "John von Neumann was a Hungarian-American mathematician. "
                "He contributed to quantum mechanics, game theory, and the architecture "
                "of modern computers (the von Neumann architecture)."
            ),
            "created_at": "2024-01-10T09:05:00Z",
            "valid_at": "2024-01-10T09:05:00Z",
        },
    ],
    "entities": [
        {
            "uuid": "ent-turing",
            "name": "Alan Turing",
            "labels": ["Person", "Mathematician"],
            "summary": "British mathematician who formalized computation and pioneered AI.",
            "created_at": "2024-01-10T09:00:00Z",
        },
        {
            "uuid": "ent-von-neumann",
            "name": "John von Neumann",
            "labels": ["Person", "Mathematician"],
            "summary": "Hungarian-American mathematician; contributed to computer architecture.",
            "created_at": "2024-01-10T09:05:00Z",
        },
        {
            "uuid": "ent-turing-machine",
            "name": "Turing Machine",
            "labels": ["Concept"],
            "summary": "Abstract model of computation introduced by Turing in 1936.",
            "created_at": "2024-01-10T09:01:00Z",
        },
    ],
    "facts": [
        {
            "uuid": "fact-001",
            "source_node_uuid": "ent-turing",
            "target_node_uuid": "ent-turing-machine",
            "name": "formalized",
            "fact": "Alan Turing formalized the concept of computation with the Turing machine in 1936.",
            # valid_at and invalid_at mark the real-world time window for this fact.
            "valid_at": "1936-01-01T00:00:00Z",
            "invalid_at": None,  # None means this fact is still true today
            "created_at": "2024-01-10T09:01:00Z",
            "episodes": ["ep-001"],
        },
        {
            "uuid": "fact-002",
            "source_node_uuid": "ent-von-neumann",
            "target_node_uuid": "ent-turing",
            "name": "collaborated_with",
            "fact": "Von Neumann and Turing both worked on the foundations of computing.",
            "valid_at": "1945-01-01T00:00:00Z",
            "invalid_at": None,
            "created_at": "2024-01-10T09:06:00Z",
            "episodes": ["ep-001", "ep-002"],
        },
    ],
}

DATASET = "graphiti_migration"


async def main():
    # ------------------------------------------------------------------
    # Step 1: Start fresh.
    #
    # Wipe any existing Cognee state so this tutorial runs clean every
    # time regardless of what's already in the database.
    # ------------------------------------------------------------------
    print("Step 1: Resetting Cognee state...")
    await cognee.forget(everything=True)
    print("  Done.\n")

    # ------------------------------------------------------------------
    # Step 2: Import the Graphiti dump using preserve mode.
    #
    # preserve mode skips the LLM entirely — it maps each record type
    # from the Graphiti export directly into the Cognee graph:
    #
    #   episodes -> COGXEpisode   (raw text blocks)
    #   entities -> COGXEntity    (named nodes)
    #   facts    -> COGXFact      (typed edges, with timestamps)
    #
    # This is the fastest option and keeps your original graph intact.
    # ------------------------------------------------------------------
    print("Step 2: Importing Graphiti dump into Cognee (preserve mode)...")
    source = GraphitiSource(GRAPHITI_DUMP, mode="preserve")
    result = await cognee.remember(source, dataset_name=DATASET)
    print(f"  Import result: {result}\n")

    # ------------------------------------------------------------------
    # Step 3: Query the migrated data.
    #
    # Once the import is done, cognee.recall() works just like it would
    # for any other dataset — the migration is transparent.
    # ------------------------------------------------------------------
    print("Step 3: Recalling migrated content...")

    answer = await cognee.recall(
        "What did Alan Turing contribute to computing?",
        datasets=[DATASET],
    )
    print("  Q: What did Alan Turing contribute to computing?")
    print(f"  A: {answer}\n")

    answer = await cognee.recall(
        "What is the relationship between Turing and von Neumann?",
        datasets=[DATASET],
    )
    print("  Q: What is the relationship between Turing and von Neumann?")
    print(f"  A: {answer}\n")

    # ------------------------------------------------------------------
    # Step 4: How bi-temporal facts are handled.
    #
    # Graphiti tracks two timestamps per fact: when the fact became true
    # in the real world (valid_at) and when it stopped being true
    # (invalid_at, or None if it's still current). GraphitiSource carries
    # these through to COGXFact, so temporal context isn't lost.
    #
    # From our sample data:
    #   fact-001  valid_at=1936-01-01  invalid_at=None
    #     -> Turing formalized the Turing machine in 1936; still holds.
    #   fact-002  valid_at=1945-01-01  invalid_at=None
    #     -> Von Neumann and Turing's collaboration; still holds.
    # ------------------------------------------------------------------
    print("Step 4: Bi-temporal fact provenance is preserved.")
    print("  fact-001 valid_at=1936-01-01 -> Turing formalized the Turing machine.")
    print("  fact-002 valid_at=1945-01-01 -> Von Neumann + Turing collaboration.\n")

    # ------------------------------------------------------------------
    # Step 5: The other two import modes.
    #
    # We're skipping these here to avoid LLM token costs, but here's
    # how you'd use them:
    #
    # re-derive — throw away the Graphiti graph and rebuild from the raw
    #             episode text using Cognee's extraction pipeline.
    #
    #   source = GraphitiSource(GRAPHITI_DUMP, mode="re-derive")
    #   await cognee.remember(source, dataset_name=DATASET)
    #
    # hybrid    — run both: preserve the existing graph AND cognify the
    #             episode text. Default for GraphitiSource, since Graphiti
    #             exports include both the graph and the raw episodes.
    #
    #   source = GraphitiSource(GRAPHITI_DUMP, mode="hybrid")
    #   await cognee.remember(source, dataset_name=DATASET)
    # ------------------------------------------------------------------
    print("Step 5: Other modes — re-derive and hybrid (skipped to save LLM tokens).")
    print("  See the module docstring at the top of this file for details.\n")

    # ------------------------------------------------------------------
    # Step 6: Clean up.
    # ------------------------------------------------------------------
    print("Step 6: Cleaning up...")
    await cognee.forget(everything=True)
    print("  Done.")

    print("\nMigration tutorial complete.")


if __name__ == "__main__":
    asyncio.run(main())
