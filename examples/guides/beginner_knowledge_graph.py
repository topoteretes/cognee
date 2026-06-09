"""
Beginner-Friendly Knowledge Graph Example
==========================================

This example walks you through Cognee's core workflow step by step,
showing exactly how raw text becomes a searchable knowledge graph.

The workflow has three stages:

    1. REMEMBER  -- Feed text into Cognee.  Cognee chunks the text,
                    uses an LLM to extract entities and relationships,
                    and stores them in a knowledge graph + vector index.

    2. EXPLORE   -- Inspect the graph that Cognee built so you can see
                    the nodes (entities) and edges (relationships).

    3. RECALL    -- Ask natural-language questions and get answers
                    grounded in the knowledge graph.

Prerequisites
-------------
    pip install cognee          # or: uv pip install cognee

    # Set your OpenAI API key (or any supported LLM provider):
    export LLM_API_KEY="sk-..."

Usage
-----
    python examples/guides/beginner_knowledge_graph.py

What to expect
--------------
    The script prints each stage with clear headers so you can follow
    along.  The "Explore" section prints every entity and relationship
    that Cognee extracted -- this is the knowledge graph.
"""

import asyncio
import cognee
from cognee.infrastructure.databases.graph import get_graph_engine


# ---------------------------------------------------------------------------
# Sample text -- a short paragraph with several entities and relationships
# that are easy to verify by eye.
# ---------------------------------------------------------------------------
SAMPLE_TEXT = (
    "Marie Curie was a physicist and chemist who conducted pioneering "
    "research on radioactivity. She was born in Warsaw, Poland, and later "
    "moved to Paris, France, where she worked at the University of Paris. "
    "In 1903, she became the first woman to win a Nobel Prize, sharing the "
    "Nobel Prize in Physics with her husband Pierre Curie and Henri Becquerel. "
    "She won a second Nobel Prize in Chemistry in 1911 for her discovery of "
    "the elements radium and polonium."
)


async def main():
    # ==================================================================
    # Stage 0: Start fresh (optional -- useful for a clean demo)
    # ==================================================================
    print("=" * 60)
    print("Stage 0: Resetting Cognee (clean slate for the demo)")
    print("=" * 60)
    await cognee.forget(everything=True)
    print("Done.\n")

    # ==================================================================
    # Stage 1: REMEMBER -- ingest text into the knowledge graph
    # ==================================================================
    print("=" * 60)
    print("Stage 1: REMEMBER -- feeding text into Cognee")
    print("=" * 60)
    print(f"\nInput text:\n  {SAMPLE_TEXT[:80]}...\n")

    await cognee.remember(SAMPLE_TEXT, self_improvement=False)
    print("Cognee has processed the text and built a knowledge graph.\n")

    # ==================================================================
    # Stage 2: EXPLORE -- inspect the graph that was built
    # ==================================================================
    print("=" * 60)
    print("Stage 2: EXPLORE -- what did Cognee extract?")
    print("=" * 60)
    print()
    print("A knowledge graph stores information as:")
    print("  - NODES (entities): people, places, concepts, events")
    print("  - EDGES (relationships): how entities relate to each other")
    print()
    print("Below is the graph Cognee built from the input text.")
    print("Compare it to the original paragraph to see how Cognee")
    print("turned unstructured prose into structured knowledge.")

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    # --- Print nodes (entities) ---
    # Each node has a type (e.g. Person, Place) and properties like
    # name and description.  Cognee extracted these using an LLM.
    print(f"\nEntities (graph nodes): {len(nodes)}")
    print("-" * 40)
    for node_id, properties in nodes:
        name = properties.get("name", properties.get("text", str(node_id)))
        node_type = properties.get("_type", "unknown")
        description = properties.get("description", "")
        if name and node_type:
            line = f"  [{node_type}] {name}"
            if description:
                line += f"  --  {description[:80]}"
            print(line)

    # --- Print edges (relationships) ---
    # Each edge connects two nodes and has a label describing the
    # relationship, e.g. "Marie Curie --[born_in]--> Warsaw".
    print(f"\nRelationships (graph edges): {len(edges)}")
    print("-" * 40)
    # Build a lookup so we can show human-readable names instead of UUIDs.
    id_to_name = {}
    for node_id, properties in nodes:
        id_to_name[str(node_id)] = properties.get("name", properties.get("text", str(node_id)[:8]))

    for source_id, target_id, relationship, _props in edges:
        src = id_to_name.get(str(source_id), str(source_id)[:8])
        tgt = id_to_name.get(str(target_id), str(target_id)[:8])
        print(f"  {src}  --[{relationship}]-->  {tgt}")

    print()
    print("Tip: The exact entities and labels depend on the LLM, so your")
    print("output may differ slightly.  The important thing is that Cognee")
    print("extracted structured knowledge from plain text automatically.")

    # ==================================================================
    # Stage 3: RECALL -- ask questions about the knowledge graph
    # ==================================================================
    print("=" * 60)
    print("Stage 3: RECALL -- querying the knowledge graph")
    print("=" * 60)

    questions = [
        "Where was Marie Curie born?",
        "What Nobel Prizes did Marie Curie win?",
        "Who did Marie Curie share the Nobel Prize in Physics with?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        answers = await cognee.recall(question)
        for answer in answers:
            print(f"A: {answer}")

    # ==================================================================
    # Cleanup
    # ==================================================================
    print("\n" + "=" * 60)
    print("Done!  You have successfully:")
    print("  1. Fed text into Cognee          (remember)")
    print("  2. Inspected the knowledge graph (explore)")
    print("  3. Queried it with questions      (recall)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
