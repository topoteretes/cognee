"""Consolidate Entity descriptions and EntityType summaries from the graph.

Calls consolidate_entity_descriptions_pipeline(), which rewrites each Entity's
description from its graph neighborhood, then summarizes each EntityType from
its member Entities and writes is_a edge text.
"""

import asyncio
import sys
from os import path

sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), "..", "..")))

import cognee
from cognee import visualize_graph
from cognee.memify_pipelines.consolidate_entity_descriptions import (
    consolidate_entity_descriptions_pipeline,
)

custom_prompt = """
Extract only people and cities as entities.
Connect people to cities with whatever relationship the text actually
describes (e.g. born_in, lives_in, resides_in, settled_in, visited).
Ignore all other entities.
"""

graph_visualization_path_before_enrichment = path.join(
    path.dirname(__file__), ".artifacts", "before_consolidate_enrichment_entity_descriptions.html"
)
graph_visualization_path_after_enrichment = path.join(
    path.dirname(__file__), ".artifacts", "after_consolidate_enrichment_entity_descriptions.html"
)


async def main():
    # Prune data and system metadata before running, only if we want "fresh" state.
    await cognee.forget(everything=True)
    await cognee.remember(
        [
            "Alice moved to Paris in 2010, while Bob has always lived in New York.",
            "Bob visited Paris in 2015 to see Alice.",
            "Andreas was born in Venice, but later settled in Lisbon.",
            "Diana and Tom were born and raised in Helsinki. Diana currently resides in Berlin, while Tom never moved.",
        ],
        custom_prompt=custom_prompt,
        self_improvement=False,
    )

    await visualize_graph(graph_visualization_path_before_enrichment)

    await consolidate_entity_descriptions_pipeline()

    await visualize_graph(graph_visualization_path_after_enrichment)

    # Only recall() can prove the new EntityType/is_a text is actually used in
    # retrieval - the description and edge text themselves are visible in the
    # graph visualization above.
    answer = await cognee.recall(
        "How many Person entities are in this graph, and what do they have in common?"
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
