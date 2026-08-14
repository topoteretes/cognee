import asyncio
import cognee

from os import path
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.memify_pipelines.consolidate_entities import consolidate_entities_pipeline

custom_prompt = """
Extract every place mentioned in the text as an entity, keeping the exact
surface form used in the text (so "NYC" stays "NYC").
Connect people to places with the relationship "visited".
Ignore all other entities.
"""


async def main():
    # Prune data and system metadata before running, only if we want "fresh" state.
    await cognee.forget(everything=True)
    # Ingest the two texts separately: extracted together, the LLM resolves the
    # abbreviation and emits a single entity, leaving nothing to merge.
    await cognee.remember(
        "Sara visited New York City last spring.",
        custom_prompt=custom_prompt,
        self_improvement=False,
    )
    await cognee.remember(
        "Bob thinks NYC has the best bagels.",
        custom_prompt=custom_prompt,
        self_improvement=False,
    )

    await visualize_graph(
        path.join(path.dirname(__file__), ".artifacts", "before_entity_deduplication.html")
    )

    # Preview the merge plan in the logs without touching the graph.
    await consolidate_entities_pipeline(similarity_threshold=0.6, dry_run=True)

    # Apply the merge for real.
    await consolidate_entities_pipeline(similarity_threshold=0.6)

    await visualize_graph(
        path.join(path.dirname(__file__), ".artifacts", "after_entity_deduplication.html")
    )


if __name__ == "__main__":
    asyncio.run(main())
