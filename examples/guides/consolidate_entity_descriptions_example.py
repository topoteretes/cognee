import asyncio
import cognee

from os import path
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.memify_pipelines.consolidate_entity_descriptions import consolidate_entity_descriptions

custom_prompt = """
Extract only people and cities as entities.
Connect people to cities with the relationship "lives_in".
Ignore all other entities.
"""

graph_visualization_path_before_enrichment = path.join(
    path.dirname(__file__), "before_enrichment.html"
)
graph_visualization_path_after_enrichment = path.join(
    path.dirname(__file__), "after_enrichment.html"
)


async def main():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add(
        [
            "Alice moved to Paris in 2010, while Bob has always lived in New York.",
            "Andreas was born in Venice, but later settled in Lisbon.",
            "Diana and Tom were born and raised in Helsinki. Diana currently resides in Berlin, while Tom never moved.",
        ]
    )
    await cognee.cognify(custom_prompt=custom_prompt)

    await visualize_graph(graph_visualization_path_before_enrichment)

    await consolidate_entity_descriptions()

    await visualize_graph(graph_visualization_path_after_enrichment)


if __name__ == "__main__":
    asyncio.run(main())
