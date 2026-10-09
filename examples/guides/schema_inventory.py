"""Schema and entity inventory in the rendered graph.

``visualize_graph`` writes an HTML file with a schema side panel: every entity type
cognee extracted, how many instances of each, sample names, and which relationships
connect the types. This guide ingests a handful of sentences about one domain so
several distinct types show up, then renders the panel.

Open the HTML and use the side panel — clicking a type highlights its instances in
the graph.
"""

import asyncio
import os

import cognee
from cognee import visualize_graph

DATASET = "schema_inventory_guide"

# One domain, several entity types (people, brokers, a tool) so the panel has
# something to group.
TEXT = [
    "Carlos works at Echo Global Logistics and uses the load-search tool every morning.",
    "Mika works at Landstar and also uses the load-search tool.",
    "Priya manages the load-search tool.",
    "Sandra interviewed Carlos and Priya about their daily routine.",
]


async def main():
    # Prune data and system metadata before running, only if we want "fresh" state.
    await cognee.forget(everything=True)

    await cognee.remember(TEXT, dataset_name=DATASET, self_improvement=False)

    destination = os.path.join(os.path.dirname(__file__), ".artifacts", "schema_inventory.html")
    await visualize_graph(destination, dataset=DATASET)

    print(f"Wrote {destination}")
    print("Open it and expand the schema panel to see types, counts, and samples.")


if __name__ == "__main__":
    asyncio.run(main())
