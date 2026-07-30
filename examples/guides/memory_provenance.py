"""Memory provenance: where each piece of memory came from.

``get_memory_provenance_graph()`` reads the relational bookkeeping cognee keeps —
tenants, users, datasets, files, sessions — and projects it into a ``(nodes, edges)``
graph. With ``include_memory=True`` the extracted entities are folded in and linked
back to the file they came from, so you can trace a fact to its source.

This guide ingests two small documents into two datasets, then projects and renders
the provenance of what it just created.
"""

import asyncio
import os
from collections import Counter

import cognee
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)

FLEET_NOTES = "Carlos drives for Echo Global Logistics and files a dispatch log each morning."
DRIVER_NOTES = "Mika is a driver at Landstar. Priya reviews driver records every quarter."


async def main():
    # Prune data and system metadata before running, only if we want "fresh" state.
    await cognee.forget(everything=True)

    # Two datasets so the projection has more than one branch to show.
    await cognee.remember(FLEET_NOTES, dataset_name="fleet_ops", self_improvement=False)
    await cognee.remember(DRIVER_NOTES, dataset_name="driver_records", self_improvement=False)

    # include_memory=True links extracted entities back to their source file.
    nodes, edges = await cognee.get_memory_provenance_graph(include_memory=True)

    counts = Counter(properties.get("type") for _, properties in nodes)
    print("Provenance levels found:")
    for node_type, count in counts.most_common():
        print(f"  {node_type}: {count}")
    print(f"Edges: {len(edges)}")

    destination = os.path.join(os.path.dirname(__file__), ".artifacts", "memory_provenance.html")
    await cognee_network_visualization((nodes, edges), destination)
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    asyncio.run(main())
