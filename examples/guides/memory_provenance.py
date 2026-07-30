"""Memory provenance: tracing a fact back to the file it came from.

``get_memory_provenance_graph()`` reads the bookkeeping cognee keeps in its relational
database and projects it into a ``(nodes, edges)`` graph whose edges form an ownership
chain:

    Tenant --has_member--> User --owns--> Dataset --contains--> TextDocument
                                                                     |
                                                                     +--mentions--> Entity / DocumentChunk / ...

The provenance lives in those **edges**, and the one that answers "where did this come
from?" is ``mentions``: it links a source file to each memory node extracted from it.
With ``include_memory=True`` the extracted graph is folded in so those links exist.

This guide ingests two documents into two datasets, then walks the chain and prints it
as a tree.

One naming note: files from the relational ``Data`` table are typed ``TextDocument``,
and the extracted memory layer can contain ``TextDocument`` nodes too — so that type
name shows up on both sides of a ``mentions`` edge.
"""

import asyncio
import os
from collections import defaultdict

import cognee
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)

FLEET_NOTES = "Carlos drives for Echo Global Logistics and files a dispatch log each morning."
DRIVER_NOTES = "Mika is a driver at Landstar. Priya reviews driver records every quarter."


async def main():
    # Prune data and system metadata before running, only if we want "fresh" state.
    await cognee.forget(everything=True)

    # Two datasets so the ownership chain has more than one branch to show.
    await cognee.remember(FLEET_NOTES, dataset_name="fleet_ops", self_improvement=False)
    await cognee.remember(DRIVER_NOTES, dataset_name="driver_records", self_improvement=False)

    # include_memory=True folds in the extracted graph and links it back to source files.
    nodes, edges = await cognee.get_memory_provenance_graph(include_memory=True)

    name_of = {node_id: properties.get("name") for node_id, properties in nodes}
    type_of = {node_id: properties.get("type") for node_id, properties in nodes}

    # Index targets by (relation, source) so the chain can be walked downwards.
    targets = defaultdict(list)
    for source, target, relation, _properties in edges:
        targets[(relation, source)].append(target)

    print("Provenance chain — who owns what, and which file each memory came from:\n")
    for node_id, properties in nodes:
        if properties.get("type") != "User":
            continue
        print(f"User: {name_of[node_id]}")
        for dataset_id in targets[("owns", node_id)]:
            print(f"  Dataset: {name_of[dataset_id]}")
            for document_id in targets[("contains", dataset_id)]:
                print(f"    File: {name_of[document_id]}")
                mentioned = targets[("mentions", document_id)]
                if not mentioned:
                    print("      (no memory linked — was include_memory=True?)")
                for memory_id in mentioned:
                    print(f"      mentions {type_of[memory_id]}: {name_of[memory_id]}")

    destination = os.path.join(os.path.dirname(__file__), ".artifacts", "memory_provenance.html")
    await cognee_network_visualization((nodes, edges), destination)
    print(f"\nSame graph rendered to {destination}")


if __name__ == "__main__":
    asyncio.run(main())
