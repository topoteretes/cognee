def get_entity_nodes_from_triplets(triplets):
    entity_nodes = []
    seen_ids = set()
    for triplet in triplets:
        if hasattr(triplet, "node1") and triplet.node1 and triplet.node1.id not in seen_ids:
            entity_nodes.append({"id": str(triplet.node1.id)})
            seen_ids.add(triplet.node1.id)
        if hasattr(triplet, "node2") and triplet.node2 and triplet.node2.id not in seen_ids:
            entity_nodes.append({"id": str(triplet.node2.id)})
            seen_ids.add(triplet.node2.id)

    return entity_nodes
