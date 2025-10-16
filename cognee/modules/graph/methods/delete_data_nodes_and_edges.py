from uuid import UUID
from typing import Dict, List

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine
from cognee.modules.engine.utils import generate_edge_id
from cognee.modules.graph.methods import (
    delete_data_related_edges,
    delete_data_related_nodes,
    get_data_related_nodes,
    get_data_related_edges,
)


async def delete_data_nodes_and_edges(dataset_id: UUID, data_id: UUID) -> None:
    affected_nodes = await get_data_related_nodes(dataset_id, data_id)

    if len(affected_nodes) == 0:
        return

    graph_engine = await get_graph_engine()
    await graph_engine.delete_nodes([str(node.slug) for node in affected_nodes])

    affected_vector_collections: Dict[str, List] = {}
    for node in affected_nodes:
        for indexed_field in node.indexed_fields:
            collection_name = f"{node.type}_{indexed_field}"
            if collection_name not in affected_vector_collections:
                affected_vector_collections[collection_name] = []
            affected_vector_collections[collection_name].append(node)

    vector_engine = get_vector_engine()
    for affected_collection, affected_nodes in affected_vector_collections.items():
        await vector_engine.delete_data_points(
            affected_collection, [node.id for node in affected_nodes]
        )

    affected_relationships = await get_data_related_edges(dataset_id, data_id)

    await vector_engine.delete_data_points(
        "EdgeType_relationship_name",
        [generate_edge_id(edge.relationship_name) for edge in affected_relationships],
    )

    await delete_data_related_nodes(data_id)
    await delete_data_related_edges(data_id)
