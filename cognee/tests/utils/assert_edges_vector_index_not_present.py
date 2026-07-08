from uuid import UUID
from typing import Dict, List, Tuple
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.graph.models.EdgeType import EdgeType


async def assert_edges_vector_index_not_present(relationships: List[Tuple[UUID, UUID, str, Dict]]):
    vector_engine = await get_vector_engine()

    query_edge_ids = {
        str(EdgeType.id_for(relationship[2])): relationship[2] for relationship in relationships
    }

    vector_items = await vector_engine.retrieve(
        "EdgeType_relationship_name", list(query_edge_ids.keys())
    )

    vector_items_by_id = {str(vector_item.id): vector_item for vector_item in vector_items}

    for relationship_id, relationship_name in query_edge_ids.items():
        assert relationship_id not in vector_items_by_id, (
            f"Relationship '{relationship_name}' still present in the vector store."
        )
