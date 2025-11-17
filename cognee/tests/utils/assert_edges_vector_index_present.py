from uuid import UUID
from typing import List, Tuple
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.engine.utils import generate_edge_id


async def assert_edges_vector_index_present(relationships: List[Tuple[UUID, UUID, str]]):
    vector_engine = get_vector_engine()

    query_edge_ids = {
        str(generate_edge_id(relationship[2])): relationship[2] for relationship in relationships
    }

    vector_items = await vector_engine.retrieve(
        "EdgeType_relationship_name", list(query_edge_ids.keys())
    )

    vector_items_by_id = {str(vector_item.id): vector_item for vector_item in vector_items}

    for relationship_id, relationship_name in query_edge_ids.items():
        assert relationship_id in vector_items_by_id, (
            f"Relationship '{relationship_name}' not found in vector store."
        )

        vector_relationship = vector_items_by_id[relationship_id]
        assert vector_relationship.payload["text"] == relationship_name, (
            f"Vectorized edge '{vector_relationship.payload['text']}' does not match the relationship text '{relationship_name}'."
        )
