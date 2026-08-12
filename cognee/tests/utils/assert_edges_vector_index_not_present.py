from typing import Any, Sequence

from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.graph.models.EdgeType import EdgeType


async def assert_edges_vector_index_not_present(relationships: Sequence[Sequence[Any]]):
    """Assert no type or instance points remain for the supplied graph edges."""
    vector_engine = await get_vector_engine_async()
    type_ids = {str(EdgeType.id_for(str(edge[2]))): str(edge[2]) for edge in relationships}
    instance_ids = {
        str((edge[3] if len(edge) > 3 and edge[3] else {}).get("edge_object_id"))
        if (edge[3] if len(edge) > 3 and edge[3] else {}).get("edge_object_id")
        else generate_edge_object_id(str(edge[0]), str(edge[1]), str(edge[2]))
        for edge in relationships
    }

    type_items = await vector_engine.retrieve("EdgeType_relationship_name", list(type_ids))
    for vector_item in type_items:
        assert str(vector_item.id) not in type_ids, (
            f"Relationship '{type_ids[str(vector_item.id)]}' still present in the vector store."
        )

    instance_items = await vector_engine.retrieve("EdgeInstance_text", list(instance_ids))
    for vector_item in instance_items:
        assert str(vector_item.id) not in instance_ids, (
            f"Edge instance '{vector_item.id}' still present in the vector store."
        )
