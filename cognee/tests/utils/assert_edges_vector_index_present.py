from typing import Any, Mapping, Sequence

from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text


def _edge_parts(edge: Sequence[Any]) -> tuple[str, str, str, Mapping[str, Any]]:
    source_id, target_id, relationship_name = edge[:3]
    properties = edge[3] if len(edge) > 3 and edge[3] else {}
    return str(source_id), str(target_id), str(relationship_name), properties


async def assert_edges_vector_index_present(
    relationships: Sequence[Sequence[Any]], convert_to_new_format: bool = True
):
    """Assert separated type and instance points for the supplied graph edges.

    ``convert_to_new_format`` is retained for legacy callers; both formats now
    use relationship names for ``EdgeType`` identity and edge-object IDs for
    ``EdgeInstance`` identity.
    """
    vector_engine = await get_vector_engine_async()

    type_points = {}
    instance_points = {}
    for relationship in relationships:
        source_id, target_id, relationship_name, properties = _edge_parts(relationship)
        type_id = str(EdgeType.id_for(relationship_name))
        instance_id = str(
            properties.get("edge_object_id")
            or generate_edge_object_id(source_id, target_id, relationship_name)
        )
        type_points[type_id] = relationship_name
        instance_points[instance_id] = get_edge_retrieval_text(
            properties.get("edge_text"), relationship_name
        )

    vector_items = await vector_engine.retrieve("EdgeType_relationship_name", list(type_points))
    vector_items_by_id = {str(vector_item.id): vector_item for vector_item in vector_items}
    for type_id, relationship_name in type_points.items():
        assert type_id in vector_items_by_id, (
            f"Relationship '{relationship_name}' not found in vector store."
        )
        assert vector_items_by_id[type_id].payload["text"] == relationship_name

    vector_items = await vector_engine.retrieve("EdgeInstance_text", list(instance_points))
    vector_items_by_id = {str(vector_item.id): vector_item for vector_item in vector_items}
    for instance_id, edge_text in instance_points.items():
        assert instance_id in vector_items_by_id, (
            f"Edge instance '{edge_text}' not found in vector store."
        )
        assert vector_items_by_id[instance_id].payload["text"] == edge_text
