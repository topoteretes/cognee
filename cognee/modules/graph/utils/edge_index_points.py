from collections import Counter
from typing import Any, Iterable, Mapping, NamedTuple

from cognee.modules.engine.utils import generate_edge_object_id
from cognee.modules.graph.models.EdgeInstance import EdgeInstance
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text


class EdgeIndexPoints(NamedTuple):
    edge_types: list[EdgeType]
    edge_instances: list[EdgeInstance]


def edge_instance_id(
    source_id: Any,
    target_id: Any,
    relationship_name: str,
    properties: Mapping[str, Any] | None,
) -> str:
    stored = (properties or {}).get("edge_object_id")
    return (
        str(stored) if stored else generate_edge_object_id(source_id, target_id, relationship_name)
    )


def _relationship_name(value: Any) -> str:
    relationship_name = "" if value is None else str(value).strip()
    if not relationship_name:
        raise ValueError("relationship_name must be nonblank")
    return relationship_name


def build_edge_index_points(
    edges: Iterable[tuple[Any, Any, Any, Mapping[str, Any] | None]],
    relationship_counts: Mapping[str, int] | None = None,
) -> EdgeIndexPoints:
    local_counts: Counter[str] = Counter()
    instances_by_id: dict[str, EdgeInstance] = {}

    for source_id, target_id, raw_relationship_name, properties in edges:
        relationship_name = _relationship_name(raw_relationship_name)
        properties = properties or {}
        local_counts[relationship_name] += 1
        instance_id = edge_instance_id(source_id, target_id, relationship_name, properties)
        instances_by_id[instance_id] = EdgeInstance(
            id=instance_id,
            text=get_edge_retrieval_text(properties.get("edge_text"), relationship_name),
            relationship_name=relationship_name,
            source_node_id=str(source_id),
            target_node_id=str(target_id),
        )

    edge_types = [
        EdgeType(
            relationship_name=relationship_name,
            number_of_edges=(relationship_counts or {}).get(relationship_name, count),
        )
        for relationship_name, count in local_counts.items()
    ]
    return EdgeIndexPoints(edge_types=edge_types, edge_instances=list(instances_by_id.values()))
