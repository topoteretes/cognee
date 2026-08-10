"""Keep edge vector indexes aligned with graph-edge deletion."""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.edge_index_points import edge_instance_id
from cognee.tasks.storage.index_data_points import index_data_points


def _edge_fields(edge: Any) -> tuple[Any, Any, str, Mapping[str, Any]]:
    """Read structural fields from an edge deletion snapshot."""
    if hasattr(edge, "edge") and hasattr(edge, "edge_properties"):
        identity = edge.edge
        return (
            identity.source_id,
            identity.target_id,
            identity.relationship_name,
            edge.edge_properties,
        )
    if hasattr(edge, "source_node_id"):
        return (
            edge.source_node_id,
            edge.destination_node_id,
            edge.relationship_name,
            edge.attributes or {},
        )
    source_id, target_id, relationship_name, properties = edge
    return source_id, target_id, relationship_name, properties or {}


async def delete_edge_instances(vector_engine, edges: Iterable[Any]) -> None:
    """Delete instance points for graph edges that were actually hard-deleted."""
    instance_ids = dict.fromkeys(
        str(edge_instance_id(source, target, relationship_name, properties))
        for source, target, relationship_name, properties in map(_edge_fields, edges)
    )
    if instance_ids:
        await vector_engine.delete_data_points("EdgeInstance_text", list(instance_ids))


async def sync_edge_types(
    graph_engine,
    vector_engine,
    relationship_names: Iterable[str],
    *,
    removed_edge_counts: Mapping[str, int] | None = None,
) -> None:
    """Reindex positive relationship counts and delete relationship types at zero.

    ``removed_edge_counts`` lets callers synchronize the post-delete vector
    state before hard-deleting graph edges. This keeps a failed vector cleanup
    retryable through the edge's still-present provenance.
    """
    names = list(dict.fromkeys(name for name in relationship_names if name))
    if not names:
        return

    counts = await graph_engine.get_edge_type_counts(names)
    if removed_edge_counts:
        removed = Counter(removed_edge_counts)
        counts = {name: max(0, count - removed[name]) for name, count in counts.items()}
    positive = [
        EdgeType(relationship_name=name, number_of_edges=count)
        for name, count in counts.items()
        if count > 0
    ]
    zero_ids = [str(EdgeType.id_for(name)) for name, count in counts.items() if count == 0]
    if positive:
        await index_data_points(positive, vector_engine=vector_engine)
    if zero_ids:
        await vector_engine.delete_data_points("EdgeType_relationship_name", zero_ids)
