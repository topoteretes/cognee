"""Schema and entity inventory over the knowledge graph.

Summarizes the graph by semantic type: per-type instance counts, representative
sample names, and the per-pair relationship distribution. Composes the verified
building blocks ``get_graph_engine`` and ``get_graph_data`` and reads node/edge
fields using their exact shapes.

A node's semantic type is its ``type`` property, except for extracted entities
whose ``type`` is the literal ``"Entity"``. For those, the semantic type is
resolved by following the ``is_a`` edge (Entity -> EntityType) to the target
EntityType node's ``name``.
"""

from typing import Any
from uuid import UUID

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

# Relationship name and direction of the Entity -> EntityType edge (verified).
ENTITY_TYPE_RELATION = "is_a"


# Internal graph taxonomy types that must not appear as separate type groups in
# the inventory. EntityType nodes act as semantic-type labels — their role is
# resolved onto the Entity instances via the is_a edge, so surfacing them as a
# separate group would double-count. Filtering them here also drops is_a edges
# from the relationship distribution (their target is always an EntityType node).
_INTERNAL_TYPES: frozenset[str] = frozenset({"EntityType"})


def _resolve_node_types(
    nodes: list[tuple[str, dict[str, Any]]],
    edges: list[tuple[str, str, str, dict[str, Any]]],
) -> dict[str, str | None]:
    """Map each node id to its semantic type name.

    Non-Entity nodes keep their ``type`` property. Entity nodes (``type ==
    "Entity"``) resolve to the EntityType ``name`` reached via the ``is_a`` edge.
    """
    node_props = {node_id: props for node_id, props in nodes}

    # Collect EntityType target name for each Entity source via the is_a edge
    entity_type_name: dict[str, str | None] = {}
    for source_id, target_id, relation, _ in edges:
        if relation == ENTITY_TYPE_RELATION and target_id in node_props:
            entity_type_name[source_id] = node_props[target_id].get("name")

    node_type: dict[str, str | None] = {}
    for node_id, props in nodes:
        raw_type = props.get("type")
        if raw_type == "Entity" and node_id in entity_type_name:
            node_type[node_id] = entity_type_name[node_id]
        else:
            node_type[node_id] = raw_type
    return node_type


def _compute_degrees(
    node_ids: list[str] | set[str],
    edges: list[tuple[str, str, str, dict[str, Any]]],
) -> dict[str, int]:
    """Count edges touching each node (undirected degree) from the edge list."""
    degree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for source_id, target_id, _, _ in edges:
        if source_id in degree:
            degree[source_id] += 1
        if target_id in degree:
            degree[target_id] += 1
    return degree


VALID_SORTS: frozenset[str] = frozenset({"count", "none"})


async def get_schema_inventory(
    dataset: str | UUID | None = None,
    samples_per_type: int = 5,
    sort: str = "count",
) -> list[dict[str, Any]]:
    """Summarize the knowledge graph by semantic type.

    Parameters:
        dataset: optional dataset id/name to scope the graph databases to.
            When set, scoping mirrors the visualize router via
            ``set_database_global_context_variables``.
        samples_per_type: maximum number of sample instance names per type.
        sort: one of ``VALID_SORTS``. ``"count"`` (default) orders types by
            descending count, then type name; ``"none"`` preserves discovery
            order. Any other value raises ``ValueError``.

    Returns:
        A list of dicts with keys ``type``, ``count``, ``samples``,
        ``sample_size``, and ``relationships``. Each ``relationships`` entry is
        ``{"to_type", "relation", "count"}`` aggregated over edges.
    """
    if samples_per_type < 0:
        raise ValueError("samples_per_type must be non-negative")
    if sort not in VALID_SORTS:
        raise ValueError(f"sort must be one of {sorted(VALID_SORTS)}, got {sort!r}")
    if dataset is not None:
        # Scope graph databases to the dataset, mirroring the visualize router.
        # String dataset names cannot resolve to an owner_id; skip scoping for them
        # rather than calling the context manager with a None owner (which would raise).
        owner_id = await _resolve_dataset_owner(dataset)
        if owner_id is not None:
            async with set_database_global_context_variables(dataset, owner_id):
                return await _build_inventory(samples_per_type, sort)
    return await _build_inventory(samples_per_type, sort)


async def _resolve_dataset_owner(dataset: str | UUID) -> UUID | None:
    """Return the owner id for a dataset, or None when it cannot be resolved."""
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Dataset

    if not isinstance(dataset, UUID):
        return None

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        record = await session.get(Dataset, dataset)
        return record.owner_id if record else None


async def _build_inventory(samples_per_type: int, sort: str) -> list[dict[str, Any]]:
    """Fetch graph data and assemble the per-type inventory."""
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()

    node_type = _resolve_node_types(nodes, edges)
    node_name = {node_id: props.get("name") for node_id, props in nodes}
    degree = _compute_degrees(node_type.keys(), edges)

    # Group node ids by semantic type, skipping internal taxonomy types
    ids_by_type: dict[str, list[str]] = {}
    for node_id, type_name in node_type.items():
        if type_name in _INTERNAL_TYPES:
            continue
        ids_by_type.setdefault(type_name, []).append(node_id)

    # Aggregate relationship counts keyed by (source_type, target_type, relation).
    # Skip edges where either endpoint is an internal type.
    relation_counts: dict[tuple[str, str, str], int] = {}
    for source_id, target_id, relation, _ in edges:
        source_type = node_type.get(source_id)
        target_type = node_type.get(target_id)
        if source_type is None or target_type is None:
            continue
        if source_type in _INTERNAL_TYPES or target_type in _INTERNAL_TYPES:
            continue
        key = (source_type, relation, target_type)
        relation_counts[key] = relation_counts.get(key, 0) + 1

    # Build outgoing relationships per type AND incoming (so types like DocumentChunk
    # whose primary connections are incoming are not shown as isolated nodes).
    relationships_by_type: dict[str, list[dict[str, Any]]] = {}
    for (source_type, relation, target_type), count in relation_counts.items():
        outgoing = {"to_type": target_type, "relation": relation, "count": count}
        relationships_by_type.setdefault(source_type, []).append(outgoing)
        incoming = {"to_type": source_type, "relation": f"\u2190 {relation}", "count": count}
        relationships_by_type.setdefault(target_type, []).append(incoming)

    # Build one inventory record per semantic type
    inventory = []
    for type_name, node_ids in ids_by_type.items():
        # Sort instances by descending degree, then name as a stable tiebreaker
        ranked = sorted(node_ids, key=lambda nid: (-degree[nid], node_name.get(nid) or ""))
        samples = [node_name[nid] for nid in ranked[:samples_per_type]]

        relationships = sorted(
            relationships_by_type.get(type_name, []),
            key=lambda rel: (-rel["count"], rel["to_type"] or "", rel["relation"]),
        )
        inventory.append(
            {
                "type": type_name,
                "count": len(node_ids),
                "samples": samples,
                "sample_size": len(samples),
                "relationships": relationships,
            }
        )

    if sort == "count":
        inventory.sort(key=lambda rec: (-rec["count"], rec["type"] or ""))
    return inventory
