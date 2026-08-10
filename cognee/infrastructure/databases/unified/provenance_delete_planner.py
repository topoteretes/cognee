"""Retry-safe planner for graph-provenance source-ref removal.

Given snapshots of the matched nodes/edges and the set of source refs to remove
from each, the planner decides which artifacts become *unowned* (no owning
source ref remains -> hard delete) versus which merely *survive* (some ref
remains -> detach the targeted refs only). It then performs the removal in a
retry-safe order:

  1. delete vectors for unowned artifacts (from snapshots only),
  2. ``remove_*_source_refs`` for the targeted refs on ALL matched artifacts
     (idempotent),
  3. ``delete_nodes`` / ``delete_edge_triples`` for the unowned artifacts.

Vectors are deleted first so that a failure leaves graph provenance intact and a
retry converges. All three steps are individually idempotent.

Vector ids mirror ``delete_from_graph_and_vector``:
  - node -> collection ``f"{node_type}_{field}"`` for each indexed field,
    id = node_id;
  - edge -> an ``EdgeInstance_text`` row keyed by its stable ``edge_object_id``;
    shared ``EdgeType_relationship_name`` rows are synchronized after graph
    deletion from surviving relationship names. ``Triplet_text`` remains a
    best-effort compatibility index keyed by
    ``generate_node_id(source_id + relationship_name + target_id)``.
"""

from cognee.infrastructure.databases.provenance import (
    EdgeDeleteData,
    EdgeIdentity,
    NodeDeleteData,
)
from cognee.modules.engine.utils import generate_node_id
from cognee.shared.logging_utils import get_logger

logger = get_logger("provenance_delete_planner")


def _is_unowned(current_refs: list[str], removed_refs: list[str]) -> bool:
    """True when removing ``removed_refs`` leaves no owning source ref."""
    return len(set(current_refs) - set(removed_refs)) == 0


async def _delete_vector_points(vector_engine, collection: str, ids: list[str]) -> None:
    """Delete vector points, treating absent collections as an idempotent no-op."""
    if not ids:
        return

    if not await vector_engine.has_collection(collection):
        return

    await vector_engine.delete_data_points(collection, ids)


async def execute_source_ref_removal(
    graph_engine,
    vector_engine,
    *,
    node_data: dict[str, NodeDeleteData],
    edge_data: dict[EdgeIdentity, EdgeDeleteData],
    refs_by_node: dict[str, list[str]],
    refs_by_edge: dict[EdgeIdentity, list[str]],
) -> None:
    """Remove the given source refs and hard-delete artifacts that become unowned."""
    # ------------------------------------------------------------------
    # 1. Partition matched artifacts into unowned (delete) vs surviving (detach).
    # ------------------------------------------------------------------
    unowned_node_ids: list[str] = []
    for node_id, data in node_data.items():
        removed = refs_by_node.get(node_id, [])
        if _is_unowned(data.source_ref_keys, removed):
            unowned_node_ids.append(node_id)

    unowned_edges: list[EdgeIdentity] = []
    for edge, data in edge_data.items():
        removed = refs_by_edge.get(edge, [])
        if _is_unowned(data.source_ref_keys, removed):
            unowned_edges.append(edge)

    # ------------------------------------------------------------------
    # 2. Delete vectors for unowned artifacts (from snapshots only).
    # ------------------------------------------------------------------
    node_vector_collections: dict[str, list[str]] = {}
    for node_id in unowned_node_ids:
        data = node_data[node_id]
        for field in data.indexed_fields:
            collection_name = f"{data.node_type}_{field}"
            node_vector_collections.setdefault(collection_name, []).append(node_id)

    for collection, ids in node_vector_collections.items():
        await _delete_vector_points(vector_engine, collection, ids)

    if unowned_edges:
        # Per-edge triplet vectors are tied to a single deleted edge instance, so
        # they are safe to delete here. EdgeType vectors are keyed by *shared*
        # relationship names and may still be used by a surviving edge, so they
        # are handled in _cleanup_orphaned_edge_types (after the graph delete,
        # against the truly-orphaned text set) — never blindly here.
        triplet_ids: list[str] = [
            str(generate_node_id(edge.source_id + edge.relationship_name + edge.target_id))
            for edge in unowned_edges
        ]
        if triplet_ids:
            await _delete_vector_points(vector_engine, "Triplet_text", triplet_ids)

    # ------------------------------------------------------------------
    # 3. Remove the targeted refs from SURVIVING artifacts only (idempotent).
    #    Unowned artifacts keep their refs until they are hard-deleted below, so
    #    a failed hard delete leaves them rediscoverable by source ref on retry
    #    (Part 0 retry-safe order: delete unowned without first stripping refs).
    # ------------------------------------------------------------------
    unowned_node_set = set(unowned_node_ids)
    nodes_by_removed_refs: dict[tuple[str, ...], list[str]] = {}
    for node_id in node_data:
        if node_id in unowned_node_set:
            continue
        removed = refs_by_node.get(node_id, [])
        if not removed:
            continue
        nodes_by_removed_refs.setdefault(tuple(removed), []).append(node_id)

    for removed_refs, node_ids in nodes_by_removed_refs.items():
        await graph_engine.remove_node_source_refs(node_ids, list(removed_refs))

    unowned_edge_set = set(unowned_edges)
    edges_by_removed_refs: dict[tuple[str, ...], list[EdgeIdentity]] = {}
    for edge in edge_data:
        if edge in unowned_edge_set:
            continue
        removed = refs_by_edge.get(edge, [])
        if not removed:
            continue
        edges_by_removed_refs.setdefault(tuple(removed), []).append(edge)

    for removed_refs, edges in edges_by_removed_refs.items():
        await graph_engine.remove_edge_source_refs(edges, list(removed_refs))

    # ------------------------------------------------------------------
    # 4. Hard-delete unowned artifacts.
    # ------------------------------------------------------------------
    if unowned_node_ids:
        await graph_engine.delete_nodes(unowned_node_ids)

    if unowned_edges:
        await graph_engine.delete_edge_triples(unowned_edges)
        from cognee.modules.graph.methods.sync_edge_indexes import (
            delete_edge_instances,
            sync_edge_types,
        )

        await delete_edge_instances(vector_engine, [edge_data[edge] for edge in unowned_edges])
        await sync_edge_types(
            graph_engine,
            vector_engine,
            [edge.relationship_name for edge in unowned_edges],
        )

    # ------------------------------------------------------------------
    # 5. Post-delete cleanup parity with delete_from_graph_and_vector
    #    (best-effort, non-fatal).
    # ------------------------------------------------------------------
    await _cleanup_orphaned_nodeset_tags(graph_engine, vector_engine, unowned_node_ids, node_data)


async def _cleanup_orphaned_nodeset_tags(
    graph_engine,
    vector_engine,
    unowned_node_ids: list[str],
    node_data: dict[str, NodeDeleteData],
) -> None:
    """Strip NodeSet tags belonging to deleted NodeSet nodes from surviving rows."""
    removed_nodeset_tags: set[str] = set()
    for node_id in unowned_node_ids:
        data = node_data[node_id]
        if data.node_type == "NodeSet":
            label = data.node_properties.get("name")
            if label:
                removed_nodeset_tags.add(label)

    if not removed_nodeset_tags:
        return

    tags_to_remove = sorted(removed_nodeset_tags)
    try:
        await graph_engine.remove_belongs_to_set_tags(tags_to_remove)
    except Exception as error:
        logger.warning("Graph NodeSet tag cleanup failed (non-fatal): %s", error)

    try:
        await vector_engine.remove_belongs_to_set_tags(tags_to_remove)
    except Exception as error:
        logger.warning("Vector NodeSet tag cleanup failed (non-fatal): %s", error)
