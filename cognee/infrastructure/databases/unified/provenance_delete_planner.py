"""Retry-safe planner for graph-provenance source-ref removal.

Given snapshots of the matched nodes/edges and the set of source refs to remove
from each, the planner decides which artifacts become *unowned* (no owning
source ref remains -> hard delete) versus which merely *survive* (some ref
remains -> detach the targeted refs only). It then performs the removal in a
retry-safe order:

  1. delete per-edge triplet vectors for unowned edges (from snapshots only),
  2. ``remove_*_source_refs`` for the targeted refs on SURVIVING matched
     artifacts (idempotent),
  3. ``delete_edge_triples`` for the unowned edges (edge-first),
  4. retention check: an unowned-candidate node that is still an endpoint of any
     remaining relationship is NOT hard-deleted — backends implement node
     deletion as a detaching delete, which would destroy foreign-owned incident
     relationships. Such nodes only lose the targeted refs (idempotent) and keep
     their vectors,
  5. delete vectors for the remaining truly-unowned nodes, then ``delete_nodes``
     for them (they have no incident relationships at this point, so the
     backend's detach semantics cannot remove anything foreign-owned).

Unowned artifacts keep their source refs until their hard delete, so a failure
at any step leaves them rediscoverable by source ref and a retry converges. All
steps are individually idempotent.

Vector ids mirror ``delete_from_graph_and_vector``:
  - node -> collection ``f"{node_type}_{field}"`` for each indexed field,
    id = node_id;
  - edge -> ``EdgeType.id_for(edge_text)`` in ``EdgeType_relationship_name``;
    ``generate_node_id(source_id + relationship_name + target_id)`` in
    ``Triplet_text`` (best-effort; the collection may not exist).
"""

import dataclasses
from dataclasses import dataclass

from cognee.infrastructure.databases.provenance import (
    EdgeDeleteData,
    EdgeIdentity,
    NodeDeleteData,
)
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.shared.logging_utils import get_logger

logger = get_logger("provenance_delete_planner")


@dataclass
class SourceRefRemovalResult:
    """Artifacts hard-deleted by a source-ref removal (survivors are excluded).

    Consumed by session invalidation to find cached answers that used the
    deleted elements. Instances are always truthy — "handled, deleted nothing"
    must still read as handled.
    """

    deleted_node_ids: list[str] = dataclasses.field(default_factory=list)
    deleted_edges: list[EdgeIdentity] = dataclasses.field(default_factory=list)


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
) -> SourceRefRemovalResult:
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
    # 2. Delete per-edge triplet vectors for unowned edges (from snapshots
    #    only). Node vectors are deleted later, after the surviving-endpoint
    #    retention check decides which nodes are actually hard-deleted.
    # ------------------------------------------------------------------
    if unowned_edges:
        # Per-edge triplet vectors are tied to a single deleted edge instance, so
        # they are safe to delete here. EdgeType vectors are keyed by *shared*
        # relationship text and may still be used by a surviving edge, so they
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
    # 4. Hard-delete unowned edges FIRST. Node deletion is a detaching delete
    #    on graph backends, so deleting nodes before edges would also destroy
    #    incident relationships that still carry foreign ownership — including
    #    relationships that never matched the targeted refs at all.
    # ------------------------------------------------------------------
    if unowned_edges:
        await graph_engine.delete_edge_triples(unowned_edges)

    # ------------------------------------------------------------------
    # 5. Surviving-endpoint retention check. With the unowned edges gone, any
    #    relationship still incident to an unowned-candidate node is by
    #    definition foreign-owned and must survive; the node is retained as its
    #    endpoint, loses only the targeted refs, and keeps its vectors.
    # ------------------------------------------------------------------
    retained_node_ids: list[str] = []
    deletable_node_ids: list[str] = []
    for node_id in unowned_node_ids:
        incident_edges = await graph_engine.get_edges(node_id)
        if incident_edges:
            retained_node_ids.append(node_id)
        else:
            deletable_node_ids.append(node_id)

    if retained_node_ids:
        logger.info(
            "Retained %d unowned node(s) still incident to surviving relationships",
            len(retained_node_ids),
        )

    retained_by_removed_refs: dict[tuple[str, ...], list[str]] = {}
    for node_id in retained_node_ids:
        removed = refs_by_node.get(node_id, [])
        if not removed:
            continue
        retained_by_removed_refs.setdefault(tuple(removed), []).append(node_id)

    for removed_refs, node_ids in retained_by_removed_refs.items():
        await graph_engine.remove_node_source_refs(node_ids, list(removed_refs))

    # ------------------------------------------------------------------
    # 6. Delete vectors for the truly-unowned nodes, then hard-delete them.
    #    They have no incident relationships at this point, so the backend's
    #    detach semantics cannot remove anything foreign-owned. Refs stay on
    #    the nodes until this delete, keeping a failed run rediscoverable.
    # ------------------------------------------------------------------
    node_vector_collections: dict[str, list[str]] = {}
    for node_id in deletable_node_ids:
        data = node_data[node_id]
        for field in data.indexed_fields:
            collection_name = f"{data.node_type}_{field}"
            node_vector_collections.setdefault(collection_name, []).append(node_id)

    for collection, ids in node_vector_collections.items():
        await _delete_vector_points(vector_engine, collection, ids)

    if deletable_node_ids:
        await graph_engine.delete_nodes(deletable_node_ids)

    # ------------------------------------------------------------------
    # 7. Post-delete cleanup parity with delete_from_graph_and_vector
    #    (best-effort, non-fatal). NodeSet tag cleanup only considers nodes
    #    that were actually hard-deleted; retained NodeSet nodes keep tags.
    # ------------------------------------------------------------------
    await _cleanup_orphaned_edge_types(graph_engine, vector_engine, unowned_edges, edge_data)
    await _cleanup_orphaned_nodeset_tags(graph_engine, vector_engine, deletable_node_ids, node_data)

    return SourceRefRemovalResult(
        deleted_node_ids=list(unowned_node_ids),
        deleted_edges=list(unowned_edges),
    )


async def _cleanup_orphaned_edge_types(
    graph_engine,
    vector_engine,
    unowned_edges: list[EdgeIdentity],
    edge_data: dict[EdgeIdentity, EdgeDeleteData],
) -> None:
    """Prune EdgeType nodes (and their vectors) whose text no longer appears.

    EdgeType artifacts are keyed by *shared* relationship text, so an EdgeType is
    only orphaned when no surviving edge in the graph still uses that text. We
    delete the graph node and the vector point together, off the same orphaned-
    text set, so a relationship that another edge still uses keeps both.
    """
    if not unowned_edges:
        return

    deleted_edge_texts: set[str] = set()
    for edge in unowned_edges:
        data = edge_data[edge]
        edge_text = get_edge_retrieval_text(data.edge_text, edge.relationship_name)
        if edge_text:
            deleted_edge_texts.add(edge_text)

    if not deleted_edge_texts:
        return

    try:
        _, remaining_edges = await graph_engine.get_graph_data()
        remaining_edge_texts: set[str] = set()
        for edge in remaining_edges:
            properties = edge[3] if len(edge) > 3 and isinstance(edge[3], dict) else {}
            edge_text = get_edge_retrieval_text(properties.get("edge_text"), edge[2])
            if edge_text:
                remaining_edge_texts.add(edge_text)

        orphaned_edge_texts = [
            edge_text for edge_text in deleted_edge_texts if edge_text not in remaining_edge_texts
        ]
        orphaned_edge_type_ids = [str(EdgeType.id_for(text)) for text in orphaned_edge_texts]

        if orphaned_edge_type_ids:
            await graph_engine.delete_nodes(orphaned_edge_type_ids)
            logger.info(
                "Deleted %d orphaned EdgeType node(s)",
                len(orphaned_edge_type_ids),
            )
            try:
                await vector_engine.delete_data_points(
                    "EdgeType_relationship_name", orphaned_edge_type_ids
                )
            except Exception as error:
                logger.warning("EdgeType vector cleanup failed (non-fatal): %s", error)
    except Exception as error:
        logger.warning("EdgeType cleanup failed (non-fatal): %s", error)


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
