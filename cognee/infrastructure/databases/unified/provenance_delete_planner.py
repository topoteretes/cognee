"""Graph-native delete planner (Part 2).

Given delete-data snapshots read off the graph (Part 0 ``NodeDeleteData`` /
``EdgeDeleteData``) and a target ref to remove, this planner decides which
artifacts are now unowned (their last owning ref is gone → hard delete) versus
which survive (another ref still owns them → detach), computes the vector ids to
clean **only from the snapshots** (never from relational rows), and executes a
retry-safe ordering:

    read provenance + payloads  (done by the caller, passed in)
    → compute vector ids        (from snapshots only)
    → delete vectors            (unowned artifacts)
    → detach surviving artifacts (remove the ref in the graph)
    → delete unowned artifacts  (graph)

Because every graph mutation happens *after* the vector deletes, a failed vector
delete leaves graph provenance untouched and a retry re-reads the same state and
converges. The detach/delete steps are idempotent, so a retry after a partial
graph mutation also converges to the clean final state. Unsupported-capability
errors from the reads propagate to the caller before any mutation.

Vector id derivation mirrors ``delete_from_graph_and_vector`` exactly so the two
paths target identical collections/ids:
  - node: collection ``f"{node_type}_{field}"`` for each indexed field, id = node_id
  - edge: ``EdgeType.id_for(edge_retrieval_text)`` in ``EdgeType_relationship_name``
          and ``generate_node_id(source+relationship+target)`` in ``Triplet_text``
"""

from __future__ import annotations

from typing import Callable, Dict, List

from cognee.modules.engine.utils import generate_node_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.provenance.results import ProvenanceDeleteResult
from cognee.modules.graph.provenance.snapshots import EdgeDeleteData, NodeDeleteData
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.shared.logging_utils import get_logger

logger = get_logger("provenance_delete_planner")

# Survival predicate: given a snapshot whose target ref(s) have been removed,
# return True if the artifact should survive (be detached) rather than deleted.
NodeSurvives = Callable[[NodeDeleteData], bool]
EdgeSurvives = Callable[[EdgeDeleteData], bool]


def _edge_retrieval_text(edge: EdgeDeleteData) -> str:
    return get_edge_retrieval_text(edge.edge_retrieval_text, edge.identity.relationship_name)


def _node_vector_ids(nodes: List[NodeDeleteData]) -> Dict[str, List[str]]:
    """Map vector collection name → node ids to delete, from snapshots only."""
    collections: Dict[str, List[str]] = {}
    for node in nodes:
        for field in node.indexed_fields:
            collections.setdefault(f"{node.node_type}_{field}", []).append(str(node.node_id))
    return collections


def _edge_type_vector_ids(edges: List[EdgeDeleteData]) -> List[str]:
    ids = []
    for edge in edges:
        text = _edge_retrieval_text(edge)
        if text:
            ids.append(str(EdgeType.id_for(text)))
    return ids


def _triplet_vector_ids(edges: List[EdgeDeleteData]) -> List[str]:
    ids = []
    for edge in edges:
        identity = edge.identity
        ids.append(
            str(
                generate_node_id(
                    str(identity.source_node_id)
                    + identity.relationship_name
                    + str(identity.target_node_id)
                )
            )
        )
    return ids


async def execute_ref_removal(
    graph_engine,
    vector_engine,
    *,
    nodes: List[NodeDeleteData],
    edges: List[EdgeDeleteData],
    property_key: str,
    refs_to_remove: List[str],
    node_survives: NodeSurvives,
    edge_survives: EdgeSurvives,
) -> ProvenanceDeleteResult:
    """Remove ``refs_to_remove`` (stored under ``property_key``) from the given
    artifacts, deleting unowned ones from graph + vector and detaching survivors.

    The caller supplies the already-read snapshots and the survival predicates
    that encode the operation's ownership rule (source-ref delete, dataset
    delete, or run rollback).
    """
    surviving_nodes = [n for n in nodes if node_survives(n)]
    unowned_nodes = [n for n in nodes if not node_survives(n)]
    surviving_edges = [e for e in edges if edge_survives(e)]
    unowned_edges = [e for e in edges if not edge_survives(e)]

    if not nodes and not edges:
        return ProvenanceDeleteResult()

    # 1) Compute vector ids for the unowned artifacts (snapshots only).
    node_vector_ids = _node_vector_ids(unowned_nodes)
    edge_type_ids = _edge_type_vector_ids(unowned_edges)
    triplet_ids = _triplet_vector_ids(unowned_edges)

    # 2) Delete vectors FIRST so a failure here leaves graph provenance intact.
    for collection, ids in node_vector_ids.items():
        if ids:
            await vector_engine.delete_data_points(collection, ids)
    if edge_type_ids:
        await vector_engine.delete_data_points("EdgeType_relationship_name", edge_type_ids)
    if triplet_ids:
        try:
            await vector_engine.delete_data_points("Triplet_text", triplet_ids)
        except Exception:
            # Triplet collection may not exist if triplet embedding was never enabled.
            logger.debug("Triplet_text collection absent; skipping triplet vector delete.")

    # 3) Detach the removed refs from surviving artifacts (idempotent).
    if surviving_nodes:
        await graph_engine.detach_provenance_refs_from_nodes(
            [str(n.node_id) for n in surviving_nodes], property_key, list(refs_to_remove)
        )
    if surviving_edges:
        await graph_engine.detach_provenance_refs_from_edges(
            [e.identity for e in surviving_edges], property_key, list(refs_to_remove)
        )

    # 4) Delete unowned artifacts from the graph (idempotent).
    if unowned_nodes:
        await graph_engine.delete_nodes([str(n.node_id) for n in unowned_nodes])
    if unowned_edges:
        await graph_engine.delete_edges([e.identity for e in unowned_edges])

    return ProvenanceDeleteResult(
        nodes_deleted=len(unowned_nodes),
        edges_deleted=len(unowned_edges),
        nodes_detached=len(surviving_nodes),
        edges_detached=len(surviving_edges),
    )


__all__ = ["execute_ref_removal"]
