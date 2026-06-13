from typing import Dict, List

from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine
from cognee.modules.graph.legacy.mark_ledger_as_deleted import (
    mark_ledger_edges_as_deleted,
    mark_ledger_nodes_as_deleted,
)
from cognee.modules.graph.models import Node, Edge
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.modules.engine.utils import generate_node_id
from cognee.modules.engine.utils.generate_edge_id import generate_edge_id
from cognee.shared.logging_utils import get_logger

logger = get_logger("delete_from_graph_and_vector")


def _get_deleted_edge_retrieval_text(edge: Edge) -> str:
    attributes = edge.attributes or {}
    return get_edge_retrieval_text(attributes.get("edge_text"), edge.relationship_name)


def _get_remaining_edge_retrieval_text(edge) -> str:
    properties = edge[3] if len(edge) > 3 and isinstance(edge[3], dict) else {}
    return get_edge_retrieval_text(properties.get("edge_text"), edge[2])


async def delete_from_graph_and_vector(
    affected_nodes: List[Node],
    affected_edges: List[Edge],
    is_legacy_node: List[bool],
    is_legacy_edge: List[bool],
) -> None:
    """Delete non-legacy nodes/edges from graph DB, vector DB, and mark ledger entries.

    Shared logic used by both data-level and dataset-level deletion flows.
    Deduplicates by slug before issuing deletes to avoid redundant I/O.
    """
    non_legacy_nodes = [
        node for index, node in enumerate(affected_nodes) if not is_legacy_node[index]
    ]
    non_legacy_edges = [
        edge for index, edge in enumerate(affected_edges) if not is_legacy_edge[index]
    ]

    # Deduplicate nodes by slug to avoid redundant graph/vector deletes
    seen_node_slugs = set()
    unique_nodes = []
    for node in non_legacy_nodes:
        if node.slug not in seen_node_slugs:
            seen_node_slugs.add(node.slug)
            unique_nodes.append(node)

    # Capture NodeSet names before their ledger/graph rows are deleted so
    # we can strip these tags from any surviving shared DataPoint afterwards.
    # When a dataset is deleted in single-tenant mode its uniquely-owned
    # NodeSet nodes are in `unique_nodes`; shared entities tagged with
    # those names are not. The detag pass in `remove_belongs_to_set_tags`
    # keeps the stored `belongs_to_set` arrays consistent with the graph
    # after the hard delete.
    removed_nodeset_tags = {
        node.label for node in unique_nodes if node.type == "NodeSet" and node.label
    }

    graph_engine = None

    # Delete from graph DB
    if unique_nodes:
        graph_engine = await get_graph_engine()
        await graph_engine.delete_nodes([str(node.slug) for node in unique_nodes])

    # Delete from vector DB - group by collection
    affected_vector_collections: Dict[str, List[Node]] = {}
    for node in unique_nodes:
        for indexed_field in node.indexed_fields:
            collection_name = f"{node.type}_{indexed_field}"
            affected_vector_collections.setdefault(collection_name, []).append(node)

    vector_engine = get_vector_engine()
    for collection, nodes in affected_vector_collections.items():
        await vector_engine.delete_data_points(collection, [str(node.slug) for node in nodes])

    # Delete edge embeddings and triplets
    if non_legacy_edges:
        # Deduplicate edges by slug
        seen_edge_slugs = set()
        unique_edges = []
        for edge in non_legacy_edges:
            if edge.slug not in seen_edge_slugs:
                seen_edge_slugs.add(edge.slug)
                unique_edges.append(edge)

        edge_type_ids = []
        for edge in unique_edges:
            edge_text = _get_deleted_edge_retrieval_text(edge)
            if edge_text:
                edge_type_ids.append(str(generate_edge_id(edge_id=edge_text)))

        await vector_engine.delete_data_points("EdgeType_relationship_name", edge_type_ids)

        triplet_ids = [
            str(
                generate_node_id(
                    str(edge.source_node_id)
                    + edge.relationship_name
                    + str(edge.destination_node_id)
                )
            )
            for edge in unique_edges
        ]
        if triplet_ids:
            try:
                await vector_engine.delete_data_points("Triplet_text", triplet_ids)
            except Exception:
                # Triplet collection might not exist if triplet embedding was never enabled
                pass

    # Clean up orphaned EdgeType nodes from the graph.
    # EdgeType nodes are created by index_graph_edges for each unique edge
    # retrieval text. When edges are deleted, check if any edges with that
    # retrieval text remain in the graph. If not, remove the EdgeType node.
    if non_legacy_edges:
        deleted_edge_texts = set()
        for edge in unique_edges:
            edge_text = _get_deleted_edge_retrieval_text(edge)
            if edge_text:
                deleted_edge_texts.add(edge_text)

        try:
            if not graph_engine:
                graph_engine = await get_graph_engine()
            _, remaining_edges = await graph_engine.get_graph_data()
            remaining_edge_texts = set()
            for edge in remaining_edges:
                edge_text = _get_remaining_edge_retrieval_text(edge)
                if edge_text:
                    remaining_edge_texts.add(edge_text)

            orphaned_edge_type_ids = [
                str(generate_edge_id(edge_id=edge_text))
                for edge_text in deleted_edge_texts
                if edge_text not in remaining_edge_texts
            ]

            if orphaned_edge_type_ids:
                await graph_engine.delete_nodes(orphaned_edge_type_ids)
                logger.info(
                    "Deleted %d orphaned EdgeType node(s)",
                    len(orphaned_edge_type_ids),
                )
        except Exception as e:
            logger.warning("EdgeType cleanup failed (non-fatal): %s", e)

    # Strip now-orphaned NodeSet tags from surviving rows/nodes so shared
    # entities stop advertising membership in a dataset that no longer
    # exists. Runs after the hard-delete pass so freshly-deleted rows
    # aren't touched redundantly.
    if removed_nodeset_tags:
        tags_to_remove = sorted(removed_nodeset_tags)
        try:
            if not graph_engine:
                graph_engine = await get_graph_engine()
            await graph_engine.remove_belongs_to_set_tags(tags_to_remove)
        except Exception as e:
            logger.warning("Graph NodeSet tag cleanup failed (non-fatal): %s", e)

        try:
            await vector_engine.remove_belongs_to_set_tags(tags_to_remove)
        except Exception as e:
            logger.warning("Vector NodeSet tag cleanup failed (non-fatal): %s", e)

    # Mark ledger entries as deleted
    await mark_ledger_nodes_as_deleted([node.slug for node in non_legacy_nodes])
    if non_legacy_edges:
        await mark_ledger_edges_as_deleted([edge.relationship_name for edge in non_legacy_edges])
