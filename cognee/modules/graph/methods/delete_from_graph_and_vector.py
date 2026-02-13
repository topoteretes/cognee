from typing import Dict, List

from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine
from cognee.modules.graph.legacy.mark_ledger_as_deleted import (
    mark_ledger_edges_as_deleted,
    mark_ledger_nodes_as_deleted,
)
from cognee.modules.graph.models import Node, Edge
from cognee.modules.engine.utils import generate_node_id


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

        await vector_engine.delete_data_points(
            "EdgeType_relationship_name",
            [str(edge.slug) for edge in unique_edges],
        )

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

    # Mark ledger entries as deleted
    await mark_ledger_nodes_as_deleted([node.slug for node in non_legacy_nodes])
    if non_legacy_edges:
        await mark_ledger_edges_as_deleted([edge.relationship_name for edge in non_legacy_edges])
