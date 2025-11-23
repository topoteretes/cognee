from uuid import UUID
from typing import Dict, List

from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine
from cognee.infrastructure.environment.config.is_backend_access_control_enabled import (
    is_multi_user_support_possible,
)
from cognee.modules.graph.legacy.has_edges_in_legacy_ledger import has_edges_in_legacy_ledger
from cognee.modules.graph.legacy.has_nodes_in_legacy_ledger import has_nodes_in_legacy_ledger
from cognee.modules.graph.methods import (
    delete_data_related_edges,
    delete_data_related_nodes,
    get_data_related_nodes,
    get_data_related_edges,
    get_global_data_related_nodes,
    get_global_data_related_edges,
)


async def delete_data_nodes_and_edges(dataset_id: UUID, data_id: UUID, user_id: UUID) -> None:
    if is_multi_user_support_possible():
        affected_nodes = await get_data_related_nodes(dataset_id, data_id)

        if len(affected_nodes) == 0:
            return

        is_legacy_node = await has_nodes_in_legacy_ledger(affected_nodes)

        affected_relationships = await get_data_related_edges(dataset_id, data_id)
        is_legacy_relationship = await has_edges_in_legacy_ledger(affected_relationships)

        non_legacy_nodes = [
            node for index, node in enumerate(affected_nodes) if not is_legacy_node[index]
        ]

        graph_engine = await get_graph_engine()
        await graph_engine.delete_nodes([str(node.slug) for node in non_legacy_nodes])

        affected_vector_collections: Dict[str, List] = {}
        for node in non_legacy_nodes:
            for indexed_field in node.indexed_fields:
                collection_name = f"{node.type}_{indexed_field}"
                if collection_name not in affected_vector_collections:
                    affected_vector_collections[collection_name] = []
                affected_vector_collections[collection_name].append(node)

        vector_engine = get_vector_engine()
        for affected_collection, non_legacy_nodes in affected_vector_collections.items():
            await vector_engine.delete_data_points(
                affected_collection, [str(node.slug) for node in non_legacy_nodes]
            )

        if len(affected_relationships) > 0:
            non_legacy_relationships = [
                edge
                for index, edge in enumerate(affected_relationships)
                if not is_legacy_relationship[index]
            ]

            await vector_engine.delete_data_points(
                "EdgeType_relationship_name",
                [str(relationship.slug) for relationship in non_legacy_relationships],
            )

        await delete_data_related_nodes(data_id)
        await delete_data_related_edges(data_id)
    else:
        affected_nodes = await get_global_data_related_nodes(data_id)

        if len(affected_nodes) == 0:
            return

        is_legacy_node = await has_nodes_in_legacy_ledger(affected_nodes)

        affected_relationships = await get_global_data_related_edges(data_id)
        is_legacy_relationship = await has_edges_in_legacy_ledger(affected_relationships)

        non_legacy_nodes = [
            node for index, node in enumerate(affected_nodes) if not is_legacy_node[index]
        ]

        graph_engine = await get_graph_engine()
        await graph_engine.delete_nodes([str(node.slug) for node in non_legacy_nodes])

        affected_vector_collections: Dict[str, List] = {}
        for node in non_legacy_nodes:
            for indexed_field in node.indexed_fields:
                collection_name = f"{node.type}_{indexed_field}"
                if collection_name not in affected_vector_collections:
                    affected_vector_collections[collection_name] = []
                affected_vector_collections[collection_name].append(node)

        vector_engine = get_vector_engine()
        for affected_collection, non_legacy_nodes in affected_vector_collections.items():
            await vector_engine.delete_data_points(
                affected_collection, [str(node.slug) for node in non_legacy_nodes]
            )

        if len(affected_relationships) > 0:
            non_legacy_relationships = [
                edge
                for index, edge in enumerate(affected_relationships)
                if not is_legacy_relationship[index]
            ]

            await vector_engine.delete_data_points(
                "EdgeType_relationship_name",
                [str(relationship.slug) for relationship in non_legacy_relationships],
            )

        await delete_data_related_nodes(data_id)
        await delete_data_related_edges(data_id)
