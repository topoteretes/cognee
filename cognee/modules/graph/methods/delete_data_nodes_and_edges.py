from uuid import UUID

from cognee.context_global_variables import is_multi_user_support_possible
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
from cognee.modules.graph.methods.delete_from_graph_and_vector import (
    delete_from_graph_and_vector,
)


async def delete_data_nodes_and_edges(dataset_id: UUID, data_id: UUID, user_id: UUID) -> None:
    if is_multi_user_support_possible():
        affected_nodes = await get_data_related_nodes(dataset_id, data_id)
        if len(affected_nodes) == 0:
            return
        affected_edges = await get_data_related_edges(dataset_id, data_id)
    else:
        affected_nodes = await get_global_data_related_nodes(data_id)
        if len(affected_nodes) == 0:
            return
        affected_edges = await get_global_data_related_edges(data_id)

    is_legacy_node = await has_nodes_in_legacy_ledger(affected_nodes)
    is_legacy_edge = await has_edges_in_legacy_ledger(affected_edges)

    await delete_from_graph_and_vector(
        affected_nodes, affected_edges, is_legacy_node, is_legacy_edge
    )

    await delete_data_related_nodes(data_id, dataset_id)
    await delete_data_related_edges(data_id, dataset_id)
