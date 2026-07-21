from uuid import UUID

from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.modules.graph.legacy.has_nodes_in_legacy_ledger import has_nodes_in_legacy_ledger
from cognee.modules.graph.legacy.has_edges_in_legacy_ledger import has_edges_in_legacy_ledger
from cognee.modules.graph.methods import (
    delete_dataset_related_edges,
    delete_dataset_related_nodes,
    get_dataset_related_nodes,
    get_dataset_related_edges,
    get_global_dataset_related_nodes,
    get_global_dataset_related_edges,
)
from cognee.modules.graph.methods.delete_from_graph_and_vector import (
    delete_from_graph_and_vector,
)
from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset
from cognee.modules.users.methods.get_user import get_user


def _invalidate_bm25_cache(dataset_id: UUID) -> None:
    from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever

    BM25ChunksRetriever.invalidate_cache(str(dataset_id))


async def delete_dataset_nodes_and_edges(dataset_id: UUID, user_id: UUID) -> None:
    user = await get_user(user_id)
    # Check if user has delete permission for the dataset before proceeding with deletion of related graph/vector nodes and edges.
    dataset = await get_authorized_dataset(user, dataset_id, "delete")
    dataset_id = dataset.id

    # Graph-provenance graphs carry provenance in the graph (no relational ledger
    # rows). Route them through the unified boundary, which removes the
    # dataset's source refs and hard-deletes any artifact left unowned.
    unified = await get_unified_engine()
    if unified.supports_graph_provenance_delete():
        graph_engine = unified.graph
        if await stores_provenance_in_graph(graph_engine):
            await unified.delete_by_dataset_id(str(dataset_id))
            _invalidate_bm25_cache(dataset_id)
            return

    if backend_access_control_enabled():
        affected_nodes = await get_dataset_related_nodes(dataset_id)
        affected_edges = await get_dataset_related_edges(dataset_id) if affected_nodes else []
    else:
        affected_nodes = await get_global_dataset_related_nodes(dataset_id)
        affected_edges = (
            await get_global_dataset_related_edges(dataset_id) if affected_nodes else []
        )

    if affected_nodes:
        is_legacy_node = await has_nodes_in_legacy_ledger(affected_nodes)
        is_legacy_edge = await has_edges_in_legacy_ledger(affected_edges)

        await delete_from_graph_and_vector(
            affected_nodes, affected_edges, is_legacy_node, is_legacy_edge
        )

    # Always clean up relational ownership records, even when no unique
    # graph/vector nodes needed deletion (e.g. shared nodes across datasets).
    await delete_dataset_related_nodes(dataset_id)
    await delete_dataset_related_edges(dataset_id)
    _invalidate_bm25_cache(dataset_id)
