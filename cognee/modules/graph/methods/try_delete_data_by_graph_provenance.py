from uuid import UUID

from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.unified import get_unified_engine


async def try_delete_data_by_graph_provenance(dataset_id: UUID, data_id: UUID) -> bool:
    """Delete a data item's graph-provenance refs when the graph is marked.

    Authorization belongs to the public callers. This helper only answers whether
    the graph-provenance path handled the delete.
    """
    unified = await get_unified_engine()
    if not unified.supports_graph_provenance_delete():
        return False

    if not await stores_provenance_in_graph(unified.graph):
        return False

    # Document scope covers BOTH ref versions: the v1 doc key and every v2
    # chunk key the document's chunks own — a v1-only delete would strand
    # chunk-owned artifacts.
    await unified.delete_by_document(str(dataset_id), str(data_id))
    return True
