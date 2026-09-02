from typing import Optional
from uuid import UUID

from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.unified.provenance_delete_planner import (
    SourceRefRemovalResult,
)


async def try_delete_data_by_graph_provenance(
    dataset_id: UUID, data_id: UUID
) -> Optional[SourceRefRemovalResult]:
    """Delete a data item's graph-provenance refs when the graph is marked.

    Authorization belongs to the public callers. This helper only answers whether
    the graph-provenance path handled the delete: None means "not handled";
    a (always truthy) SourceRefRemovalResult means handled, carrying the
    hard-deleted node/edge identities for derived-cache invalidation.
    """
    unified = await get_unified_engine()
    if not unified.supports_graph_provenance_delete():
        return None

    if not await stores_provenance_in_graph(unified.graph):
        return None

    # Document scope covers BOTH ref versions: the v1 doc key and every v2
    # chunk key the document's chunks own — a v1-only delete would strand
    # chunk-owned artifacts.
    return await unified.delete_by_document(str(dataset_id), str(data_id))
