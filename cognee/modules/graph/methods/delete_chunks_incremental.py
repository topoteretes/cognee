"""Delete chunk-owned output through the shared provenance planner."""

from typing import List
from uuid import UUID

from cognee.infrastructure.databases.provenance import make_chunk_source_ref_key
from cognee.infrastructure.databases.unified import get_unified_engine


def edge_endpoints(source: dict, edge: dict, target: dict) -> tuple:
    """Return adapter-tolerant true edge endpoints."""
    source_id = edge.get("source_node_id") or source.get("id")
    target_id = edge.get("target_node_id") or target.get("id")
    return str(source_id), str(target_id)


async def delete_chunks_incremental(chunk_ids: List[str], dataset_id, data_id):
    """Retire every given chunk's ownership in ONE last-owner planner pass.

    Artifacts owned only by retired chunks are hard-deleted (graph and
    vectors); output shared with surviving chunks loses just the retired
    keys. One pass, not one per chunk: the planner's post-delete cleanup
    loads the whole graph to find orphaned EdgeTypes, so per-chunk calls
    made deletion cost grow with graph size times chunk count.
    """
    keys = [
        make_chunk_source_ref_key(UUID(str(dataset_id)), UUID(str(data_id)), UUID(chunk_id))
        for chunk_id in dict.fromkeys(str(chunk_id) for chunk_id in chunk_ids)
    ]
    if not keys:
        return None
    unified = await get_unified_engine()
    return await unified.delete_by_source_refs(keys)
