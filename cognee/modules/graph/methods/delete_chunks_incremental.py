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


async def delete_chunks_incremental(chunk_ids: List[str], dataset_id, data_id) -> None:
    """Remove each retired chunk owner through the last-owner deletion planner."""
    unified = await get_unified_engine()
    for chunk_id in dict.fromkeys(str(chunk_id) for chunk_id in chunk_ids):
        source_ref = make_chunk_source_ref_key(
            UUID(str(dataset_id)),
            UUID(str(data_id)),
            UUID(chunk_id),
        )
        await unified.delete_by_source_ref(source_ref)
