"""Chunk-scoped deletion for incremental document updates.

Deletes a subset of a document's chunks from the graph and vector stores,
together with their summaries and any entity that is contained ONLY in the
deleted chunks. Unlike ``legacy_delete`` (document-scoped: an entity survives
only when another *document* references it), the orphan check here is
chunk-local: an entity survives when any chunk outside the deleted set —
same document or not — still contains it.
"""

from typing import List
from uuid import UUID, uuid5

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.shared.logging_utils import get_logger

logger = get_logger("delete_chunks_incremental")

# Collections the deleted node kinds are embedded in.
_VECTOR_COLLECTIONS = ["DocumentChunk_text", "TextSummary_text", "Entity_name"]


def _relationship_name(edge: dict) -> str:
    name = str(edge.get("relationship_name", ""))
    if "relationship_name:" in name:  # serialized property-bag form
        name = name.split("relationship_name:", 1)[1].split(";", 1)[0].strip()
    return name


async def delete_chunks_incremental(chunk_ids: List[str]) -> List[str]:
    """Delete the given chunk nodes, their summaries, and chunk-orphaned entities.

    Returns the ids of every node removed (used for vector cleanup, which is
    performed here as well).
    """
    graph_engine = await get_graph_engine()
    deleting = {str(chunk_id) for chunk_id in chunk_ids}

    # NOTE: get_connections returns the queried node in the "source" slot no
    # matter the direction — the true endpoints live on the edge dict.

    # 1. Collect entity candidates: everything a deleted chunk `contains`.
    candidates = set()
    for chunk_id in deleting:
        for _source, edge, _target in await graph_engine.get_connections(chunk_id):
            if _relationship_name(edge) != "contains":
                continue
            if str(edge.get("source_node_id")) == chunk_id:
                candidates.add(str(edge.get("target_node_id")))

    # 2. Chunk-local orphan check: an entity dies only when every chunk that
    #    contains it is in the deleted set.
    orphan_entities = []
    for entity_id in candidates:
        containing_chunks = {
            str(edge.get("source_node_id"))
            for _source, edge, _target in await graph_engine.get_connections(entity_id)
            if _relationship_name(edge) == "contains"
            and str(edge.get("target_node_id")) == entity_id
        }
        if containing_chunks <= deleting:
            orphan_entities.append(entity_id)

    # 3. Summaries are keyed deterministically off the chunk id (summarize_text.py).
    summary_ids = [str(uuid5(UUID(chunk), "TextSummary")) for chunk in deleting]

    doomed = orphan_entities + summary_ids + sorted(deleting)
    await graph_engine.delete_nodes(doomed)

    vector_engine = await get_vector_engine_async()
    for collection in _VECTOR_COLLECTIONS:
        if await vector_engine.has_collection(collection):
            await vector_engine.delete_data_points(collection, doomed)

    logger.info(
        "incremental delete: %d chunks, %d summaries, %d orphaned entities",
        len(deleting),
        len(summary_ids),
        len(orphan_entities),
    )
    return doomed
