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
from cognee.modules.engine.utils.generate_node_id import generate_node_id
from cognee.shared.logging_utils import get_logger

logger = get_logger("delete_chunks_incremental")

# Collections the deleted node kinds are embedded in.
_VECTOR_COLLECTIONS = ["DocumentChunk_text", "TextSummary_text", "Entity_name"]


def _triplet_id(edge: dict) -> str:
    """Triplet-embedding id for an edge (mirrors delete_from_graph_and_vector)."""
    return str(
        generate_node_id(
            str(edge.get("source_node_id"))
            + _relationship_name(edge)
            + str(edge.get("target_node_id"))
        )
    )


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

    # 1. Collect entity candidates (everything a deleted chunk `contains`) and
    #    the triplet-embedding ids of every edge that dies with a deleted node.
    candidates = set()
    doomed_triplet_ids = set()
    for chunk_id in deleting:
        for _source, edge, _target in await graph_engine.get_connections(chunk_id):
            doomed_triplet_ids.add(_triplet_id(edge))
            if _relationship_name(edge) != "contains":
                continue
            if str(edge.get("source_node_id")) == chunk_id:
                candidates.add(str(edge.get("target_node_id")))

    # 2. Chunk-local orphan check: an entity dies only when every chunk that
    #    contains it is in the deleted set.
    orphan_entities = []
    candidate_type_ids = set()
    for entity_id in candidates:
        entity_connections = await graph_engine.get_connections(entity_id)
        containing_chunks = {
            str(edge.get("source_node_id"))
            for _source, edge, _target in entity_connections
            if _relationship_name(edge) == "contains"
            and str(edge.get("target_node_id")) == entity_id
        }
        if containing_chunks <= deleting:
            orphan_entities.append(entity_id)
            for _source, edge, _target in entity_connections:
                doomed_triplet_ids.add(_triplet_id(edge))
                if (
                    _relationship_name(edge) == "is_a"
                    and str(edge.get("source_node_id")) == entity_id
                ):
                    candidate_type_ids.add(str(edge.get("target_node_id")))

    # 3. Summaries are keyed deterministically off the chunk id (summarize_text.py).
    summary_ids = [str(uuid5(UUID(chunk), "TextSummary")) for chunk in deleting]

    doomed = orphan_entities + summary_ids + sorted(deleting)
    await graph_engine.delete_nodes(doomed)

    vector_engine = await get_vector_engine_async()
    for collection in _VECTOR_COLLECTIONS:
        if await vector_engine.has_collection(collection):
            await vector_engine.delete_data_points(collection, doomed)

    # Triplet embeddings of edges that died with the deleted nodes — without
    # this, facts from deleted chunks stay reachable through triplet search.
    if doomed_triplet_ids and await vector_engine.has_collection("Triplet_text"):
        await vector_engine.delete_data_points("Triplet_text", sorted(doomed_triplet_ids))

    # EntityType nodes referenced only by deleted entities are orphans too.
    orphan_type_ids = []
    for type_id in candidate_type_ids:
        remaining = [
            edge
            for _source, edge, _target in await graph_engine.get_connections(type_id)
            if _relationship_name(edge) == "is_a" and str(edge.get("target_node_id")) == type_id
        ]
        if not remaining:
            orphan_type_ids.append(type_id)
    if orphan_type_ids:
        await graph_engine.delete_nodes(orphan_type_ids)
        if await vector_engine.has_collection("EntityType_name"):
            await vector_engine.delete_data_points("EntityType_name", orphan_type_ids)
        doomed.extend(orphan_type_ids)

    logger.info(
        "incremental delete: %d chunks, %d summaries, %d orphaned entities",
        len(deleting),
        len(summary_ids),
        len(orphan_entities),
    )
    return doomed
