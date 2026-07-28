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


def edge_endpoints(source: dict, edge: dict, target: dict) -> tuple:
    """Adapter-tolerant (source_id, target_id) for a get_connections triple.

    The ladybug/kuzu adapter always puts the QUERIED node in the tuple's source
    slot but carries the true endpoints on the edge dict; Neo4j and the
    Postgres graph adapter carry true orientation in the tuple positions and
    omit endpoint ids from the edge. Prefer the edge, fall back to the slots.
    """
    source_id = edge.get("source_node_id") or source.get("id")
    target_id = edge.get("target_node_id") or target.get("id")
    return str(source_id), str(target_id)


def _triplet_id(source: dict, edge: dict, target: dict) -> str:
    """Triplet-embedding id for an edge (mirrors delete_from_graph_and_vector)."""
    source_id, target_id = edge_endpoints(source, edge, target)
    return str(generate_node_id(source_id + _relationship_name(edge) + target_id))


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
        for source, edge, target in await graph_engine.get_connections(chunk_id):
            source_id, target_id = edge_endpoints(source, edge, target)
            doomed_triplet_ids.add(_triplet_id(source, edge, target))
            if _relationship_name(edge) != "contains":
                continue
            if source_id == chunk_id:
                candidates.add(target_id)

    # 2. Chunk-local orphan check: an entity dies only when every chunk that
    #    contains it is in the deleted set.
    orphan_entities = []
    candidate_type_ids = set()
    for entity_id in candidates:
        entity_connections = await graph_engine.get_connections(entity_id)
        containing_chunks = set()
        for source, edge, target in entity_connections:
            source_id, target_id = edge_endpoints(source, edge, target)
            if _relationship_name(edge) == "contains" and target_id == entity_id:
                containing_chunks.add(source_id)
        if containing_chunks <= deleting:
            orphan_entities.append(entity_id)
            for source, edge, target in entity_connections:
                source_id, target_id = edge_endpoints(source, edge, target)
                doomed_triplet_ids.add(_triplet_id(source, edge, target))
                if _relationship_name(edge) == "is_a" and source_id == entity_id:
                    candidate_type_ids.add(target_id)

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
            for source, edge, target in await graph_engine.get_connections(type_id)
            if _relationship_name(edge) == "is_a"
            and edge_endpoints(source, edge, target)[1] == type_id
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
