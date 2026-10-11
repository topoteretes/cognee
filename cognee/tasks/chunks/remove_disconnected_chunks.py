from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk


def _node_id(node: dict | str) -> str:
    """Return a node's id from a ``get_predecessors``/``get_successors`` item.

    Adapters disagree on the shape: Ladybug returns ``{"id": ...}`` dicts,
    Neo4j and Neptune return bare id strings.
    """
    if isinstance(node, dict):
        return str(node["id"])
    return str(node)


async def remove_disconnected_chunks(data_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """
    Removes chunks of the given documents that the graph still stores but the
    given batch no longer contains — leftovers of a previous chunking.

    Notes:
        - A chunk hangs off its document through an ``is_part_of`` edge (chunk
          -> document) and is identified by its ``id``; the lookup follows the
          edges and ids the current pipeline actually writes.
        - Only stored chunks of the given documents are deleted. Nodes
          unrelated to the batch — including isolated nodes elsewhere in the
          graph — are left alone.
        - This function assumes that the graph database is properly initialized and accessible.
    """
    graph_engine = await get_graph_engine()

    current_chunk_ids = {str(data_chunk.id) for data_chunk in data_chunks}

    document_ids = {
        str(data_chunk.document_id) if data_chunk.document_id else str(data_chunk.is_part_of.id)
        for data_chunk in data_chunks
    }

    obsolete_chunk_ids = set()

    for document_id in document_ids:
        stored_chunks = await graph_engine.get_predecessors(document_id, edge_label="is_part_of")

        for stored_chunk in stored_chunks:
            stored_chunk_id = _node_id(stored_chunk)
            if stored_chunk_id not in current_chunk_ids:
                obsolete_chunk_ids.add(stored_chunk_id)

    if len(obsolete_chunk_ids) > 0:
        await graph_engine.delete_nodes(list(obsolete_chunk_ids))

    return data_chunks
