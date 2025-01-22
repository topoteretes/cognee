from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk


async def remove_disconnected_chunks(data_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """
    Removes disconnected or obsolete chunks from the graph database.

    Notes:
        - Obsolete chunks are defined as chunks with no "next_chunk" predecessor.
        - Fully disconnected nodes are identified and deleted separately.
        - This function assumes that the graph database is properly initialized and accessible.
    """
    graph_engine = await get_graph_engine()

    document_ids = set((data_chunk.document_id for data_chunk in data_chunks))

    obsolete_chunk_ids = []

    for document_id in document_ids:
        chunks = await graph_engine.get_successors(document_id, edge_label="has_chunk")

        for chunk in chunks:
            previous_chunks = await graph_engine.get_predecessors(
                chunk["uuid"], edge_label="next_chunk"
            )

            if len(previous_chunks) == 0:
                obsolete_chunk_ids.append(chunk["uuid"])

    if len(obsolete_chunk_ids) > 0:
        await graph_engine.delete_nodes(obsolete_chunk_ids)

    disconnected_nodes = await graph_engine.get_disconnected_nodes()
    if len(disconnected_nodes) > 0:
        await graph_engine.delete_nodes(disconnected_nodes)

    return data_chunks
