
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.processing.chunk_types.DocumentChunk import DocumentChunk


# from cognee.infrastructure.databases.vector import get_vector_engine


async def chunk_remove_disconnected_task(data_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    graph_engine = await get_graph_engine()

    document_ids = set((data_chunk.document_id for data_chunk in data_chunks))

    obsolete_chunk_ids = []

    for document_id in document_ids:
        chunk_ids = await graph_engine.get_successor_ids(document_id, edge_label = "has_chunk")

        for chunk_id in chunk_ids:
            previous_chunks = await graph_engine.get_predecessor_ids(chunk_id, edge_label = "next_chunk")

            if len(previous_chunks) == 0:
                obsolete_chunk_ids.append(chunk_id)

    if len(obsolete_chunk_ids) > 0:
        await graph_engine.delete_nodes(obsolete_chunk_ids)

    disconnected_nodes = await graph_engine.get_disconnected_nodes()
    if len(disconnected_nodes) > 0:
        await graph_engine.delete_nodes(disconnected_nodes)

    return data_chunks
