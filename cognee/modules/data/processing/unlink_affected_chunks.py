from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from .chunk_types import DocumentChunk

async def unlink_affected_chunks(data_chunks: list[DocumentChunk], collection_name: str) -> list[DocumentChunk]:
    vector_engine = get_vector_engine()

    if not await vector_engine.has_collection(collection_name):
        # If collection doesn't exist, all data_chunks are new
        return data_chunks

    existing_chunks = await vector_engine.retrieve(
        collection_name,
        [str(chunk.chunk_id) for chunk in data_chunks],
    )

    existing_chunks_map = {chunk.id: chunk.payload for chunk in existing_chunks}

    affected_data_chunks = [
        chunk for chunk in data_chunks \
            if chunk.chunk_id in existing_chunks_map
    ]

    graph_engine = await get_graph_engine()
    await graph_engine.remove_connection_to_successors_of([chunk.chunk_id for chunk in affected_data_chunks], "next_chunk")
    await graph_engine.remove_connection_to_predecessors_of([chunk.chunk_id for chunk in affected_data_chunks], "has_chunk")

    return data_chunks
