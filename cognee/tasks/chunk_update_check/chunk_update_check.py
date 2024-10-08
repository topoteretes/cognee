from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.chunking import DocumentChunk


async def chunk_update_check(data_chunks: list[DocumentChunk], collection_name: str) -> list[DocumentChunk]:
    vector_engine = get_vector_engine()

    if not await vector_engine.has_collection(collection_name):
        # If collection doesn't exist, all data_chunks are new
        return data_chunks

    existing_chunks = await vector_engine.retrieve(
        collection_name,
        [str(chunk.chunk_id) for chunk in data_chunks],
    )

    existing_chunks_map = {chunk.id: chunk.payload for chunk in existing_chunks}

    affected_data_chunks = []

    for chunk in data_chunks:
        if chunk.chunk_id not in existing_chunks_map or \
        chunk.text != existing_chunks_map[chunk.chunk_id]["text"]:
            affected_data_chunks.append(chunk)

    return affected_data_chunks
