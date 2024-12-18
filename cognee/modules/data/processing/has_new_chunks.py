from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.chunking import DocumentChunk


async def has_new_chunks(
    data_chunks: list[DocumentChunk], collection_name: str
) -> list[DocumentChunk]:
    vector_engine = get_vector_engine()

    if not await vector_engine.has_collection(collection_name):
        # There is no collection created,
        # so no existing chunks, all chunks are new.
        return True

    existing_chunks = await vector_engine.retrieve(
        collection_name,
        [str(chunk.chunk_id) for chunk in data_chunks],
    )

    if len(existing_chunks) == 0:
        # If we don't find any existing chunk,
        # all chunks are new.
        return True

    existing_chunks_map = {chunk.id: chunk.payload for chunk in existing_chunks}

    new_data_chunks = [
        chunk
        for chunk in data_chunks
        if chunk.chunk_id not in existing_chunks_map
        or chunk.text != existing_chunks_map[chunk.chunk_id]["text"]
    ]

    return len(new_data_chunks) > 0
