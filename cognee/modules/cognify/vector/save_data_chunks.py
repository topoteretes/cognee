
from cognee.infrastructure.databases.vector import DataPoint, get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.processing.chunk_types.DocumentChunk import DocumentChunk

async def save_data_chunks(data_chunks: list[DocumentChunk], collection_name: str):
    # Add to vector storage
    vector_engine = get_vector_engine()

    await vector_engine.create_collection(collection_name, payload_schema = DocumentChunk)

    await vector_engine.create_data_points(
        collection_name,
        [
            DataPoint[DocumentChunk](
                id = str(chunk.chunk_id),
                payload = chunk,
                embed_field = "text",
            ) for chunk in data_chunks
        ],
    )

    # Add to graph storage
    graph_engine = await get_graph_engine()

    await graph_engine.add_nodes([(str(chunk.chunk_id), chunk) for chunk in data_chunks])
    await graph_engine.add_edges([(
        str(chunk.document_id),
        str(chunk.chunk_id),
        "has_chunks",
        dict(relationship_name = "has_chunks"),
    ) for chunk in data_chunks])

    chunk_connections = [(
        str(data_chunks[chunk_index - 1].chunk_id),
        str(chunk.chunk_id),
        "next_chunk",
        dict(relationship_name = "next_chunk"),
    ) for (chunk_index, chunk) in enumerate(data_chunks) if chunk_index > 0]

    await graph_engine.add_edges(chunk_connections)

    return data_chunks
