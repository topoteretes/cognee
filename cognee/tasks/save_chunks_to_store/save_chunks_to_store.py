from cognee.infrastructure.databases.vector import DataPoint, get_vector_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.chunking import DocumentChunk

async def save_chunks_to_store(data_chunks: list[DocumentChunk], collection_name: str):
    if len(data_chunks) == 0:
        return data_chunks

    vector_engine = get_vector_engine()
    graph_engine = await get_graph_engine()

    # Remove and unlink existing chunks
    if await vector_engine.has_collection(collection_name):
        existing_chunks = [DocumentChunk.parse_obj(chunk.payload) for chunk in (await vector_engine.retrieve(
            collection_name,
            [str(chunk.chunk_id) for chunk in data_chunks],
        ))]

        if len(existing_chunks) > 0:
            await vector_engine.delete_data_points(collection_name, [str(chunk.chunk_id) for chunk in existing_chunks])

            await graph_engine.remove_connection_to_successors_of([chunk.chunk_id for chunk in existing_chunks], "next_chunk")
            await graph_engine.remove_connection_to_predecessors_of([chunk.chunk_id for chunk in existing_chunks], "has_chunk")
    else:
        await vector_engine.create_collection(collection_name, payload_schema = DocumentChunk)

    # Add to vector storage
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
    chunk_nodes = []
    chunk_edges = []

    for chunk in data_chunks:
        chunk_nodes.append((
            str(chunk.chunk_id),
            dict(
                uuid = str(chunk.chunk_id),
                chunk_id = str(chunk.chunk_id),
                document_id = str(chunk.document_id),
                word_count = chunk.word_count,
                chunk_index = chunk.chunk_index,
                cut_type = chunk.cut_type,
            )
        ))

        chunk_edges.append((
            str(chunk.document_id),
            str(chunk.chunk_id),
            "has_chunk",
            dict(
                relationship_name = "has_chunk",
                source_node_id = str(chunk.document_id),
                target_node_id = str(chunk.chunk_id),
            ),
        ))

        previous_chunk_id = get_previous_chunk_id(data_chunks, chunk)

        if previous_chunk_id is not None:
            chunk_edges.append((
                str(previous_chunk_id),
                str(chunk.chunk_id),
                "next_chunk",
                dict(
                    relationship_name = "next_chunk",
                    source_node_id = str(previous_chunk_id),
                    target_node_id = str(chunk.chunk_id),
                ),
            ))

    await graph_engine.add_nodes(chunk_nodes)
    await graph_engine.add_edges(chunk_edges)

    return data_chunks


def get_previous_chunk_id(document_chunks: list[DocumentChunk], current_chunk: DocumentChunk) -> DocumentChunk:
    if current_chunk.chunk_index == 0:
        return current_chunk.document_id

    for chunk in document_chunks:
        if str(chunk.document_id) == str(current_chunk.document_id) \
            and chunk.chunk_index == current_chunk.chunk_index - 1:
            return chunk.chunk_id

    return None
