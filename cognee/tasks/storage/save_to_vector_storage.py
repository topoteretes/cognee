from cognee.infrastructure.databases.vector import get_vector_engine, DataPoint

async def save_to_vector_storage(data_chunks: list, collection_name: str, embed_field: str):
    if len(data_chunks) == 0:
        return data_chunks

    if not all(isinstance(chunk, type(data_chunks[0])) for chunk in data_chunks):
        raise ValueError("All data chunks must be of the same type.")

    vector_engine = get_vector_engine()

    PayloadSchema = type(data_chunks[0])

    await vector_engine.create_collection(collection_name, payload_schema = PayloadSchema)

    await vector_engine.create_data_points(
        collection_name,
        [
            DataPoint[PayloadSchema](
                id = str(chunk.id),
                payload = parse_data(chunk, chunk_index),
                embed_field = embed_field,
            ) for (chunk_index, chunk) in enumerate(data_chunks)
        ],
    )

    return data_chunks

def parse_data(chunk, chunk_index: int) -> dict:
    from uuid import UUID

    data = {
        "chunk_index": chunk_index,
    }

    for key, value in vars(chunk).items():
        if isinstance(value, UUID):
            data[key] = str(value)
        else:
            data[key] = value

    return data
