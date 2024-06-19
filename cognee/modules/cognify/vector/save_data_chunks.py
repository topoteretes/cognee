
from uuid import UUID
from typing import TypedDict
from pydantic import BaseModel
from cognee.infrastructure.databases.vector import DataPoint, get_vector_engine

class DataChunk(TypedDict):
    id: UUID
    text: str
    word_count: int
    paragraph_id: UUID
    chunk_index: int
    is_end_chunk: bool

async def save_data_chunks(collection_name: str, data_chunks: list[DataChunk]):
    vector_engine = get_vector_engine()

    class PayloadSchema(BaseModel):
        id: UUID
        text: str
        word_count: int
        paragraph_id: UUID
        chunk_index: int
        is_end_chunk: bool

    await vector_engine.create_collection(collection_name, payload_schema = PayloadSchema)

    await vector_engine.create_data_points(
        collection_name,
        [
            DataPoint[PayloadSchema](
                id = chunk["id"],
                payload = PayloadSchema.parse_obj(dict(
                    text = chunk["text"],
                    word_count = chunk["word_count"],
                    paragraph_id = chunk["paragraph_id"],
                    chunk_index = chunk["chunk_index"],
                    is_end_chunk = chunk["is_end_chunk"],
                )),
                embed_field = "text",
            ) for chunk in data_chunks
        ],
    )

    return data_chunks
