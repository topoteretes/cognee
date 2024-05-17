
import json
import logging

from typing import TypedDict
from pydantic import BaseModel, Field
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.vector import DataPoint

class TextChunk(TypedDict):
    text: str
    chunk_id: str
    file_metadata: dict

async def add_data_chunks(dataset_data_chunks: dict[str, list[TextChunk]]):
    vector_client = infrastructure_config.get_config("vector_engine")

    identified_chunks = []

    class PayloadSchema(BaseModel):
        text: str = Field(...)

    for (dataset_name, chunks) in dataset_data_chunks.items():
        try:

            await vector_client.create_collection(dataset_name, payload_schema = PayloadSchema)
        except Exception as error:
            print(error)
            pass

        dataset_chunks = [
            dict(
                chunk_id = chunk["chunk_id"],
                collection = dataset_name,
                text = chunk["text"],
                document_id = chunk["document_id"],
                file_metadata = chunk["file_metadata"],
            ) for chunk in chunks
        ]

        identified_chunks.extend(dataset_chunks)

        await vector_client.create_data_points(
            dataset_name,
            [
                DataPoint[PayloadSchema](
                    id = chunk["chunk_id"],
                    payload = PayloadSchema.parse_obj(dict(text = chunk["text"])),
                    embed_field = "text",
                ) for chunk in dataset_chunks
            ],
        )

    return identified_chunks
