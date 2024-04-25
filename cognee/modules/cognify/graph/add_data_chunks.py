from typing import TypedDict
from cognee.infrastructure import infrastructure_config
from cognee.infrastructure.databases.vector import DataPoint

class TextChunk(TypedDict):
    text: str
    chunk_id: str
    file_metadata: dict

async def add_data_chunks(dataset_data_chunks: dict[str, list[TextChunk]]):
    vector_client = infrastructure_config.get_config("vector_engine")

    identified_chunks = []

    for dataset_name in dataset_data_chunks.keys():
        try:
            await vector_client.create_collection(dataset_name)
        except Exception as error:
            print(error)
            pass

    for (dataset_name, chunks) in dataset_data_chunks.items():
        dataset_chunks = [
            dict(
                chunk_id = chunk["chunk_id"],
                collection = dataset_name,
                text = chunk["text"],
                file_metadata = chunk["file_metadata"],
            ) for chunk in chunks
        ]

        identified_chunks.extend(dataset_chunks)

        await vector_client.create_data_points(
            dataset_name,
            [
                DataPoint(
                    id = chunk["chunk_id"],
                    payload = dict(text = chunk["text"]),
                    embed_field = "text"
                ) for chunk in dataset_chunks
            ],
        )

    return identified_chunks
