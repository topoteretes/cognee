import logging
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

    for (dataset_name, chunks) in dataset_data_chunks.items():
        try:
        # if not await vector_client.collection_exists(dataset_name):
        #     logging.error(f"Creating collection {str(dataset_name)}")
            await vector_client.create_collection(dataset_name)
        except Exception:
            pass

        dataset_chunks = [
            dict(
                chunk_id = chunk["chunk_id"],
                collection = dataset_name,
                text = chunk["text"],
                file_metadata = chunk["file_metadata"],
            ) for chunk in chunks
        ]

        identified_chunks.extend(dataset_chunks)

        # if not await vector_client.collection_exists(dataset_name):
        try:
            logging.error("Collection still not found. Creating collection again.")
            await vector_client.create_collection(dataset_name)
        except:
            pass

        async def create_collection_retry(dataset_name, dataset_chunks):
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

        try:
            await create_collection_retry(dataset_name, dataset_chunks)
        except Exception:
            logging.error("Collection not found in create data points.")
            await create_collection_retry(dataset_name, dataset_chunks)

    return identified_chunks
