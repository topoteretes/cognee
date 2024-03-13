import asyncio

from qdrant_client import models
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.infrastructure.databases.vector import get_vector_database

async def get_embeddings(texts:list):
    """ Get embeddings for a list of texts"""
    client = get_llm_client()
    tasks = [client.async_get_embedding_with_backoff(text, "text-embedding-3-large") for text in texts]
    return await asyncio.gather(*tasks)

async def add_proposition_to_vector_store(id, metadata, embeddings, collection_name):
    """ Upload a single embedding to a collection in Qdrant."""
    client = get_vector_database()

    await client.create_data_points(
        collection_name = collection_name,
        data_points = [
            models.PointStruct(
                id = id,
                payload = metadata,
                vector = {"content" : embeddings}
            )
        ]
    )


async def add_propositions(node_descriptions):
    for item in node_descriptions:
        embeddings = await get_embeddings([item["description"]])

        await add_proposition_to_vector_store(
            id = item["node_id"],
            metadata = {
                "meta": item["description"]
            },
            embeddings = embeddings[0],
            collection_name = item["layer_decomposition_uuid"]
        )
