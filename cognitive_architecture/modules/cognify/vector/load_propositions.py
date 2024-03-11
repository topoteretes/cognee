import asyncio

from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client
from qdrant_client import models
from cognitive_architecture.infrastructure.databases.vector.get_vector_database import get_vector_database

async def get_embeddings(texts:list):
    """ Get embeddings for a list of texts"""
    client = get_llm_client()
    tasks = [ client.async_get_embedding_with_backoff(text, "text-embedding-3-large") for text in texts]
    results = await asyncio.gather(*tasks)
    return results
async def upload_embedding(id, metadata, some_embeddings, collection_name):
    """ Upload a single embedding to a collection in Qdrant."""
    client = get_vector_database()
    # print("Uploading embeddings")
    await client.create_data_points(
        collection_name=collection_name,
        data_points=[
            models.PointStruct(
                id=id, vector={"content" :some_embeddings}, payload=metadata
            )
        ]
        ,
    )


async def add_propositions(node_descriptions):
    for item in node_descriptions:
        embeddings = await get_embeddings([item['description']])
        await upload_embedding(id = item['node_id'], metadata = {"meta":item['description']},
                               some_embeddings = embeddings[0],
                               collection_name= item['layer_decomposition_uuid'])


