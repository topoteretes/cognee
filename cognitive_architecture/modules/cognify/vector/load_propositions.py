import asyncio

from cognitive_architecture.infrastructure.llm.get_llm_client import get_llm_client
from qdrant_client import models
from cognitive_architecture.infrastructure.databases.vector.get_vector_database import get_vector_database

async def get_embeddings(texts):
    client = get_llm_client()
    tasks = [ client.async_get_embedding_with_backoff(text, "text-embedding-3-large") for text in texts]
    results = await asyncio.gather(*tasks)
    return results
async def upload_embedding(id, metadata, some_embeddings, collection_name, client):
    print(id)
    # if some_embeddings and isinstance(some_embeddings[0], list):
    #     some_embeddings = [item for sublist in some_embeddings for item in sublist]

    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=id, vector={"content" :some_embeddings}, payload=metadata
            )
        ]
        ,
    )


async def add_propositions(node_descriptions, client):
    for item in node_descriptions:
        print(item['node_id'])
        await upload_embedding(id = item['node_id'], metadata = {"meta":item['description']}, some_embeddings = get_embeddings(item['description']), collection_name= item['layer_decomposition_uuid'],client= client)


