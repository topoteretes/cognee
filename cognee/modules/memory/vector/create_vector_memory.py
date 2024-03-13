from cognee.infrastructure.databases.vector.qdrant.adapter import CollectionConfig
from cognee.infrastructure.databases.vector.get_vector_database import get_vector_database

async def create_vector_memory(memory_name: str, collection_config: CollectionConfig):
    vector_db = get_vector_database()

    return await vector_db.create_collection(memory_name, collection_config)
