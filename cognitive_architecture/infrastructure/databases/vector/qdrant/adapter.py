from vector.vector_db_interface import VectorDBInterface
from qdrant_client import AsyncQdrantClient

class QDrantAdapter(VectorDBInterface):
    def __init__(self, qdrant_url, qdrant_api_key):
        self.qdrant_client = AsyncQdrantClient(qdrant_url, qdrant_api_key)
  
    async def create_collection(
      self,
      collection_name: str,
      collection_config: object
    ):
        return await self.qdrant_client.create_collection(collection_name, collection_config)
