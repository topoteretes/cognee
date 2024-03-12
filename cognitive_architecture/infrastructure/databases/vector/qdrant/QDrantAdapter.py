from typing import List, Optional
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models
from ..vector_db_interface import VectorDBInterface

class CollectionConfig(BaseModel, extra = "forbid"):
    vector_config: models.VectorParams = Field(..., description="Vector configuration")
    hnsw_config: Optional[models.HnswConfig] = Field(default = None, description="HNSW vector index configuration")
    optimizers_config: Optional[models.OptimizersConfig] = Field(default = None, description="Optimizers configuration")
    quantization_config: Optional[models.QuantizationConfig] = Field(default = None, description="Quantization configuration")

class QDrantAdapter(VectorDBInterface):
    qdrant_url: str = None
    qdrant_path: str = None
    qdrant_api_key: str = None
  
    def __init__(self, qdrant_path, qdrant_url, qdrant_api_key):
        if qdrant_path is not None:
            self.qdrant_path = qdrant_path
        else:
            self.qdrant_url = qdrant_url

        self.qdrant_api_key = qdrant_api_key

    def get_qdrant_client(self) -> AsyncQdrantClient:
        if self.qdrant_path is not None:
            return AsyncQdrantClient(
                path = self.qdrant_path,
            )
        elif self.qdrant_url is not None:
            return AsyncQdrantClient(
                url = self.qdrant_url,
                api_key = self.qdrant_api_key,
            )

        return AsyncQdrantClient(
            location = ":memory:"
        )

    async def create_collection(
      self,
      collection_name: str,
      collection_config: CollectionConfig,
    ):
        client = self.get_qdrant_client()

        return await client.create_collection(
            collection_name = collection_name,
            vectors_config = collection_config.vector_config,
            hnsw_config = collection_config.hnsw_config,
            optimizers_config = collection_config.optimizers_config,
            quantization_config = collection_config.quantization_config
        )

    async def create_data_points(self, collection_name: str, data_points: List[any]):
        client = self.get_qdrant_client()

        return await client.upload_points(
            collection_name = collection_name,
            points = data_points
        )

    async def find_related_data_points(self, collection_name: str, query_vector):
        client = self.get_qdrant_client()

        return await client.search(
            collection_name = collection_name,
            query_vector = query_vector,
            with_payload = True,
            score_threshold = 0.8
        )
