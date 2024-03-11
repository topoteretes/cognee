from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models
from ..vector_db_interface import VectorDBInterface

class CollectionConfig(BaseModel, extra = "forbid"):
    vector_config: Dict[str, models.VectorParams] = Field(..., description="Vectors configuration" )
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

    async def create_data_points(self, collection_name: str, data_points):
        client = self.get_qdrant_client()

        return await client.upload_points(
            collection_name = collection_name,
            points = data_points
        )

    async def search(self, collection_name: str, query_vector: List[float], limit: int, with_vector: bool = False):
        client = self.get_qdrant_client()

        return await client.search(
            collection_name = collection_name,
            query_vector = query_vector,
            limit = limit,
            with_vectors = with_vector
        )


    async def batch_search(self, collection_name: str, embeddings: List[List[float]],
                                  with_vectors: List[bool] = None):
        """
        Perform batch search in a Qdrant collection with dynamic search requests.

        Args:
        - collection_name (str): Name of the collection to search in.
        - embeddings (List[List[float]]): List of embeddings to search for.
        - limits (List[int]): List of result limits for each search request.
        - with_vectors (List[bool], optional): List indicating whether to return vectors for each search request.
            Defaults to None, in which case vectors are not returned.

        Returns:
        - results: The search results from Qdrant.
        """

        client = self.get_qdrant_client()

        # Default with_vectors to False for each request if not provided
        if with_vectors is None:
            with_vectors = [False] * len(embeddings)

        # Ensure with_vectors list matches the length of embeddings and limits
        if len(with_vectors) != len(embeddings):
            raise ValueError("The length of with_vectors must match the length of embeddings and limits")

        # Generate dynamic search requests based on the provided embeddings
        requests = [
            models.SearchRequest(vector=models.NamedVector(
                name="content",
                vector=embedding,
            ),
                # vector= embedding,
                limit=3,
                with_vector=False
            ) for embedding in [embeddings]
        ]

        # Perform batch search with the dynamically generated requests
        results = await client.search_batch(
            collection_name=collection_name,
            requests=requests
        )

        return results
