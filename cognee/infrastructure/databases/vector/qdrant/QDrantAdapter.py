import asyncio
from typing import List, Dict
# from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models
from ..vector_db_interface import VectorDBInterface
from ..models.DataPoint import DataPoint
from ..models.VectorConfig import VectorConfig
from ..models.CollectionConfig import CollectionConfig
from cognee.infrastructure.llm.get_llm_client import get_llm_client

# class CollectionConfig(BaseModel, extra = "forbid"):
#     vector_config: Dict[str, models.VectorParams] = Field(..., description="Vectors configuration" )
#     hnsw_config: Optional[models.HnswConfig] = Field(default = None, description="HNSW vector index configuration")
#     optimizers_config: Optional[models.OptimizersConfig] = Field(default = None, description="Optimizers configuration")
#     quantization_config: Optional[models.QuantizationConfig] = Field(default = None, description="Quantization configuration")

async def embed_data(data: str):
    llm_client = get_llm_client()

    return await llm_client.async_get_embedding_with_backoff(data)

async def convert_to_qdrant_point(data_point: DataPoint):
    return models.PointStruct(
        id = data_point.id,
        payload = data_point.payload,
        vector = {
            "text": await embed_data(data_point.get_embeddable_data())
        }
    )

def create_vector_config(vector_config: VectorConfig):
    return models.VectorParams(
        size = vector_config.size,
        distance = vector_config.distance
    )

def create_hnsw_config(hnsw_config: Dict):
    if hnsw_config is not None:
        return models.HnswConfig()
    return None

def create_optimizers_config(optimizers_config: Dict):
    if optimizers_config is not None:
        return models.OptimizersConfig()
    return None

def create_quantization_config(quantization_config: Dict):
    if quantization_config is not None:
        return models.QuantizationConfig()
    return None

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
            vectors_config = {
                "text": create_vector_config(collection_config.vector_config)
            }
        )

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        client = self.get_qdrant_client()

        awaitables = []

        for point in data_points:
            awaitables.append(convert_to_qdrant_point(point))

        points = await asyncio.gather(*awaitables)

        return await client.upload_points(
            collection_name = collection_name,
            points = points
        )

    async def search(self, collection_name: str, query_text: str, limit: int, with_vector: bool = False):
        client = self.get_qdrant_client()

        return await client.search(
            collection_name = collection_name,
            query_vector = models.NamedVector(
                name = "text",
                vector = await embed_data(query_text)
            ),
            limit = limit,
            with_vectors = with_vector
        )


    async def batch_search(self, collection_name: str, query_texts: List[str], limit: int, with_vectors: bool = False):
        """
        Perform batch search in a Qdrant collection with dynamic search requests.

        Args:
        - collection_name (str): Name of the collection to search in.
        - query_texts (List[str]): List of query texts to search for.
        - limit (int): List of result limits for search requests.
        - with_vectors (bool, optional): Bool indicating whether to return vectors for search requests.

        Returns:
        - results: The search results from Qdrant.
        """

        client = self.get_qdrant_client()

        vectors = await asyncio.gather(*[embed_data(query_text) for query_text in query_texts])

        # Generate dynamic search requests based on the provided embeddings
        requests = [
            models.SearchRequest(
                vector = models.NamedVector(
                    name = "text",
                    vector = vector
                ),
                limit = limit,
                with_vector = with_vectors
            ) for vector in vectors
        ]

        # Perform batch search with the dynamically generated requests
        results = await client.search_batch(
            collection_name = collection_name,
            requests = requests
        )

        return [filter(lambda result: result.score > 0.9, result_group) for result_group in results]

    async def prune(self):
        client = self.get_qdrant_client()

        response = await client.get_collections()

        for collection in response.collections:
            await client.delete_collection(collection.name)
