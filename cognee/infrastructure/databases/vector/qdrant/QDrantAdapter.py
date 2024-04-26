from typing import List, Dict, Optional
from qdrant_client import AsyncQdrantClient, models
from ..vector_db_interface import VectorDBInterface
from ..models.DataPoint import DataPoint
from ..embeddings.EmbeddingEngine import EmbeddingEngine

# class CollectionConfig(BaseModel, extra = "forbid"):
#     vector_config: Dict[str, models.VectorParams] = Field(..., description="Vectors configuration" )
#     hnsw_config: Optional[models.HnswConfig] = Field(default = None, description="HNSW vector index configuration")
#     optimizers_config: Optional[models.OptimizersConfig] = Field(default = None, description="Optimizers configuration")
#     quantization_config: Optional[models.QuantizationConfig] = Field(default = None, description="Quantization configuration")

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

    def __init__(self, qdrant_url, qdrant_api_key, embedding_engine: EmbeddingEngine, qdrant_path = None):
        self.embedding_engine = embedding_engine

        if qdrant_path is not None:
            self.qdrant_path = qdrant_path
        else:
            self.qdrant_url = qdrant_url

        self.qdrant_api_key = qdrant_api_key

    def get_qdrant_client(self) -> AsyncQdrantClient:
        if self.qdrant_path is not None:
            return AsyncQdrantClient(
                path = self.qdrant_path, port=6333
            )
        elif self.qdrant_url is not None:
            return AsyncQdrantClient(
                url = self.qdrant_url,
                api_key = self.qdrant_api_key,
                port = 6333
            )

        return AsyncQdrantClient(
            location = ":memory:"
        )

    async def embed_data(self, data: List[str]) -> List[float]:
        return await self.embedding_engine.embed_text(data)

    async def collection_exists(self, collection_name: str) -> bool:
        client = self.get_qdrant_client()
        result = await client.collection_exists(collection_name)
        await client.close()
        return result

    async def create_collection(
      self,
      collection_name: str,
    ):
        client = self.get_qdrant_client()

        result = await client.create_collection(
            collection_name = collection_name,
            vectors_config = {
                "text": models.VectorParams(
                    size = self.embedding_engine.get_vector_size(),
                    distance = "Cosine"
                )
            }
        )

        await client.close()

        return result

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        client = self.get_qdrant_client()

        data_vectors = await self.embed_data([data_point.get_embeddable_data() for data_point in data_points])

        def convert_to_qdrant_point(data_point: DataPoint):
            return models.PointStruct(
                id = data_point.id,
                payload = data_point.payload,
                vector = {
                    "text": data_vectors[data_points.index(data_point)]
                }
            )

        points = [convert_to_qdrant_point(point) for point in data_points]

        result = await client.upload_points(
            collection_name = collection_name,
            points = points
        )

        await client.close()

        return result

    async def retrieve(self, collection_name: str, data_id: str):
        client = self.get_qdrant_client()
        results = await client.retrieve(collection_name, [data_id], with_payload = True)
        await client.close()
        return results[0] if len(results) > 0 else None

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = None,
        with_vector: bool = False
    ):
        if query_text is None and query_vector is None:
            raise ValueError("One of query_text or query_vector must be provided!")

        client = self.get_qdrant_client()

        result = await client.search(
            collection_name = collection_name,
            query_vector = models.NamedVector(
                name = "text",
                vector = query_vector if query_vector is not None else (await self.embed_data([query_text]))[0],
            ),
            limit = limit,
            with_vectors = with_vector
        )

        await client.close()

        return result


    async def batch_search(self, collection_name: str, query_texts: List[str], limit: int = None, with_vectors: bool = False):
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

        vectors = await self.embed_data(query_texts)

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

        client = self.get_qdrant_client()

        # Perform batch search with the dynamically generated requests
        results = await client.search_batch(
            collection_name = collection_name,
            requests = requests
        )

        await client.close()

        return [filter(lambda result: result.score > 0.9, result_group) for result_group in results]

    async def prune(self):
        client = self.get_qdrant_client()

        response = await client.get_collections()

        for collection in response.collections:
            await client.delete_collection(collection.name)

        await client.close()
