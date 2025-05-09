from typing import Dict, List, Optional
from qdrant_client import AsyncQdrantClient, models

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine.utils import parse_id
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..vector_db_interface import VectorDBInterface

logger = get_logger("QDrantAdapter")


class IndexSchema(DataPoint):
    text: str

    metadata: dict = {"index_fields": ["text"]}


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
    name = "Qdrant"
    url: str = None
    api_key: str = None
    qdrant_path: str = None

    def __init__(self, url, api_key, embedding_engine: EmbeddingEngine, qdrant_path=None):
        self.embedding_engine = embedding_engine

        if qdrant_path is not None:
            self.qdrant_path = qdrant_path
        else:
            self.url = url
            self.api_key = api_key

    def get_qdrant_client(self) -> AsyncQdrantClient:
        if self.qdrant_path is not None:
            return AsyncQdrantClient(path=self.qdrant_path, port=6333)
        elif self.url is not None:
            return AsyncQdrantClient(url=self.url, api_key=self.api_key, port=6333)

        return AsyncQdrantClient(location=":memory:")

    async def embed_data(self, data: List[str]) -> List[float]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        client = self.get_qdrant_client()
        result = await client.collection_exists(collection_name)
        await client.close()
        return result

    async def create_collection(
        self,
        collection_name: str,
        payload_schema=None,
    ):
        client = self.get_qdrant_client()

        if not await client.collection_exists(collection_name):
            await client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text": models.VectorParams(
                        size=self.embedding_engine.get_vector_size(), distance="Cosine"
                    )
                },
            )

        await client.close()

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = self.get_qdrant_client()

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        def convert_to_qdrant_point(data_point: DataPoint):
            return models.PointStruct(
                id=str(data_point.id),
                payload=data_point.model_dump(),
                vector={"text": data_vectors[data_points.index(data_point)]},
            )

        points = [convert_to_qdrant_point(point) for point in data_points]

        try:
            client.upload_points(collection_name=collection_name, points=points)
        except UnexpectedResponse as error:
            if "Collection not found" in str(error):
                raise CollectionNotFoundError(
                    message=f"Collection {collection_name} not found!"
                ) from error
            else:
                raise error
        except Exception as error:
            logger.error("Error uploading data points to Qdrant: %s", str(error))
            raise error
        finally:
            await client.close()

    async def create_vector_index(self, index_name: str, index_property_name: str):
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=data_point.id,
                    text=getattr(data_point, data_point.metadata["index_fields"][0]),
                )
                for data_point in data_points
            ],
        )

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        client = self.get_qdrant_client()
        results = await client.retrieve(collection_name, data_point_ids, with_payload=True)
        await client.close()
        return results

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 15,
        with_vector: bool = False,
    ):
        from qdrant_client.http.exceptions import UnexpectedResponse

        if query_text is None and query_vector is None:
            raise InvalidValueError(message="One of query_text or query_vector must be provided!")

        try:
            client = self.get_qdrant_client()

            results = await client.search(
                collection_name=collection_name,
                query_vector=models.NamedVector(
                    name="text",
                    vector=query_vector
                    if query_vector is not None
                    else (await self.embed_data([query_text]))[0],
                ),
                limit=limit if limit > 0 else None,
                with_vectors=with_vector,
            )

            await client.close()

            return [
                ScoredResult(
                    id=parse_id(result.id),
                    payload={
                        **result.payload,
                        "id": parse_id(result.id),
                    },
                    score=1 - result.score,
                )
                for result in results
            ]
        except UnexpectedResponse as error:
            if "Collection not found" in str(error):
                raise CollectionNotFoundError(
                    message=f"Collection {collection_name} not found!"
                ) from error
            else:
                raise error
        finally:
            await client.close()

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = None,
        with_vectors: bool = False,
    ):
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
                vector=models.NamedVector(name="text", vector=vector),
                limit=limit,
                with_vector=with_vectors,
            )
            for vector in vectors
        ]

        client = self.get_qdrant_client()

        # Perform batch search with the dynamically generated requests
        results = await client.search_batch(collection_name=collection_name, requests=requests)

        await client.close()

        return [filter(lambda result: result.score > 0.9, result_group) for result_group in results]

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        client = self.get_qdrant_client()
        results = await client.delete(collection_name, data_point_ids)
        return results

    async def prune(self):
        client = self.get_qdrant_client()

        response = await client.get_collections()

        for collection in response.collections:
            await client.delete_collection(collection.name)

        await client.close()
