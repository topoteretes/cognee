import asyncio
import logging
from typing import List, Optional
from ..vector_db_interface import VectorDBInterface
from ..models.DataPoint import DataPoint
from ..models.ScoredResult import ScoredResult
from ..embeddings.EmbeddingEngine import EmbeddingEngine

logger = logging.getLogger("WeaviateAdapter")

class WeaviateAdapter(VectorDBInterface):
    name = "Weaviate"
    url: str
    api_key: str
    embedding_engine: EmbeddingEngine = None

    def __init__(self, url: str, api_key: str, embedding_engine: EmbeddingEngine):
        import weaviate
        import weaviate.classes as wvc

        self.url = url
        self.api_key = api_key

        self.embedding_engine = embedding_engine

        self.client = weaviate.connect_to_wcs(
            cluster_url = url,
            auth_credentials = weaviate.auth.AuthApiKey(api_key),
            additional_config = wvc.init.AdditionalConfig(timeout = wvc.init.Timeout(init=30))
        )

    async def embed_data(self, data: List[str]) -> List[float]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        future = asyncio.Future()

        future.set_result(self.client.collections.exists(collection_name))

        return await future

    async def create_collection(
        self,
        collection_name: str,
        payload_schema = None,
    ):
        import weaviate.classes.config as wvcc

        future = asyncio.Future()

        future.set_result(
            self.client.collections.create(
                name=collection_name,
                properties=[
                    wvcc.Property(
                        name="text",
                        data_type=wvcc.DataType.TEXT,
                        skip_vectorization=True
                    )
                ]
            )
        )

        return await future

    def get_collection(self, collection_name: str):
        return self.client.collections.get(collection_name)

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        from weaviate.classes.data import DataObject

        data_vectors = await self.embed_data(
            list(map(lambda data_point: data_point.get_embeddable_data(), data_points)))

        def convert_to_weaviate_data_points(data_point: DataPoint):
            vector = data_vectors[data_points.index(data_point)]
            return DataObject(
                uuid = data_point.id,
                properties = data_point.payload.dict(),
                vector = vector
            )

        data_points = list(map(convert_to_weaviate_data_points, data_points))

        collection = self.get_collection(collection_name)

        try:
            if len(data_points) > 1:
                return collection.data.insert_many(data_points)
            else:
                return collection.data.insert(data_points[0])
            # with collection.batch.dynamic() as batch:
            #     for point in data_points:
            #         batch.add_object(
            #             uuid = point.uuid,
            #             properties = point.properties,
            #             vector = point.vector
            #         )
        except Exception as error:
            logger.error("Error creating data points: %s", str(error))
            raise error

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        from weaviate.classes.query import Filter
        future = asyncio.Future()

        data_points = self.get_collection(collection_name).query.fetch_objects(
            filters = Filter.by_id().contains_any(data_point_ids)
        )

        for data_point in data_points:
            data_point.payload = data_point.properties
            del data_point.properties

        future.set_result(data_points)

        return await future

    async def search(
            self,
            collection_name: str,
            query_text: Optional[str] = None,
            query_vector: Optional[List[float]] = None,
            limit: int = None,
            with_vector: bool = False
    ):
        import weaviate.classes as wvc

        if query_text is None and query_vector is None:
            raise ValueError("One of query_text or query_vector must be provided!")

        if query_vector is None:
            query_vector = (await self.embed_data([query_text]))[0]

        search_result = self.get_collection(collection_name).query.hybrid(
            query = None,
            vector = query_vector,
            limit = limit,
            include_vector = with_vector,
            return_metadata = wvc.query.MetadataQuery(score=True),
        )

        return [
            ScoredResult(
                id=str(result.uuid),
                payload=result.properties,
                score=float(result.metadata.score)
            ) for result in search_result.objects
        ]

    async def batch_search(self, collection_name: str, query_texts: List[str], limit: int, with_vectors: bool = False):
        def query_search(query_vector):
            return self.search(collection_name, query_vector=query_vector, limit=limit, with_vector=with_vectors)

        return [await query_search(query_vector) for query_vector in await self.embed_data(query_texts)]
        
    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        from weaviate.classes.query import Filter
        future = asyncio.Future()

        result = self.get_collection(collection_name).data.delete_many(
            filters = Filter.by_id().contains_any(data_point_ids)
        )
        future.set_result(result)

        return await future

    async def prune(self):
        self.client.collections.delete_all()
