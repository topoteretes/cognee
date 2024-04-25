import asyncio
from uuid import UUID
from typing import List, Optional
from multiprocessing import Pool
from ..vector_db_interface import VectorDBInterface
from ..models.DataPoint import DataPoint
from ..models.ScoredResult import ScoredResult
from ..embeddings.EmbeddingEngine import EmbeddingEngine


class WeaviateAdapter(VectorDBInterface):
    async_pool: Pool = None
    embedding_engine: EmbeddingEngine = None

    def __init__(self, url: str, api_key: str, embedding_engine: EmbeddingEngine):
        import weaviate
        import weaviate.classes as wvc
      
        self.embedding_engine = embedding_engine

        self.client = weaviate.connect_to_wcs(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key),
            # headers = {
            #     "X-OpenAI-Api-Key": openai_api_key
            # },
            additional_config=wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=30))
        )

    async def embed_data(self, data: List[str]) -> List[float]:
        return await self.embedding_engine.embed_text(data)

    async def collection_exists(self, collection_name: str) -> bool:
        event_loop = asyncio.get_event_loop()

        def sync_collection_exists():
            return self.client.collections.exists(collection_name)

        return await event_loop.run_in_executor(None, sync_collection_exists)

    async def create_collection(self, collection_name: str):
        import weaviate.classes.config as wvcc

        event_loop = asyncio.get_event_loop()

        def sync_create_collection():
            return self.client.collections.create(
                name=collection_name,
                properties=[
                    wvcc.Property(
                        name="text",
                        data_type=wvcc.DataType.TEXT,
                        skip_vectorization=True
                    )
                ]
            )

        # try:
        result = await event_loop.run_in_executor(None, sync_create_collection)
        # finally:
        #     event_loop.shutdown_executor()

        return result

    def get_collection(self, collection_name: str):
        return self.client.collections.get(collection_name)

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        from weaviate.classes.data import DataObject

        data_vectors = await self.embed_data(
            list(map(lambda data_point: data_point.get_embeddable_data(), data_points)))

        def convert_to_weaviate_data_points(data_point: DataPoint):
            return DataObject(
                uuid=data_point.id,
                properties=data_point.payload,
                vector=data_vectors[data_points.index(data_point)]
            )

        objects = list(map(convert_to_weaviate_data_points, data_points))

        return self.get_collection(collection_name).data.insert_many(objects)

    async def retrieve(self, collection_name: str, data_id: str):
        def sync_retrieve():
            return self.get_collection(collection_name).query.fetch_object_by_id(UUID(data_id))

        event_loop = asyncio.get_event_loop()

        # try:
        data_point = await event_loop.run_in_executor(None, sync_retrieve)
        # finally:
            # event_loop.shutdown_executor()

        data_point.payload = data_point.properties
        del data_point.properties

        return data_point

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

        # def sync_search():
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

    async def prune(self):
        self.client.collections.delete_all()
