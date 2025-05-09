from typing import List, Optional

from cognee.shared.logging_utils import get_logger
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.ScoredResult import ScoredResult
from ..vector_db_interface import VectorDBInterface

logger = get_logger("WeaviateAdapter")


class IndexSchema(DataPoint):
    text: str

    metadata: dict = {"index_fields": ["text"]}


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

        self.client = weaviate.use_async_with_weaviate_cloud(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(api_key),
            additional_config=wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=30)),
        )

    async def get_client(self):
        await self.client.connect()

        return self.client

    async def embed_data(self, data: List[str]) -> List[float]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        client = await self.get_client()
        return await client.collections.exists(collection_name)

    async def create_collection(
        self,
        collection_name: str,
        payload_schema=None,
    ):
        import weaviate.classes.config as wvcc

        if not await self.has_collection(collection_name):
            client = await self.get_client()
            return await client.collections.create(
                name=collection_name,
                properties=[
                    wvcc.Property(
                        name="text", data_type=wvcc.DataType.TEXT, skip_vectorization=True
                    )
                ],
            )
        else:
            return await self.get_collection(collection_name)

    async def get_collection(self, collection_name: str):
        if not await self.has_collection(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found.")

        client = await self.get_client()
        return client.collections.get(collection_name)

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        from weaviate.classes.data import DataObject

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        def convert_to_weaviate_data_points(data_point: DataPoint):
            vector = data_vectors[data_points.index(data_point)]
            properties = data_point.model_dump()

            if "id" in properties:
                properties["uuid"] = str(data_point.id)
                del properties["id"]

            return DataObject(uuid=data_point.id, properties=properties, vector=vector)

        data_points = [convert_to_weaviate_data_points(data_point) for data_point in data_points]

        collection = await self.get_collection(collection_name)

        try:
            if len(data_points) > 1:
                return await collection.data.insert_many(data_points)
                # with collection.batch.dynamic() as batch:
                #     for data_point in data_points:
                #         batch.add_object(
                #             uuid=data_point.uuid,
                #             vector=data_point.vector,
                #             properties=data_point.properties,
                #             references=data_point.references,
                #         )
            else:
                data_point: DataObject = data_points[0]
                if collection.data.exists(data_point.uuid):
                    return await collection.data.update(
                        uuid=data_point.uuid,
                        vector=data_point.vector,
                        properties=data_point.properties,
                        references=data_point.references,
                    )
                else:
                    return await collection.data.insert(
                        uuid=data_point.uuid,
                        vector=data_point.vector,
                        properties=data_point.properties,
                        references=data_point.references,
                    )
        except Exception as error:
            logger.error("Error creating data points: %s", str(error))
            raise error

    async def create_vector_index(self, index_name: str, index_property_name: str):
        return await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        return await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=data_point.id,
                    text=DataPoint.get_embeddable_data(data_point),
                )
                for data_point in data_points
            ],
        )

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        from weaviate.classes.query import Filter

        collection = await self.get_collection(collection_name)
        data_points = await collection.query.fetch_objects(
            filters=Filter.by_id().contains_any(data_point_ids)
        )

        for data_point in data_points.objects:
            data_point.payload = data_point.properties
            data_point.id = data_point.uuid
            del data_point.properties

        return data_points.objects

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 15,
        with_vector: bool = False,
    ):
        import weaviate.classes as wvc
        import weaviate.exceptions

        if query_text is None and query_vector is None:
            raise InvalidValueError(message="One of query_text or query_vector must be provided!")

        if query_vector is None:
            query_vector = (await self.embed_data([query_text]))[0]

        collection = await self.get_collection(collection_name)

        try:
            search_result = await collection.query.hybrid(
                query=None,
                vector=query_vector,
                limit=limit if limit > 0 else None,
                include_vector=with_vector,
                return_metadata=wvc.query.MetadataQuery(score=True),
            )

            return [
                ScoredResult(
                    id=parse_id(str(result.uuid)),
                    payload=result.properties,
                    score=1 - float(result.metadata.score),
                )
                for result in search_result.objects
            ]
        except weaviate.exceptions.WeaviateInvalidInputError:
            # Ignore if the collection doesn't exist
            return []

    async def batch_search(
        self, collection_name: str, query_texts: List[str], limit: int, with_vectors: bool = False
    ):
        def query_search(query_vector):
            return self.search(
                collection_name, query_vector=query_vector, limit=limit, with_vector=with_vectors
            )

        return [
            await query_search(query_vector) for query_vector in await self.embed_data(query_texts)
        ]

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        from weaviate.classes.query import Filter

        collection = await self.get_collection(collection_name)
        result = await collection.data.delete_many(
            filters=Filter.by_id().contains_any(data_point_ids)
        )

        return result

    async def prune(self):
        client = await self.get_client()
        await client.collections.delete_all()
