from typing import List, Optional, get_type_hints, TypeVar, Type
import asyncio
import lancedb
from lancedb.pydantic import Vector, LanceModel
from cognee.infrastructure.files.storage import LocalStorage
from ..vector_db_interface import VectorDBInterface, DataPoint
from ..embeddings.EmbeddingEngine import EmbeddingEngine

class LanceDBAdapter(VectorDBInterface):
    connection: lancedb.AsyncConnection = None

    def __init__(
        self,
        uri: Optional[str],
        api_key: Optional[str],
        embedding_engine: EmbeddingEngine,
    ):
        self.uri = uri
        self.api_key = api_key
        self.embedding_engine = embedding_engine

    async def get_connection(self):
        if self.connection is None:
            self.connection = await lancedb.connect_async(self.uri, api_key = self.api_key)

        return self.connection

    async def embed_data(self, data: List[str]) -> List[float]:
        return await self.embedding_engine.embed_text(data)

    async def collection_exists(self, collection_name: str) -> bool:
        connection = await self.get_connection()
        collection_names = await connection.table_names()
        return collection_name in collection_names

    PayloadSchema = TypeVar("PayloadSchema")

    async def create_collection(self, collection_name: str, payload_schema: Type[PayloadSchema]):
        data_point_types = get_type_hints(DataPoint)
        class LanceDataPoint(LanceModel):
            id: data_point_types["id"]
            vector: Vector(self.embedding_engine.get_vector_size())
            payload: payload_schema

        if not await self.collection_exists(collection_name):
            connection = await self.get_connection()
            return await connection.create_table(name = collection_name, schema = LanceDataPoint, exist_ok = True)
        else:
            raise ValueError(f"Collection {collection_name} already exists.")

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        connection = await self.get_connection()
        collection = await connection.open_table(collection_name)

        data_vectors = await self.embed_data([data_point.get_embeddable_data() for data_point in data_points])

        lance_data_points = [
            dict(
                id = data_point.id,
                vector = data_vectors[data_index],
                payload = data_point.payload
            ) for (data_index, data_point) in enumerate(data_points)
        ]

        await collection.add(lance_data_points)

    async def retrieve(self, collection_name: str, data_point_id: str):
        return (await self.search(collection_name, query_text = data_point_id))[0]

    async def search(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: List[float] = None,
        limit: int = None,
        # with_vector: bool = False
    ):
        if query_text is None and query_vector is None:
            raise ValueError("One of query_text or query_vector must be provided!")

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        connection = await self.get_connection()
        collection = await connection.open_table(collection_name)
        return await collection.search(query_vector).limit(limit).to_list()

    async def batch_search(self, collection_name: str, query_texts: List[str], limit: int = None):
        query_vectors = await self.embedding_engine.embed_text(query_texts)

        return asyncio.gather(
            *[self.search(
                collection_name = collection_name,
                query_vector = query_vector,
                limit = limit,
            ) for query_vector in query_vectors]
        )

    async def prune(self):
        # Clean up the database if it was set up as temporary
        if "file://" in self.uri:
            LocalStorage.remove_all(self.uri) # Remove the temporary directory and files inside
