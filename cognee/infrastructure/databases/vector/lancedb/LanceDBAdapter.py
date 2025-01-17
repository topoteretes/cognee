import asyncio
from typing import Generic, List, Optional, TypeVar, get_type_hints
from uuid import UUID

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.files.storage import LocalStorage
from cognee.modules.storage.utils import copy_model, get_own_properties

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.ScoredResult import ScoredResult
from ..utils import normalize_distances
from ..vector_db_interface import VectorDBInterface


class IndexSchema(DataPoint):
    id: str
    text: str

    _metadata: dict = {"index_fields": ["text"], "type": "IndexSchema"}


class LanceDBAdapter(VectorDBInterface):
    name = "LanceDB"
    url: str
    api_key: str
    connection: lancedb.AsyncConnection = None

    def __init__(
        self,
        url: Optional[str],
        api_key: Optional[str],
        embedding_engine: EmbeddingEngine,
    ):
        self.url = url
        self.api_key = api_key
        self.embedding_engine = embedding_engine

    async def get_connection(self):
        if self.connection is None:
            self.connection = await lancedb.connect_async(self.url, api_key=self.api_key)

        return self.connection

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        connection = await self.get_connection()
        collection_names = await connection.table_names()
        return collection_name in collection_names

    async def create_collection(self, collection_name: str, payload_schema: BaseModel):
        vector_size = self.embedding_engine.get_vector_size()

        payload_schema = self.get_data_point_schema(payload_schema)
        data_point_types = get_type_hints(payload_schema)

        class LanceDataPoint(LanceModel):
            id: data_point_types["id"]
            vector: Vector(vector_size)
            payload: payload_schema

        if not await self.has_collection(collection_name):
            connection = await self.get_connection()
            return await connection.create_table(
                name=collection_name,
                schema=LanceDataPoint,
                exist_ok=True,
            )

    async def create_data_points(self, collection_name: str, data_points: list[DataPoint]):
        connection = await self.get_connection()

        payload_schema = type(data_points[0])
        payload_schema = self.get_data_point_schema(payload_schema)

        if not await self.has_collection(collection_name):
            await self.create_collection(
                collection_name,
                payload_schema,
            )

        collection = await connection.open_table(collection_name)

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        IdType = TypeVar("IdType")
        PayloadSchema = TypeVar("PayloadSchema")
        vector_size = self.embedding_engine.get_vector_size()

        class LanceDataPoint(LanceModel, Generic[IdType, PayloadSchema]):
            id: IdType
            vector: Vector(vector_size)
            payload: PayloadSchema

        def create_lance_data_point(data_point: DataPoint, vector: list[float]) -> LanceDataPoint:
            properties = get_own_properties(data_point)
            properties["id"] = str(properties["id"])

            return LanceDataPoint[str, self.get_data_point_schema(type(data_point))](
                id=str(data_point.id),
                vector=vector,
                payload=properties,
            )

        lance_data_points = [
            create_lance_data_point(data_point, data_vectors[data_point_index])
            for (data_point_index, data_point) in enumerate(data_points)
        ]

        await (
            collection.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(lance_data_points)
        )

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        connection = await self.get_connection()
        collection = await connection.open_table(collection_name)

        if len(data_point_ids) == 1:
            results = await collection.query().where(f"id = '{data_point_ids[0]}'").to_pandas()
        else:
            results = await collection.query().where(f"id IN {tuple(data_point_ids)}").to_pandas()

        return [
            ScoredResult(
                id=UUID(result["id"]),
                payload=result["payload"],
                score=0,
            )
            for result in results.to_dict("index").values()
        ]

    async def get_distance_from_collection_elements(
        self, collection_name: str, query_text: str = None, query_vector: List[float] = None
    ):
        if query_text is None and query_vector is None:
            raise InvalidValueError(message="One of query_text or query_vector must be provided!")

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        connection = await self.get_connection()
        collection = await connection.open_table(collection_name)

        collection_size = await collection.count_rows()

        results = await collection.vector_search(query_vector).limit(collection_size).to_pandas()

        result_values = list(results.to_dict("index").values())

        normalized_values = normalize_distances(result_values)

        return [
            ScoredResult(
                id=UUID(result["id"]),
                payload=result["payload"],
                score=normalized_values[value_index],
            )
            for value_index, result in enumerate(result_values)
        ]

    async def search(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: List[float] = None,
        limit: int = 5,
        with_vector: bool = False,
        normalized: bool = True,
    ):
        if query_text is None and query_vector is None:
            raise InvalidValueError(message="One of query_text or query_vector must be provided!")

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        connection = await self.get_connection()
        collection = await connection.open_table(collection_name)

        results = await collection.vector_search(query_vector).limit(limit).to_pandas()

        result_values = list(results.to_dict("index").values())

        normalized_values = normalize_distances(result_values)

        return [
            ScoredResult(
                id=UUID(result["id"]),
                payload=result["payload"],
                score=normalized_values[value_index],
            )
            for value_index, result in enumerate(result_values)
        ]

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = None,
        with_vectors: bool = False,
    ):
        query_vectors = await self.embedding_engine.embed_text(query_texts)

        return await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_vector=with_vectors,
                )
                for query_vector in query_vectors
            ]
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        connection = await self.get_connection()
        collection = await connection.open_table(collection_name)
        if len(data_point_ids) == 1:
            results = await collection.delete(f"id = '{data_point_ids[0]}'")
        else:
            results = await collection.delete(f"id IN {tuple(data_point_ids)}")
        return results

    async def create_vector_index(self, index_name: str, index_property_name: str):
        await self.create_collection(
            f"{index_name}_{index_property_name}", payload_schema=IndexSchema
        )

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=str(data_point.id),
                    text=getattr(data_point, data_point._metadata["index_fields"][0]),
                )
                for data_point in data_points
            ],
        )

    async def prune(self):
        connection = await self.get_connection()
        collection_names = await connection.table_names()

        for collection_name in collection_names:
            collection = await connection.open_table(collection_name)
            await collection.delete("id IS NOT NULL")
            await connection.drop_table(collection_name)

        if self.url.startswith("/"):
            LocalStorage.remove_all(self.url)

    def get_data_point_schema(self, model_type):
        return copy_model(
            model_type,
            include_fields={
                "id": (str, ...),
            },
            exclude_fields=["_metadata"],
        )
