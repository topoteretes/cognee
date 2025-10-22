import asyncio
from os import path
import lancedb
from pydantic import BaseModel
from lancedb.pydantic import LanceModel, Vector
from typing import Generic, List, Optional, TypeVar, Union, get_args, get_origin, get_type_hints

from cognee.infrastructure.databases.exceptions import MissingQueryParameterError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.files.storage import get_file_storage
from cognee.modules.storage.utils import copy_model, get_own_properties
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.ScoredResult import ScoredResult
from ..utils import normalize_distances
from ..vector_db_interface import VectorDBInterface


class IndexSchema(DataPoint):
    """
    Represents a schema for an index data point containing an ID and text.

    Attributes:

    - id: A string representing the unique identifier for the data point.
    - text: A string representing the content of the data point.
    - metadata: A dictionary with default index fields for the schema, currently configured
    to include 'text'.
    """

    id: str
    text: str

    metadata: dict = {"index_fields": ["text"]}


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
        self.VECTOR_DB_LOCK = asyncio.Lock()

    async def get_connection(self):
        """
        Establishes and returns a connection to the LanceDB.

        If a connection already exists, it will return the existing connection.

        Returns:
        --------

            - lancedb.AsyncConnection: An active connection to the LanceDB.
        """
        if self.connection is None:
            self.connection = await lancedb.connect_async(self.url, api_key=self.api_key)

        return self.connection

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """
        Embeds the provided textual data into vector representation.

        Uses the embedding engine to convert the list of strings into a list of float vectors.

        Parameters:
        -----------

            - data (list[str]): A list of strings representing the data to be embedded.

        Returns:
        --------

            - list[list[float]]: A list of embedded vectors corresponding to the input data.
        """
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        """
        Checks if the specified collection exists in the LanceDB.

        Returns True if the collection is present, otherwise False.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check.

        Returns:
        --------

            - bool: True if the collection exists, otherwise False.
        """
        connection = await self.get_connection()
        collection_names = await connection.table_names()
        return collection_name in collection_names

    async def create_collection(self, collection_name: str, payload_schema: BaseModel):
        vector_size = self.embedding_engine.get_vector_size()

        payload_schema = self.get_data_point_schema(payload_schema)
        data_point_types = get_type_hints(payload_schema)

        class LanceDataPoint(LanceModel):
            """
            Represents a data point in the Lance model with an ID, vector, and associated payload.

            The class inherits from LanceModel and defines the following public attributes:
            - id: A unique identifier for the data point.
            - vector: A vector representing the data point in a specified dimensional space.
            - payload: Additional data or metadata associated with the data point.
            """

            id: data_point_types["id"]
            vector: Vector(vector_size)
            payload: payload_schema

        if not await self.has_collection(collection_name):
            async with self.VECTOR_DB_LOCK:
                if not await self.has_collection(collection_name):
                    connection = await self.get_connection()
                    return await connection.create_table(
                        name=collection_name,
                        schema=LanceDataPoint,
                        exist_ok=True,
                    )

    async def get_collection(self, collection_name: str):
        if not await self.has_collection(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found!")

        connection = await self.get_connection()
        return await connection.open_table(collection_name)

    async def create_data_points(self, collection_name: str, data_points: list[DataPoint]):
        payload_schema = type(data_points[0])

        if not await self.has_collection(collection_name):
            async with self.VECTOR_DB_LOCK:
                if not await self.has_collection(collection_name):
                    await self.create_collection(
                        collection_name,
                        payload_schema,
                    )

        collection = await self.get_collection(collection_name)

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        IdType = TypeVar("IdType")
        PayloadSchema = TypeVar("PayloadSchema")
        vector_size = self.embedding_engine.get_vector_size()

        class LanceDataPoint(LanceModel, Generic[IdType, PayloadSchema]):
            """
            Represents a data point in the Lance model with an ID, vector, and payload.

            This class encapsulates a data point consisting of an identifier, a vector representing
            the data, and an associated payload, allowing for operations and manipulations specific
            to the Lance data structure.
            """

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

        async with self.VECTOR_DB_LOCK:
            await (
                collection.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(lance_data_points)
            )

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        collection = await self.get_collection(collection_name)

        if len(data_point_ids) == 1:
            results = await collection.query().where(f"id = '{data_point_ids[0]}'")
        else:
            results = await collection.query().where(f"id IN {tuple(data_point_ids)}")

        # Convert query results to list format
        results_list = results.to_list() if hasattr(results, "to_list") else list(results)

        return [
            ScoredResult(
                id=parse_id(result["id"]),
                payload=result["payload"],
                score=0,
            )
            for result in results_list
        ]

    async def search(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: List[float] = None,
        limit: Optional[int] = 15,
        with_vector: bool = False,
        normalized: bool = True,
    ):
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        collection = await self.get_collection(collection_name)

        if limit is None:
            limit = await collection.count_rows()

        # LanceDB search will break if limit is 0 so we must return
        if limit <= 0:
            return []

        result_values = await collection.vector_search(query_vector).limit(limit).to_list()

        if not result_values:
            return []

        normalized_values = normalize_distances(result_values)

        return [
            ScoredResult(
                id=parse_id(result["id"]),
                payload=result["payload"],
                score=normalized_values[value_index],
            )
            for value_index, result in enumerate(result_values)
        ]

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: Optional[int] = None,
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
        collection = await self.get_collection(collection_name)

        # Delete one at a time to avoid commit conflicts
        for data_point_id in data_point_ids:
            await collection.delete(f"id = '{data_point_id}'")

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
                    text=getattr(data_point, data_point.metadata["index_fields"][0]),
                )
                for data_point in data_points
            ],
        )

    async def prune(self):
        connection = await self.get_connection()
        collection_names = await connection.table_names()

        for collection_name in collection_names:
            collection = await self.get_collection(collection_name)
            await collection.delete("id IS NOT NULL")
            await connection.drop_table(collection_name)

        if self.url.startswith("/"):
            db_dir_path = path.dirname(self.url)
            db_file_name = path.basename(self.url)
            await get_file_storage(db_dir_path).remove_all(db_file_name)

    def get_data_point_schema(self, model_type: BaseModel):
        related_models_fields = []
        for field_name, field_config in model_type.model_fields.items():
            if hasattr(field_config, "model_fields"):
                related_models_fields.append(field_name)

            elif hasattr(field_config.annotation, "model_fields"):
                related_models_fields.append(field_name)

            elif (
                get_origin(field_config.annotation) == Union
                or get_origin(field_config.annotation) is list
            ):
                models_list = get_args(field_config.annotation)
                if any(hasattr(model, "model_fields") for model in models_list):
                    related_models_fields.append(field_name)
                elif models_list and any(get_args(model) is DataPoint for model in models_list):
                    related_models_fields.append(field_name)
                elif models_list and any(
                    submodel is DataPoint for submodel in get_args(models_list[0])
                ):
                    related_models_fields.append(field_name)

            elif get_origin(field_config.annotation) == Optional:
                model = get_args(field_config.annotation)
                if hasattr(model, "model_fields"):
                    related_models_fields.append(field_name)

        return copy_model(
            model_type,
            include_fields={
                "id": (str, ...),
            },
            exclude_fields=["metadata"] + related_models_fields,
        )
