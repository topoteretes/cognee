from typing import List, Optional, get_type_hints, Generic, TypeVar
import asyncio
from ..models.ScoredResult import ScoredResult

from ..vector_db_interface import VectorDBInterface, DataPoint
from ..embeddings.EmbeddingEngine import EmbeddingEngine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from sqlalchemy.orm import DeclarativeBase, mapped_column
from pgvector.sqlalchemy import Vector

from ...relational.sqlalchemy.SqlAlchemyAdapter import SQLAlchemyAdapter


# Define the models
class Base(DeclarativeBase):
    pass

class PGVectorAdapter(SQLAlchemyAdapter, VectorDBInterface):
    async def create_vector_extension(self):
        async with self.get_async_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    def __init__(self, connection_string: str, 
        api_key: Optional[str],
        embedding_engine: EmbeddingEngine
    ):
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.db_uri: str = connection_string

        self.engine = create_async_engine(connection_string)
        self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)
        self.create_vector_extension()

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        async with self.engine.begin() as connection:
            collection_names = await connection.table_names()
            return collection_name in collection_names

    async def create_collection(self, collection_name: str, payload_schema = None):
        data_point_types = get_type_hints(DataPoint)
        vector_size = self.embedding_engine.get_vector_size()

        class PGVectorDataPoint(Base):
            id: Mapped[int] = mapped_column(data_point_types["id"], primary_key=True)
            vector = mapped_column(Vector(vector_size))
            payload: mapped_column(payload_schema)

        if not await self.has_collection(collection_name):
            async with self.engine.begin() as connection:
                return await connection.create_table(
                    name = collection_name,
                    schema = PGVectorDataPoint,
                    exist_ok = True,
                )

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        async with self.engine.begin() as connection:
            if not await self.has_collection(collection_name):
                await self.create_collection(
                    collection_name,
                    payload_schema = type(data_points[0].payload),
                )

            collection = await connection.open_table(collection_name)

            data_vectors = await self.embed_data(
                [data_point.get_embeddable_data() for data_point in data_points]
            )

            IdType = TypeVar("IdType")
            PayloadSchema = TypeVar("PayloadSchema")
            vector_size = self.embedding_engine.get_vector_size()

            class PGVectorDataPoint(Base, Generic[IdType, PayloadSchema]):
                id: Mapped[int] = mapped_column(IdType, primary_key=True)
                vector = mapped_column(Vector(vector_size))
                payload: mapped_column(PayloadSchema)

            pgvector_data_points = [
                PGVectorDataPoint[type(data_point.id), type(data_point.payload)](
                    id = data_point.id,
                    vector = data_vectors[data_index],
                    payload = data_point.payload,
                ) for (data_index, data_point) in enumerate(data_points)
            ]

            await collection.add(pgvector_data_points)

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        async with self.engine.begin() as connection:
            collection = await connection.open_table(collection_name)

            if len(data_point_ids) == 1:
                results = await collection.query().where(f"id = '{data_point_ids[0]}'").to_pandas()
            else:
                results = await collection.query().where(f"id IN {tuple(data_point_ids)}").to_pandas()

            return [ScoredResult(
                id = result["id"],
                payload = result["payload"],
                score = 0,
            ) for result in results.to_dict("index").values()]

    async def search(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: List[float] = None,
        limit: int = 5,
        with_vector: bool = False,
    ):
        if query_text is None and query_vector is None:
            raise ValueError("One of query_text or query_vector must be provided!")

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        async with self.engine.begin() as connection:
            collection = await connection.open_table(collection_name)

            results = await collection.vector_search(query_vector).limit(limit).to_pandas()

            result_values = list(results.to_dict("index").values())

            min_value = 100
            max_value = 0

            for result in result_values:
                value = float(result["_distance"])
                if value > max_value:
                    max_value = value
                if value < min_value:
                    min_value = value

            normalized_values = [(result["_distance"] - min_value) / (max_value - min_value) for result in result_values]

            return [ScoredResult(
                id = str(result["id"]),
                payload = result["payload"],
                score = normalized_values[value_index],
            ) for value_index, result in enumerate(result_values)]

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = None,
        with_vectors: bool = False,
    ):
        query_vectors = await self.embedding_engine.embed_text(query_texts)

        return asyncio.gather(
            *[self.search(
                collection_name = collection_name,
                query_vector = query_vector,
                limit = limit,
                with_vector = with_vectors,
            ) for query_vector in query_vectors]
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        async with self.engine.begin() as connection:
            collection = await connection.open_table(collection_name)
            results = await collection.delete(f"id IN {tuple(data_point_ids)}")
            return results

    async def prune(self):
        # Clean up the database if it was set up as temporary
        self.delete_database()
