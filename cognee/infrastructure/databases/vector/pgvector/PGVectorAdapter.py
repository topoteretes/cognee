import asyncio
from typing import List, Optional, get_type_hints
from uuid import UUID

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import JSON, Column, Table, select, delete, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.infrastructure.engine import DataPoint

from ...relational.ModelBase import Base
from ...relational.sqlalchemy.SqlAlchemyAdapter import SQLAlchemyAdapter
from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.ScoredResult import ScoredResult
from ..utils import normalize_distances
from ..vector_db_interface import VectorDBInterface
from .serialize_data import serialize_data


class IndexSchema(DataPoint):
    text: str

    _metadata: dict = {"index_fields": ["text"], "type": "IndexSchema"}


class PGVectorAdapter(SQLAlchemyAdapter, VectorDBInterface):
    def __init__(
        self,
        connection_string: str,
        api_key: Optional[str],
        embedding_engine: EmbeddingEngine,
    ):
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.db_uri: str = connection_string
        self.engine = create_async_engine(self.db_uri)
        self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)

        # Has to be imported at class level
        # Functions reading tables from database need to know what a Vector column type is
        from pgvector.sqlalchemy import Vector

        self.Vector = Vector

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        async with self.engine.begin() as connection:
            # Create a MetaData instance to load table information
            metadata = MetaData()
            # Load table information from schema into MetaData
            await connection.run_sync(metadata.reflect)

            if collection_name in metadata.tables:
                return True
            else:
                return False

    async def create_collection(self, collection_name: str, payload_schema=None):
        data_point_types = get_type_hints(DataPoint)
        vector_size = self.embedding_engine.get_vector_size()

        if not await self.has_collection(collection_name):

            class PGVectorDataPoint(Base):
                __tablename__ = collection_name
                __table_args__ = {"extend_existing": True}
                # PGVector requires one column to be the primary key
                primary_key: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
                id: Mapped[data_point_types["id"]]
                payload = Column(JSON)
                vector = Column(self.Vector(vector_size))

                def __init__(self, id, payload, vector):
                    self.id = id
                    self.payload = payload
                    self.vector = vector

            async with self.engine.begin() as connection:
                if len(Base.metadata.tables.keys()) > 0:
                    await connection.run_sync(
                        Base.metadata.create_all, tables=[PGVectorDataPoint.__table__]
                    )

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        data_point_types = get_type_hints(DataPoint)
        if not await self.has_collection(collection_name):
            await self.create_collection(
                collection_name=collection_name,
                payload_schema=type(data_points[0]),
            )

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        vector_size = self.embedding_engine.get_vector_size()

        class PGVectorDataPoint(Base):
            __tablename__ = collection_name
            __table_args__ = {"extend_existing": True}
            # PGVector requires one column to be the primary key
            primary_key: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
            id: Mapped[data_point_types["id"]]
            payload = Column(JSON)
            vector = Column(self.Vector(vector_size))

            def __init__(self, id, payload, vector):
                self.id = id
                self.payload = payload
                self.vector = vector

        pgvector_data_points = [
            PGVectorDataPoint(
                id=data_point.id,
                vector=data_vectors[data_index],
                payload=serialize_data(data_point.model_dump()),
            )
            for (data_index, data_point) in enumerate(data_points)
        ]

        async with self.get_async_session() as session:
            session.add_all(pgvector_data_points)
            await session.commit()

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
                    text=DataPoint.get_embeddable_data(data_point),
                )
                for data_point in data_points
            ],
        )

    async def get_table(self, collection_name: str) -> Table:
        """
        Dynamically loads a table using the given collection name
        with an async engine.
        """
        async with self.engine.begin() as connection:
            # Create a MetaData instance to load table information
            metadata = MetaData()
            # Load table information from schema into MetaData
            await connection.run_sync(metadata.reflect)
            if collection_name in metadata.tables:
                return metadata.tables[collection_name]
            else:
                raise EntityNotFoundError(message=f"Table '{collection_name}' not found.")

    async def retrieve(self, collection_name: str, data_point_ids: List[str]):
        # Get PGVectorDataPoint Table from database
        PGVectorDataPoint = await self.get_table(collection_name)

        async with self.get_async_session() as session:
            results = await session.execute(
                select(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(data_point_ids))
            )
            results = results.all()

            return [
                ScoredResult(id=UUID(result.id), payload=result.payload, score=0)
                for result in results
            ]

    async def get_distance_from_collection_elements(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: List[float] = None,
        with_vector: bool = False,
    ) -> List[ScoredResult]:
        if query_text is None and query_vector is None:
            raise ValueError("One of query_text or query_vector must be provided!")

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        # Get PGVectorDataPoint Table from database
        PGVectorDataPoint = await self.get_table(collection_name)

        # Use async session to connect to the database
        async with self.get_async_session() as session:
            # Find closest vectors to query_vector
            closest_items = await session.execute(
                select(
                    PGVectorDataPoint,
                    PGVectorDataPoint.c.vector.cosine_distance(query_vector).label("similarity"),
                ).order_by("similarity")
            )

        vector_list = []

        # Extract distances and find min/max for normalization
        for vector in closest_items:
            # TODO: Add normalization of similarity score
            vector_list.append(vector)

        # Create and return ScoredResult objects
        return [
            ScoredResult(id=UUID(str(row.id)), payload=row.payload, score=row.similarity)
            for row in vector_list
        ]

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 5,
        with_vector: bool = False,
    ) -> List[ScoredResult]:
        if query_text is None and query_vector is None:
            raise InvalidValueError(message="One of query_text or query_vector must be provided!")

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        # Get PGVectorDataPoint Table from database
        PGVectorDataPoint = await self.get_table(collection_name)

        closest_items = []

        # Use async session to connect to the database
        async with self.get_async_session() as session:
            # Find closest vectors to query_vector
            closest_items = await session.execute(
                select(
                    PGVectorDataPoint,
                    PGVectorDataPoint.c.vector.cosine_distance(query_vector).label("similarity"),
                )
                .order_by("similarity")
                .limit(limit)
            )

        vector_list = []

        # Extract distances and find min/max for normalization
        for vector in closest_items:
            # TODO: Add normalization of similarity score
            vector_list.append(vector)

        # Create and return ScoredResult objects
        return [
            ScoredResult(id=UUID(str(row.id)), payload=row.payload, score=row.similarity)
            for row in vector_list
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
        async with self.get_async_session() as session:
            # Get PGVectorDataPoint Table from database
            PGVectorDataPoint = await self.get_table(collection_name)
            results = await session.execute(
                delete(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(data_point_ids))
            )
            await session.commit()
            return results

    async def prune(self):
        # Clean up the database if it was set up as temporary
        await self.delete_database()
