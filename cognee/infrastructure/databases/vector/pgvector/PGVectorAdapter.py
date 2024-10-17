from typing import List, Optional, get_type_hints, Any, Dict
from sqlalchemy import text, select
from sqlalchemy import JSON, Column, Table
from sqlalchemy.dialects.postgresql import ARRAY
from ..models.ScoredResult import ScoredResult

from ..vector_db_interface import VectorDBInterface, DataPoint
from sqlalchemy.orm import Mapped, mapped_column
from ..embeddings.EmbeddingEngine import EmbeddingEngine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from pgvector.sqlalchemy import Vector

from ...relational.sqlalchemy.SqlAlchemyAdapter import SQLAlchemyAdapter
from ...relational.ModelBase import Base

from datetime import datetime

# TODO: Find better location for function
def serialize_datetime(data):
    """Recursively convert datetime objects in dictionaries/lists to ISO format."""
    if isinstance(data, dict):
        return {key: serialize_datetime(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialize_datetime(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()  # Convert datetime to ISO 8601 string
    else:
        return data

class PGVectorAdapter(SQLAlchemyAdapter, VectorDBInterface):

    def __init__(self, connection_string: str,
        api_key: Optional[str],
        embedding_engine: EmbeddingEngine
    ):
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.db_uri: str = connection_string

        self.engine = create_async_engine(self.db_uri, echo=True)
        self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        async with self.engine.begin() as connection:
            #TODO: Switch to using ORM instead of raw query
            result = await connection.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            )
            tables = result.fetchall()
            for table in tables:
                if collection_name == table[0]:
                    return True
            return False

    async def create_collection(self, collection_name: str, payload_schema = None):
        data_point_types = get_type_hints(DataPoint)
        vector_size = self.embedding_engine.get_vector_size()

        if not await self.has_collection(collection_name):
            
            class PGVectorDataPoint(Base):
                __tablename__ = collection_name
                __table_args__ = {'extend_existing': True}
                # PGVector requires one column to be the primary key
                primary_key: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
                id: Mapped[data_point_types["id"]]
                payload = Column(JSON)
                vector = Column(Vector(vector_size))

                def __init__(self, id, payload, vector):
                    self.id = id
                    self.payload = payload
                    self.vector = vector

            async with self.engine.begin() as connection:
                if len(Base.metadata.tables.keys()) > 0:
                    await connection.run_sync(Base.metadata.create_all, tables=[PGVectorDataPoint.__table__])

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        async with self.get_async_session() as session:
            if not await self.has_collection(collection_name):
                await self.create_collection(
                    collection_name = collection_name,
                    payload_schema = type(data_points[0].payload),
                )

            data_vectors = await self.embed_data(
                [data_point.get_embeddable_data() for data_point in data_points]
            )

            vector_size = self.embedding_engine.get_vector_size()

            class PGVectorDataPoint(Base):
                __tablename__ = collection_name
                __table_args__ = {'extend_existing': True}
                # PGVector requires one column to be the primary key
                primary_key: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
                id: Mapped[type(data_points[0].id)]
                payload = Column(JSON)
                vector = Column(Vector(vector_size))

                def __init__(self, id, payload, vector):
                    self.id = id
                    self.payload = payload
                    self.vector = vector

            pgvector_data_points = [
                PGVectorDataPoint(
                    id = data_point.id,
                    vector = data_vectors[data_index],
                    payload = serialize_datetime(data_point.payload.dict())
                ) for (data_index, data_point) in enumerate(data_points)
            ]

            session.add_all(pgvector_data_points)
            await session.commit()

    async def retrieve(self, collection_name: str, data_point_ids: List[str]):
        async with AsyncSession(self.engine) as session:
            try:
                # Construct the SQL query
                # TODO: Switch to using ORM instead of raw query
                if len(data_point_ids) == 1:
                    query = text(f"SELECT * FROM {collection_name} WHERE id = :id")
                    result = await session.execute(query, {"id": data_point_ids[0]})
                else:
                    query = text(f"SELECT * FROM {collection_name} WHERE id = ANY(:ids)")
                    result = await session.execute(query, {"ids": data_point_ids})

                # Fetch all rows
                rows = result.fetchall()

                return [
                    ScoredResult(
                        id=row["id"],
                        payload=row["payload"],
                        score=0
                    )
                    for row in rows
                ]
            except Exception as e:
                print(f"Error retrieving data: {e}")
                return []

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 5,
        with_vector: bool = False,
    ) -> List[ScoredResult]:
        # Validate inputs
        if query_text is None and query_vector is None:
            raise ValueError("One of query_text or query_vector must be provided!")

        # Get the vector for query_text if provided
        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        # Use async session to connect to the database
        async with self.get_async_session() as session:
            try:
                PGVectorDataPoint = Table(collection_name, Base.metadata, autoload_with=self.engine)

                closest_items = await session.execute(select(PGVectorDataPoint, PGVectorDataPoint.c.vector.cosine_distance(query_vector).label('similarity')).order_by(PGVectorDataPoint.c.vector.cosine_distance(query_vector)).limit(limit))

                vector_list = []
                # Extract distances and find min/max for normalization
                for vector in closest_items:
                    #TODO: Add normalization of similarity score
                    vector_list.append(vector)

                # Create and return ScoredResult objects
                return [
                    ScoredResult(
                        id=str(row.id),
                        payload=row.payload,
                        score=row.similarity
                    )
                    for row in vector_list
                ]

            except Exception as e:
                print(f"Error during search: {e}")
                return []

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = None,
        with_vectors: bool = False,
    ):
        pass

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        pass

    async def prune(self):
        # Clean up the database if it was set up as temporary
        await self.delete_database()
