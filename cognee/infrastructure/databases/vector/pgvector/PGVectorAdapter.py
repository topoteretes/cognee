import asyncio
from typing import List, Optional, get_type_hints, Dict, Any
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import JSON, Table, select, delete, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.exc import ProgrammingError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from asyncpg import DeadlockDetectedError, DuplicateTableError, UniqueViolationError


from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.DataPoint import MetaData as DataPointMetaData
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.relational import get_relational_engine

from distributed.utils import override_distributed
from distributed.tasks.queued_add_data_points import queued_add_data_points
from cognee.infrastructure.databases.exceptions import MissingQueryParameterError

from ...relational.ModelBase import Base
from ...relational.sqlalchemy.SqlAlchemyAdapter import SQLAlchemyAdapter
from ..utils import normalize_distances
from ..models.ScoredResult import ScoredResult
from ..exceptions import CollectionNotFoundError
from ..vector_db_interface import VectorDBInterface
from ..embeddings.EmbeddingEngine import EmbeddingEngine
from .serialize_data import serialize_data

logger = get_logger("PGVectorAdapter")


class IndexSchema(DataPoint):
    """
    Define a schema for indexing data points with a text field.

    This class inherits from the DataPoint class and specifies the structure of a single
    data point that includes a text attribute. It also includes a metadata field that
    indicates which fields should be indexed.
    """

    text: str

    metadata: DataPointMetaData = {"index_fields": ["text"], "type": "IndexSchema"}


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
        self.VECTOR_DB_LOCK = asyncio.Lock()

        relational_db = get_relational_engine()

        # If postgreSQL is used we must use the same engine and sessionmaker
        if relational_db.engine.dialect.name == "postgresql":
            self.engine = relational_db.engine
            self.sessionmaker = relational_db.sessionmaker
        else:
            # If not create new instances of engine and sessionmaker
            self.engine = create_async_engine(self.db_uri)
            self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)

        # Has to be imported at class level
        # Functions reading tables from database need to know what a Vector column type is
        from pgvector.sqlalchemy import Vector

        self.Vector = Vector

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """
        Embed a list of texts into vectors using the specified embedding engine.

        Parameters:
        -----------

            - data (list[str]): A list of strings to be embedded into vectors.

        Returns:
        --------

            - list[list[float]]: A list of lists of floats representing embedded vectors.
        """
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if a specified collection exists in the database.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check for existence.

        Returns:
        --------

            - bool: Returns True if the collection exists, False otherwise.
        """
        async with self.engine.begin() as connection:
            # Create a MetaData instance to load table information
            metadata = MetaData()
            # Load table information from schema into MetaData
            await connection.run_sync(metadata.reflect)

            if collection_name in metadata.tables:
                return True
            else:
                return False

    @retry(
        retry=retry_if_exception_type(
            (DuplicateTableError, UniqueViolationError, ProgrammingError)
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=6),
    )
    async def create_collection(
        self, collection_name: str, payload_schema: Optional[Any] = None
    ) -> None:
        vector_size = self.embedding_engine.get_vector_size()

        async with self.VECTOR_DB_LOCK:
            if not await self.has_collection(collection_name):

                class PGVectorDataPoint(Base):
                    """
                    Represent a point in a vector data space with associated data and vector representation.

                    This class inherits from Base and is associated with a database table defined by
                    __tablename__. It maintains the following public methods and instance variables:

                    - __init__(self, id, payload, vector): Initializes a new PGVectorDataPoint instance.

                    Instance variables:
                    - id: Identifier for the data point, defined by data_point_types.
                    - payload: JSON data associated with the data point.
                    - vector: Vector representation of the data point, with size defined by vector_size.
                    """

                    __tablename__ = collection_name
                    __table_args__ = {"extend_existing": True}
                    # PGVector requires one column to be the primary key
                    id: Mapped[str] = mapped_column(primary_key=True)
                    payload: Mapped[Dict[str, Any]] = mapped_column(JSON)
                    vector: Mapped[List[float]] = mapped_column(self.Vector(vector_size))

                    def __init__(
                        self, id: str, payload: Dict[str, Any], vector: List[float]
                    ) -> None:
                        self.id = id
                        self.payload = payload
                        self.vector = vector

                async with self.engine.begin() as connection:
                    if len(Base.metadata.tables.keys()) > 0:
                        from sqlalchemy import Table

                        table: Table = PGVectorDataPoint.__table__  # type: ignore
                        await connection.run_sync(Base.metadata.create_all, tables=[table])

    @retry(
        retry=retry_if_exception_type(DeadlockDetectedError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=6),
    )
    @override_distributed(queued_add_data_points)  # type: ignore
    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]) -> None:
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
            """
            Represents a data point in a PGVector database. This class maps to a table defined by
            the SQLAlchemy ORM.

            It contains the following public instance variables:
            - id: An identifier for the data point.
            - payload: A JSON object containing additional data related to the data point.
            - vector: A vector representation of the data point, configured to the specified size.
            """

            __tablename__ = collection_name
            __table_args__ = {"extend_existing": True}
            # PGVector requires one column to be the primary key
            id: Mapped[str] = mapped_column(primary_key=True)
            payload: Mapped[Dict[str, Any]] = mapped_column(JSON)
            vector: Mapped[List[float]] = mapped_column(self.Vector(vector_size))

            def __init__(self, id: str, payload: Dict[str, Any], vector: List[float]) -> None:
                self.id = id
                self.payload = payload
                self.vector = vector

        async with self.get_async_session() as session:
            pgvector_data_points = []

            for data_index, data_point in enumerate(data_points):
                # Check to see if data should be updated or a new data item should be created
                # data_point_db = (
                #     await session.execute(
                #         select(PGVectorDataPoint).filter(PGVectorDataPoint.id == data_point.id)
                #     )
                # ).scalar_one_or_none()

                # If data point exists update it, if not create a new one
                # if data_point_db:
                #     data_point_db.id = data_point.id
                #     data_point_db.vector = data_vectors[data_index]
                #     data_point_db.payload = serialize_data(data_point.model_dump())
                #     pgvector_data_points.append(data_point_db)
                # else:
                pgvector_data_points.append(
                    PGVectorDataPoint(
                        id=str(data_point.id),
                        vector=data_vectors[data_index],
                        payload=serialize_data(data_point.model_dump()),
                    )
                )

            def to_dict(obj: Any) -> Dict[str, Any]:
                return {
                    column.key: getattr(obj, column.key)
                    for column in inspect(obj).mapper.column_attrs
                }

            # session.add_all(pgvector_data_points)
            insert_statement = insert(PGVectorDataPoint).values(
                [to_dict(data_point) for data_point in pgvector_data_points]
            )
            insert_statement = insert_statement.on_conflict_do_nothing(index_elements=["id"])
            await session.execute(insert_statement)
            await session.commit()

    async def create_vector_index(self, index_name: str, index_property_name: str) -> None:
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: List[DataPoint]
    ) -> None:
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

    async def get_table(self, table_name: str, schema_name: Optional[str] = None) -> Table:
        """
        Dynamically loads a table using the given table name
        with an async engine. Schema parameter is ignored for vector collections.
        """
        collection_name = table_name
        async with self.engine.begin() as connection:
            # Create a MetaData instance to load table information
            metadata = MetaData()
            # Load table information from schema into MetaData
            await connection.run_sync(metadata.reflect)
            if collection_name in metadata.tables:
                return metadata.tables[collection_name]
            else:
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' not found!",
                )

    async def retrieve(self, collection_name: str, data_point_ids: List[str]) -> List[ScoredResult]:
        # Get PGVectorDataPoint Table from database
        PGVectorDataPoint = await self.get_table(collection_name)

        async with self.get_async_session() as session:
            query_result = await session.execute(
                select(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(data_point_ids))
            )
            results = query_result.all()

            return [
                ScoredResult(id=parse_id(result.id), payload=result.payload, score=0)
                for result in results
            ]

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 15,
        with_vector: bool = False,
    ) -> List[ScoredResult]:
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        # Get PGVectorDataPoint Table from database
        PGVectorDataPoint = await self.get_table(collection_name)

        # Use async session to connect to the database
        async with self.get_async_session() as session:
            query = select(
                PGVectorDataPoint,
                PGVectorDataPoint.c.vector.cosine_distance(query_vector).label("similarity"),
            ).order_by("similarity")

            if limit > 0:
                query = query.limit(limit)

            # Find closest vectors to query_vector
            query_results = await session.execute(query)

        vector_list = []

        # Extract distances and find min/max for normalization
        for vector in query_results.all():
            vector_list.append(
                {
                    "id": parse_id(str(vector.id)),
                    "payload": vector.payload,
                    "_distance": vector.similarity,
                }
            )

        if len(vector_list) == 0:
            return []

        # Normalize vector distance and add this as score information to vector_list
        normalized_values = normalize_distances(vector_list)
        for i in range(0, len(normalized_values)):
            vector_list[i]["score"] = normalized_values[i]

        # Create and return ScoredResult objects
        return [
            ScoredResult(id=row["id"], payload=row["payload"] or {}, score=row["score"])
            for row in vector_list
        ]

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: Optional[int] = None,
        with_vectors: bool = False,
    ) -> List[List[ScoredResult]]:
        query_vectors = await self.embedding_engine.embed_text(query_texts)

        return await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit or 15,
                    with_vector=with_vectors,
                )
                for query_vector in query_vectors
            ]
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: List[str]) -> Any:
        async with self.get_async_session() as session:
            # Get PGVectorDataPoint Table from database
            PGVectorDataPoint = await self.get_table(collection_name)
            results = await session.execute(
                delete(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(data_point_ids))
            )
            await session.commit()
            return results

    async def prune(self) -> None:
        # Clean up the database if it was set up as temporary
        await self.delete_database()
