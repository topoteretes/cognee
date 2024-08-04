import os
import asyncio
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, MetaData, Table, Column, String, Boolean, TIMESTAMP, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from cognee.infrastructure.files.storage import LocalStorage
from cognee.infrastructure.databases.relational.FakeAsyncSession import FakeAsyncSession

def make_async_sessionmaker(sessionmaker):
    @asynccontextmanager
    async def async_session_maker():
        await asyncio.sleep(0.1)
        yield FakeAsyncSession(sessionmaker())

    return async_session_maker

class SQLAlchemyAdapter():
    def __init__(self, db_type: str, db_path: str, db_name: str, db_user: str, db_password: str, db_host: str, db_port: str):
        self.db_location = os.path.abspath(os.path.join(db_path, db_name))

        if db_type == "duckdb":
            LocalStorage.ensure_directory_exists(db_path)

            self.engine = create_engine(f"duckdb:///{self.db_location}")
            self.sessionmaker = make_async_sessionmaker(sessionmaker(bind = self.engine))
        else:
            self.engine = create_async_engine(f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
            self.sessionmaker = async_sessionmaker(bind = self.engine, expire_on_commit = False)


    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        async_session_maker = self.sessionmaker
        async with async_session_maker() as session:
            yield session

    def get_session(self):
        session_maker = self.sessionmaker
        with session_maker() as session:
            yield session

    async def get_datasets(self):
        async with self.engine.connect() as connection:
            result = await connection.execute(text("SELECT DISTINCT schema_name FROM information_schema.tables;"))
            tables = [row["schema_name"] for row in result]
        return list(
            filter(
                lambda schema_name: not schema_name.endswith("staging") and schema_name != "cognee",
                tables
            )
        )

    def get_files_metadata(self, dataset_name: str):
        with self.engine.connect() as connection:
            result = connection.execute(text(f"SELECT id, name, file_path, extension, mime_type FROM {dataset_name}.file_metadata;"))
            return [dict(row) for row in result]

    def create_table(self, schema_name: str, table_name: str, table_config: list[dict]):
        fields_query_parts = [f"{item['name']} {item['type']}" for item in table_config]
        with self.engine.connect() as connection:
            connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name};"))
            connection.execute(text(f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} ({', '.join(fields_query_parts)});"))

    def delete_table(self, table_name: str):
        with self.engine.connect() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {table_name};"))

    def insert_data(self, schema_name: str, table_name: str, data: list[dict]):
        columns = ", ".join(data[0].keys())
        values = ", ".join([f"({', '.join([f':{key}' for key in row.keys()])})" for row in data])
        insert_query = text(f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES {values};")
        with self.engine.connect() as connection:
            connection.execute(insert_query, data)

    def get_data(self, table_name: str, filters: dict = None):
        with self.engine.connect() as connection:
            query = f"SELECT * FROM {table_name}"
            if filters:
                filter_conditions = " AND ".join([
                    f"{key} IN ({', '.join([f':{key}{i}' for i in range(len(value))])})" if isinstance(value, list)
                    else f"{key} = :{key}" for key, value in filters.items()
                ])
                query += f" WHERE {filter_conditions};"
                query = text(query)
                results = connection.execute(query, filters)
            else:
                query += ";"
                query = text(query)
                results = connection.execute(query)
            return {result["data_id"]: result["status"] for result in results}

    def execute_query(self, query):
        with self.engine.connect() as connection:
            result = connection.execute(text(query))
            return [dict(row) for row in result]

    def load_cognify_data(self, data):
        metadata = MetaData()

        cognify_table = Table(
            "cognify",
            metadata,
            Column("document_id", String),
            Column("created_at", TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")),
            Column("updated_at", TIMESTAMP, nullable=True, default=None),
            Column("processed", Boolean, default=False),
            Column("document_id_target", String, nullable=True)
        )

        metadata.create_all(self.engine)

        insert_query = cognify_table.insert().values(document_id=text(":document_id"))
        with self.engine.connect() as connection:
            connection.execute(insert_query, data)

    def fetch_cognify_data(self, excluded_document_id: str):
        with self.engine.connect() as connection:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS cognify (
                    document_id STRING,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    document_id_target STRING NULL
                );
            """))
            query = text("""
                SELECT document_id, created_at, updated_at, processed
                FROM cognify
                WHERE document_id != :excluded_document_id AND processed = FALSE;
            """)
            records = connection.execute(query, {"excluded_document_id": excluded_document_id}).fetchall()

            if records:
                document_ids = tuple(record["document_id"] for record in records)
                update_query = text("UPDATE cognify SET processed = TRUE WHERE document_id IN :document_ids;")
                connection.execute(update_query, {"document_ids": document_ids})
            return [dict(record) for record in records]

    def delete_cognify_data(self):
        with self.engine.connect() as connection:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS cognify (
                    document_id STRING,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    document_id_target STRING NULL
                );
            """))
            connection.execute(text("DELETE FROM cognify;"))
            connection.execute(text("DROP TABLE cognify;"))

    async def delete_database(self):
        async with self.engine.begin() as connection:
            from ..ModelBase import Base
            await connection.run_sync(Base.metadata.drop_all)
