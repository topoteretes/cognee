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
            result = await connection.execute(text("SELECT DISTINCT table_schema FROM information_schema.tables;"))
            tables = [row[0] for row in result]
        return list(
            filter(
                lambda table_schema: not table_schema.endswith("staging") and table_schema != "cognee",
                tables
            )
        )

    async def create_table(self, schema_name: str, table_name: str, table_config: list[dict]):
        fields_query_parts = [f"{item['name']} {item['type']}" for item in table_config]
        async with self.engine.begin() as connection:
            await connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name};"))
            await connection.execute(text(f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} ({', '.join(fields_query_parts)});"))

    async def get_files_metadata(self, dataset_name: str):
        async with self.engine.connect() as connection:
            result = await connection.execute(
                text(f"SELECT id, name, file_path, extension, mime_type FROM {dataset_name}.file_metadata;"))
            rows = result.fetchall()
            metadata = [{"id": row.id, "name": row.name, "file_path": row.file_path, "extension": row.extension,
                         "mime_type": row.mime_type} for row in rows]
            return metadata

    async def delete_table(self, table_name: str):
        async with self.engine.connect() as connection:
            await connection.execute(text(f"DROP TABLE IF EXISTS {table_name};"))

    async def insert_data(self, schema_name: str, table_name: str, data: list[dict]):
        columns = ", ".join(data[0].keys())
        values = ", ".join([f"({', '.join([f':{key}' for key in row.keys()])})" for row in data])
        insert_query = text(f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES {values};")
        async with self.engine.connect() as connection:
            await connection.execute(insert_query, data)

    async def get_data(self, table_name: str, filters: dict = None):
        async with self.engine.connect() as connection:
            query = f"SELECT * FROM {table_name}"
            if filters:
                filter_conditions = " AND ".join([
                    f"{key} IN ({', '.join([f':{key}{i}' for i in range(len(value))])})" if isinstance(value, list)
                    else f"{key} = :{key}" for key, value in filters.items()
                ])
                query += f" WHERE {filter_conditions};"
                query = text(query)
                results = await connection.execute(query, filters)
            else:
                query += ";"
                query = text(query)
                results = await connection.execute(query)
            return {result["data_id"]: result["status"] for result in results}

    async def execute_query(self, query):
        async with self.engine.connect() as connection:
            result = await connection.execute(text(query))
            return [dict(row) for row in result]

    async def load_cognify_data(self, data):
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

        async with self.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

        insert_query = cognify_table.insert().values(document_id=text(":document_id"))
        async with self.engine.connect() as connection:
            await connection.execute(insert_query, data)

    async def fetch_cognify_data(self, excluded_document_id: str):
        async with self.engine.connect() as connection:
            await connection.execute(text("""
                CREATE TABLE IF NOT EXISTS cognify (
                    document_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    document_id_target VARCHAR NULL
                );
            """))
            query = text("""
                SELECT document_id, created_at, updated_at, processed
                FROM cognify
                WHERE document_id != :excluded_document_id AND processed = FALSE;
            """)
            records = await connection.execute(query, {"excluded_document_id": excluded_document_id})
            records = records.fetchall()

            if records:
                document_ids = tuple(record["document_id"] for record in records)
                update_query = text("UPDATE cognify SET processed = TRUE WHERE document_id IN :document_ids;")
                await connection.execute(update_query, {"document_ids": document_ids})
            return [dict(record) for record in records]

    async def delete_cognify_data(self):
        async with self.engine.connect() as connection:
            await connection.execute(text("""
                CREATE TABLE IF NOT EXISTS cognify (
                    document_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    document_id_target VARCHAR NULL
                );
            """))
            await connection.execute(text("DELETE FROM cognify;"))
            await connection.execute(text("DROP TABLE cognify;"))

    async def drop_tables(self, connection):
        try:
            await connection.execute(text("DROP TABLE IF EXISTS group_permission CASCADE"))
            await connection.execute(text("DROP TABLE IF EXISTS permissions CASCADE"))
            # Add more DROP TABLE statements for other tables as needed
            print("Database tables dropped successfully.")
        except Exception as e:
            print(f"Error dropping database tables: {e}")

    async def delete_database(self):
        async with self.engine.begin() as connection:
            try:
                await self.drop_tables(connection)
                print("Database tables dropped successfully.")
            except Exception as e:
                print(f"Error dropping database tables: {e}")