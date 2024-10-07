from os import path
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy import text, select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from ..ModelBase import Base

class SQLAlchemyAdapter():
    def __init__(self, connection_string: str):
        self.db_path: str = None
        self.db_uri: str = connection_string

        self.engine = create_async_engine(connection_string)
        self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)

        if self.engine.dialect.name == "sqlite":
            self.db_path = connection_string.split("///")[1]

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        async_session_maker = self.sessionmaker
        async with async_session_maker() as session:
            try:
                yield session
            finally:
                await session.close()  # Ensure the session is closed

    def get_session(self):
        session_maker = self.sessionmaker
        with session_maker() as session:
            try:
                yield session
            finally:
                session.close()  # Ensure the session is closed

    async def get_datasets(self):
        from cognee.modules.data.models import Dataset

        async with self.get_async_session() as session:
            result = await session.execute(select(Dataset).options(joinedload(Dataset.data)))
            datasets = result.unique().scalars().all()
            return datasets

    async def create_table(self, schema_name: str, table_name: str, table_config: list[dict]):
        fields_query_parts = [f"{item['name']} {item['type']}" for item in table_config]
        async with self.engine.begin() as connection:
            await connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name};"))
            await connection.execute(text(f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} ({', '.join(fields_query_parts)});"))
            await connection.close()

    async def delete_table(self, table_name: str):
        async with self.engine.begin() as connection:
            await connection.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE;"))

            await connection.close()

    async def insert_data(self, schema_name: str, table_name: str, data: list[dict]):
        columns = ", ".join(data[0].keys())
        values = ", ".join([f"({', '.join([f':{key}' for key in row.keys()])})" for row in data])
        insert_query = text(f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES {values};")

        async with self.engine.begin() as connection:
            await connection.execute(insert_query, data)
            await connection.close()

    async def get_data(self, table_name: str, filters: dict = None):
        async with self.engine.begin() as connection:
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
        async with self.engine.begin() as connection:
            result = await connection.execute(text(query))
            return [dict(row) for row in result]

    async def drop_tables(self):
        async with self.engine.begin() as connection:
            try:
                await connection.execute(text("DROP TABLE IF EXISTS group_permission CASCADE"))
                await connection.execute(text("DROP TABLE IF EXISTS permissions CASCADE"))
                # Add more DROP TABLE statements for other tables as needed
                print("Database tables dropped successfully.")
            except Exception as e:
                print(f"Error dropping database tables: {e}")


    async def create_database(self):
        if self.engine.dialect.name == "sqlite":
            from cognee.infrastructure.files.storage import LocalStorage

            db_directory = path.dirname(self.db_path)
            LocalStorage.ensure_directory_exists(db_directory)

        async with self.engine.begin() as connection:
            if len(Base.metadata.tables.keys()) > 0:
                await connection.run_sync(Base.metadata.create_all)


    async def delete_database(self):
        try:
            if self.engine.dialect.name == "sqlite":
                from cognee.infrastructure.files.storage import LocalStorage

                LocalStorage.remove(self.db_path)
                self.db_path = None
            else:
                async with self.engine.begin() as connection:
                    for table in Base.metadata.sorted_tables:
                        drop_table_query = text(f"DROP TABLE IF EXISTS {table.name} CASCADE")
                        await connection.execute(drop_table_query)

        except Exception as e:
            print(f"Error deleting database: {e}")

        print("Database deleted successfully.")
