import os
from os import path
import logging
from uuid import UUID
from typing import Optional
from typing import AsyncGenerator, List
from contextlib import asynccontextmanager
from sqlalchemy import text, select, MetaData, Table, delete
from sqlalchemy.orm import joinedload
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.data.models.Data import Data

from ..ModelBase import Base


logger = logging.getLogger(__name__)


class SQLAlchemyAdapter:
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
            await connection.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} ({', '.join(fields_query_parts)});"
                )
            )
            await connection.close()

    async def delete_table(self, table_name: str, schema_name: Optional[str] = "public"):
        async with self.engine.begin() as connection:
            if self.engine.dialect.name == "sqlite":
                # SQLite doesn’t support schema namespaces and the CASCADE keyword.
                # However, foreign key constraint can be defined with ON DELETE CASCADE during table creation.
                await connection.execute(text(f"DROP TABLE IF EXISTS {table_name};"))
            else:
                await connection.execute(
                    text(f"DROP TABLE IF EXISTS {schema_name}.{table_name} CASCADE;")
                )

    async def insert_data(self, schema_name: str, table_name: str, data: list[dict]):
        columns = ", ".join(data[0].keys())
        values = ", ".join([f"({', '.join([f':{key}' for key in row.keys()])})" for row in data])
        insert_query = text(f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES {values};")

        async with self.engine.begin() as connection:
            await connection.execute(insert_query, data)
            await connection.close()

    async def get_schema_list(self) -> List[str]:
        """
        Return a list of all schema names in database
        """
        if self.engine.dialect.name == "postgresql":
            async with self.engine.begin() as connection:
                result = await connection.execute(
                    text("""
                        SELECT schema_name FROM information_schema.schemata
                        WHERE schema_name NOT IN ('pg_catalog', 'pg_toast', 'information_schema');
                        """)
                )
                return [schema[0] for schema in result.fetchall()]
        return []

    async def delete_entity_by_id(
        self, table_name: str, data_id: UUID, schema_name: Optional[str] = "public"
    ):
        """
        Delete entity in given table based on id. Table must have an id Column.
        """
        if self.engine.dialect.name == "sqlite":
            async with self.get_async_session() as session:
                TableModel = await self.get_table(table_name, schema_name)

                # Foreign key constraints are disabled by default in SQLite (for backwards compatibility),
                # so must be enabled for each database connection/session separately.
                await session.execute(text("PRAGMA foreign_keys = ON;"))

                await session.execute(TableModel.delete().where(TableModel.c.id == data_id))
                await session.commit()
        else:
            async with self.get_async_session() as session:
                TableModel = await self.get_table(table_name, schema_name)
                await session.execute(TableModel.delete().where(TableModel.c.id == data_id))
                await session.commit()

    async def delete_data_entity(self, data_id: UUID):
        """
        Delete data and local files related to data if there are no references to it anymore.
        """
        async with self.get_async_session() as session:
            if self.engine.dialect.name == "sqlite":
                # Foreign key constraints are disabled by default in SQLite (for backwards compatibility),
                # so must be enabled for each database connection/session separately.
                await session.execute(text("PRAGMA foreign_keys = ON;"))

            try:
                data_entity = (await session.scalars(select(Data).where(Data.id == data_id))).one()
            except (ValueError, NoResultFound) as e:
                raise EntityNotFoundError(message=f"Entity not found: {str(e)}")

            # Check if other data objects point to the same raw data location
            raw_data_location_entities = (
                await session.execute(
                    select(Data.raw_data_location).where(
                        Data.raw_data_location == data_entity.raw_data_location
                    )
                )
            ).all()

            # Don't delete local file unless this is the only reference to the file in the database
            if len(raw_data_location_entities) == 1:
                # delete local file only if it's created by cognee
                from cognee.base_config import get_base_config

                config = get_base_config()

                if config.data_root_directory in raw_data_location_entities[0].raw_data_location:
                    if os.path.exists(raw_data_location_entities[0].raw_data_location):
                        os.remove(raw_data_location_entities[0].raw_data_location)
                    else:
                        # Report bug as file should exist
                        logger.error("Local file which should exist can't be found.")

            await session.execute(delete(Data).where(Data.id == data_id))
            await session.commit()

    async def get_table(self, table_name: str, schema_name: Optional[str] = "public") -> Table:
        """
        Dynamically loads a table using the given table name and schema name.
        """
        async with self.engine.begin() as connection:
            if self.engine.dialect.name == "sqlite":
                # Load the schema information into the MetaData object
                await connection.run_sync(Base.metadata.reflect)
                if table_name in Base.metadata.tables:
                    return Base.metadata.tables[table_name]
                else:
                    raise EntityNotFoundError(message=f"Table '{table_name}' not found.")
            else:
                # Create a MetaData instance to load table information
                metadata = MetaData()
                # Load table information from schema into MetaData
                await connection.run_sync(metadata.reflect, schema=schema_name)
                # Define the full table name
                full_table_name = f"{schema_name}.{table_name}"
                # Check if table is in list of tables for the given schema
                if full_table_name in metadata.tables:
                    return metadata.tables[full_table_name]
                raise EntityNotFoundError(message=f"Table '{full_table_name}' not found.")

    async def get_table_names(self) -> List[str]:
        """
        Return a list of all tables names in database
        """
        table_names = []
        async with self.engine.begin() as connection:
            if self.engine.dialect.name == "sqlite":
                await connection.run_sync(Base.metadata.reflect)
                for table in Base.metadata.tables:
                    table_names.append(str(table))
            else:
                schema_list = await self.get_schema_list()
                # Create a MetaData instance to load table information
                metadata = MetaData()
                # Drop all tables from all schemas
                for schema_name in schema_list:
                    # Load the schema information into the MetaData object
                    await connection.run_sync(metadata.reflect, schema=schema_name)
                    for table in metadata.sorted_tables:
                        table_names.append(str(table))
                    metadata.clear()
        return table_names

    async def get_data(self, table_name: str, filters: dict = None):
        async with self.engine.begin() as connection:
            query = f"SELECT * FROM {table_name}"
            if filters:
                filter_conditions = " AND ".join(
                    [
                        f"{key} IN ({', '.join([f':{key}{i}' for i in range(len(value))])})"
                        if isinstance(value, list)
                        else f"{key} = :{key}"
                        for key, value in filters.items()
                    ]
                )
                query += f" WHERE {filter_conditions};"
                query = text(query)
                results = await connection.execute(query, filters)
            else:
                query += ";"
                query = text(query)
                results = await connection.execute(query)
            return {result["data_id"]: result["status"] for result in results}

    async def get_all_data_from_table(self, table_name: str, schema: str = "public"):
        async with self.get_async_session() as session:
            # Validate inputs to prevent SQL injection
            if not table_name.isidentifier():
                raise ValueError("Invalid table name")
            if schema and not schema.isidentifier():
                raise ValueError("Invalid schema name")

            if self.engine.dialect.name == "sqlite":
                table = await self.get_table(table_name)
            else:
                table = await self.get_table(table_name, schema)

            # Query all data from the table
            query = select(table)
            result = await session.execute(query)

            # Fetch all rows as a list of dictionaries
            rows = result.mappings().all()
            return rows

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
            else:
                async with self.engine.begin() as connection:
                    schema_list = await self.get_schema_list()
                    # Create a MetaData instance to load table information
                    metadata = MetaData()
                    # Drop all tables from all schemas
                    for schema_name in schema_list:
                        # Load the schema information into the MetaData object
                        await connection.run_sync(metadata.reflect, schema=schema_name)
                        for table in metadata.sorted_tables:
                            drop_table_query = text(
                                f"DROP TABLE IF EXISTS {schema_name}.{table.name} CASCADE"
                            )
                            await connection.execute(drop_table_query)
                        metadata.clear()
        except Exception as e:
            print(f"Error deleting database: {e}")

        print("Database deleted successfully.")
