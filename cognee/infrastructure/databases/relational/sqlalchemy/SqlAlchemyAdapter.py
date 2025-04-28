import os
from os import path
from cognee.shared.logging_utils import get_logger
from uuid import UUID
from typing import Optional
from typing import AsyncGenerator, List
from contextlib import asynccontextmanager
from sqlalchemy import text, select, MetaData, Table, delete, inspect
from sqlalchemy.orm import joinedload
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.data.models.Data import Data

from ..ModelBase import Base


logger = get_logger()


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
                    f'CREATE TABLE IF NOT EXISTS {schema_name}."{table_name}" ({", ".join(fields_query_parts)});'
                )
            )
            await connection.close()

    async def delete_table(self, table_name: str, schema_name: Optional[str] = "public"):
        async with self.engine.begin() as connection:
            if self.engine.dialect.name == "sqlite":
                # SQLite doesn't support schema namespaces and the CASCADE keyword.
                # However, foreign key constraint can be defined with ON DELETE CASCADE during table creation.
                await connection.execute(text(f'DROP TABLE IF EXISTS "{table_name}";'))
            else:
                await connection.execute(
                    text(f'DROP TABLE IF EXISTS {schema_name}."{table_name}" CASCADE;')
                )

    async def insert_data(
        self,
        table_name: str,
        data: list[dict],
        schema_name: Optional[str] = "public",
    ) -> int:
        """
        Insert data into specified table using SQLAlchemy Core with batch optimization
        Returns number of inserted rows

        Usage Example:
            from cognee.infrastructure.databases.relational.get_relational_engine import get_relational_engine
            from uuid import UUID
            db = get_relational_engine()
            table_name = "groups"
            data = {
                "id": UUID("c70a3cec-3309-44df-8ee6-eced820cf438"),
                "name": "test"
            }
            await db.insert_data(table_name, data)
        """
        if not data:
            logger.info("No data provided for insertion")
            return 0

        try:
            # Use SQLAlchemy Core insert with execution options
            async with self.engine.begin() as conn:
                # Dialect-agnostic table reference
                if self.engine.dialect.name == "sqlite":
                    # Foreign key constraints are disabled by default in SQLite (for backwards compatibility),
                    # so must be enabled for each database connection/session separately.
                    await conn.execute(text("PRAGMA foreign_keys=ON"))
                    table = await self.get_table(table_name)  # SQLite ignores schemas
                else:
                    table = await self.get_table(table_name, schema_name)

                result = await conn.execute(table.insert().values(data))

                # Return rowcount for validation
                return result.rowcount

        except Exception as e:
            logger.error(f"Insert failed: {str(e)}")
            raise e  # Re-raise for error handling upstream

    async def get_schema_list(self) -> List[str]:
        """
        Return a list of all schema names in database
        """
        if self.engine.dialect.name == "postgresql":
            async with self.engine.begin() as connection:
                result = await connection.execute(
                    text(
                        """
                        SELECT schema_name FROM information_schema.schemata
                        WHERE schema_name NOT IN ('pg_catalog', 'pg_toast', 'information_schema');
                        """
                    )
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
        Return a list of all table names in the database, even if they don't have defined SQLAlchemy models.
        """
        table_names = []
        async with self.engine.begin() as connection:
            if self.engine.dialect.name == "sqlite":
                # Use a new MetaData instance to reflect all tables
                metadata = MetaData()
                await connection.run_sync(metadata.reflect)  # Reflect the entire database
                table_names = list(metadata.tables.keys())  # Get table names
            else:
                schema_list = await self.get_schema_list()
                metadata = MetaData()
                for schema_name in schema_list:
                    await connection.run_sync(metadata.reflect, schema=schema_name)
                    table_names.extend(metadata.tables.keys())  # Append table names from schema
                    metadata.clear()  # Clear metadata for the next schema

        return table_names

    async def get_data(self, table_name: str, filters: dict = None):
        async with self.engine.begin() as connection:
            query = f'SELECT * FROM "{table_name}"'
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
                logger.debug("Database tables dropped successfully.")
            except Exception as e:
                logger.error(f"Error dropping database tables: {e}")
                raise e

    async def create_database(self):
        if self.engine.dialect.name == "sqlite" and not os.path.exists(self.db_path):
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

                await self.engine.dispose(close=True)
                with open(self.db_path, "w") as file:
                    file.write("")
            else:
                async with self.engine.begin() as connection:
                    # Create a MetaData instance to load table information
                    metadata = MetaData()
                    # Drop all tables from the public schema
                    schema_list = ["public", "public_staging"]
                    for schema_name in schema_list:
                        # Load the schema information into the MetaData object
                        await connection.run_sync(metadata.reflect, schema=schema_name)
                        for table in metadata.sorted_tables:
                            drop_table_query = text(
                                f'DROP TABLE IF EXISTS {schema_name}."{table.name}" CASCADE'
                            )
                            await connection.execute(drop_table_query)
                        metadata.clear()
        except Exception as e:
            logger.error(f"Error deleting database: {e}")
            raise e

        logger.info("Database deleted successfully.")

    async def extract_schema(self):
        async with self.engine.begin() as connection:
            tables = await self.get_table_names()

            schema = {}

            if self.engine.dialect.name == "sqlite":
                for table_name in tables:
                    schema[table_name] = {"columns": [], "primary_key": None, "foreign_keys": []}

                    # Get column details
                    columns_result = await connection.execute(
                        text(f"PRAGMA table_info('{table_name}');")
                    )
                    columns = columns_result.fetchall()
                    for column in columns:
                        column_name = column[1]
                        column_type = column[2]
                        is_pk = column[5] == 1
                        schema[table_name]["columns"].append(
                            {"name": column_name, "type": column_type}
                        )
                        if is_pk:
                            schema[table_name]["primary_key"] = column_name

                    # Get foreign key details
                    foreign_keys_results = await connection.execute(
                        text(f"PRAGMA foreign_key_list('{table_name}');")
                    )
                    foreign_keys = foreign_keys_results.fetchall()
                    for fk in foreign_keys:
                        schema[table_name]["foreign_keys"].append(
                            {
                                "column": fk[3],  # Column in the current table
                                "ref_table": fk[2],  # Referenced table
                                "ref_column": fk[4],  # Referenced column
                            }
                        )
            else:
                schema_list = await self.get_schema_list()
                for schema_name in schema_list:
                    # Get tables for the current schema via the inspector.
                    tables = await connection.run_sync(
                        lambda sync_conn: inspect(sync_conn).get_table_names(schema=schema_name)
                    )
                    for table_name in tables:
                        # Optionally, qualify the table name with the schema if not in the default schema.
                        key = (
                            table_name if schema_name == "public" else f"{schema_name}.{table_name}"
                        )
                        schema[key] = {"columns": [], "primary_key": None, "foreign_keys": []}

                        # Helper function to get table details using the inspector.
                        def get_details(sync_conn, table, schema_name):
                            insp = inspect(sync_conn)
                            cols = insp.get_columns(table, schema=schema_name)
                            pk = insp.get_pk_constraint(table, schema=schema_name)
                            fks = insp.get_foreign_keys(table, schema=schema_name)
                            return cols, pk, fks

                        cols, pk, fks = await connection.run_sync(
                            get_details, table_name, schema_name
                        )

                        for column in cols:
                            # Convert the type to string
                            schema[key]["columns"].append(
                                {"name": column["name"], "type": str(column["type"])}
                            )
                        pk_columns = pk.get("constrained_columns", [])
                        if pk_columns:
                            schema[key]["primary_key"] = pk_columns[0]
                        for fk in fks:
                            for col, ref_col in zip(
                                fk.get("constrained_columns", []), fk.get("referred_columns", [])
                            ):
                                if col and ref_col:
                                    schema[key]["foreign_keys"].append(
                                        {
                                            "column": col,
                                            "ref_table": fk.get("referred_table"),
                                            "ref_column": ref_col,
                                        }
                                    )
                                else:
                                    logger.warning(
                                        f"Missing value in foreign key information. \nColumn value: {col}\nReference column value: {ref_col}\n"
                                    )

            return schema
