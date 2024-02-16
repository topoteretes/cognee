"""This module provides functionalities for creating and managing databases."""

import asyncio
import os
import logging
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine
from relationaldb.models import memory, metadatas, operation, sessions, user, docs
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from relationaldb.database import (
    Base,DatabaseConfig)
from cognitive_architecture.config import Config
config = Config()
config.load()

load_dotenv()
logger = logging.getLogger(__name__)




class DatabaseManager:
    """Manages database creation, deletion, and table initialization."""
    def __init__(self, config: DatabaseConfig):
        """Initialize the DatabaseManager with a given configuration."""
        self.config = config
        self.engine = create_async_engine(config.get_sqlalchemy_database_url(), echo=True)
        self.db_type = config.db_type

    @asynccontextmanager
    async def get_connection(self):
        """Initialize the DatabaseManager with a given configuration."""
        if self.db_type in ["sqlite", "duckdb"]:
            # For SQLite and DuckDB, the engine itself manages connections
            yield self.engine
        else:
            async with self.engine.connect() as connection:
                yield connection

    async def database_exists(self, db_name):
        """Check if a database exists."""
        if self.db_type in ["sqlite", "duckdb"]:
            # For SQLite and DuckDB, check if the database file exists
            return os.path.exists(db_name)
        else:
            query = text(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'")
            async with self.get_connection() as connection:
                result = await connection.execute(query)
                return await result.fetchone() is not None

    async def create_database(self, db_name):
        """Create a new database."""
        if self.db_type not in ["sqlite", "duckdb"]:
            # For databases like PostgreSQL, create the database explicitly
            async with self.get_connection() as connection:
                await connection.execute(text(f"CREATE DATABASE {db_name}"))

    async def drop_database(self, db_name):
        """Drop an existing database."""
        if self.db_type in ["sqlite", "duckdb"]:
            # For SQLite and DuckDB, simply remove the database file
            os.remove(db_name)
        else:
            async with self.get_connection() as connection:
                await connection.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))

    async def create_tables(self):
        """Create tables based on the SQLAlchemy Base metadata."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    async def main():
        """Runs as a part of startup docker scripts to create the database and tables."""

        dbconfig = DatabaseConfig(db_type=config.db_type, db_name=config.db_name)
        db_manager = DatabaseManager(config=dbconfig)
        database_name = dbconfig.db_name

        if not await db_manager.database_exists(database_name):
            print(f"Database {database_name} does not exist. Creating...")
            await db_manager.create_database(database_name)
            print(f"Database {database_name} created successfully.")

        await db_manager.create_tables()

    asyncio.run(main())
#
# def create_admin_engine(username, password, host, database_name):
#     admin_url = f"postgresql://{username}:{password}@{host}:5432/{database_name}"
#     return create_engine(admin_url)
#
# def database_exists(connection, db_name):
#     query = text(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'")
#     result = connection.execute(query).fetchone()
#     return result is not None
#
# def create_database(connection, db_name):
#     connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
#     cursor = connection.cursor()
#     cursor.execute(f"CREATE DATABASE {db_name}")
#     cursor.close()
#
# def drop_database(connection, db_name):
#     connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
#     cursor = connection.cursor()
#     cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
#     cursor.close()
#
#
#
# def create_tables(engine):
#     Base.metadata.create_all(bind = engine)
#
# if __name__ == "__main__":
#     host = os.environ.get('POSTGRES_HOST')
#     username = os.environ.get('POSTGRES_USER')
#     password = os.environ.get('POSTGRES_PASSWORD')
#     database_name = os.environ.get('POSTGRES_DB')
#
#     engine = create_admin_engine(username, password, host, database_name)
#     connection = engine.connect()
#
#     # print(Base.metadata.tables)
#
#     if not database_exists(connection, database_name):
#         logger.info(f"Database {database_name} does not exist. Creating...")
#         create_database(connection, database_name)
#         logger.info(f"Database {database_name} created successfully.")
#
#     connection.close()
#     engine.dispose()
#
#     create_tables(engine)
