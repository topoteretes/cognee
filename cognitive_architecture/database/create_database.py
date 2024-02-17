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


