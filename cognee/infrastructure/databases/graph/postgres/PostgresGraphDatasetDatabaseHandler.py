from uuid import UUID
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    evict_graph_engine,
)
from cognee.modules.users.models import User, DatasetDatabase


def _resolve_pg_credentials(graph_config) -> tuple[str, str, str, str]:
    """Return (host, port, username, password), falling back to the relational config."""
    host = graph_config.graph_database_host
    port = graph_config.graph_database_port
    username = graph_config.graph_database_username
    password = graph_config.graph_database_password

    if not (host and port and username and password):
        from cognee.infrastructure.databases.relational import get_relational_config

        relational_config = get_relational_config()
        host = host or relational_config.db_host
        port = port or relational_config.db_port
        username = username or relational_config.db_username
        password = password or relational_config.db_password

    return host, str(port), username, password


class PostgresGraphDatasetDatabaseHandler:
    """Handler for per-dataset Postgres graph databases."""

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "postgres":
            raise ValueError(
                "PostgresGraphDatasetDatabaseHandler can only be used "
                "with postgres graph database provider."
            )

        graph_db_name = f"{dataset_id}"
        host, port, _, _ = _resolve_pg_credentials(graph_config)

        new_graph_config = {
            "graph_database_provider": "postgres",
            "graph_database_url": "",
            "graph_database_name": graph_db_name,
            "graph_database_key": graph_config.graph_database_key,
            "graph_dataset_database_handler": "postgres_graph",
            "graph_database_connection_info": {
                "graph_database_host": host,
                "graph_database_port": port,
            },
        }

        await cls._create_pg_database(graph_db_name)

        return new_graph_config

    @classmethod
    async def _create_pg_database(cls, db_name: str) -> None:
        """Create the per-dataset Postgres database and initialize its tables."""
        graph_config = get_graph_config()
        host, port, username, password = _resolve_pg_credentials(graph_config)

        maintenance_url = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/postgres"

        maintenance_engine = create_async_engine(maintenance_url)
        try:
            connection = await maintenance_engine.connect()
            try:
                connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
                result = await connection.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db"),
                    {"db": db_name},
                )
                if not result.scalar():
                    await connection.execute(text(f'CREATE DATABASE "{db_name}";'))
            finally:
                await connection.close()
        finally:
            await maintenance_engine.dispose()

        engine = create_graph_engine(
            graph_database_provider="postgres",
            graph_file_path="",
            graph_database_name=db_name,
            graph_database_username=username,
            graph_database_password=password,
            graph_database_host=host,
            graph_database_port=port,
        )
        await engine.initialize()

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        # Credentials are never persisted in the DB; pull them from the live graph config.
        graph_config = get_graph_config()
        _, _, username, password = _resolve_pg_credentials(graph_config)
        dataset_database.graph_database_connection_info["graph_database_username"] = username
        dataset_database.graph_database_connection_info["graph_database_password"] = password
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        dataset_database = await cls.resolve_dataset_connection_info(dataset_database)

        info = dataset_database.graph_database_connection_info
        host = info.get("graph_database_host", "")
        port = info.get("graph_database_port", "")
        username = info.get("graph_database_username", "")
        password = info.get("graph_database_password", "")
        db_name = dataset_database.graph_database_name

        evict_graph_engine(
            graph_database_provider="postgres",
            graph_file_path="",
            graph_database_name=db_name,
            graph_database_username=username,
            graph_database_password=password,
            graph_database_host=host,
            graph_database_port=port,
        )

        maintenance_url = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/postgres"

        maintenance_engine = create_async_engine(maintenance_url)
        try:
            connection = await maintenance_engine.connect()
            try:
                connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
                await connection.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) "
                        "FROM pg_stat_activity "
                        "WHERE datname = :db AND pid <> pg_backend_pid()"
                    ),
                    {"db": db_name},
                )
                await connection.execute(text(f'DROP DATABASE IF EXISTS "{db_name}";'))
            finally:
                await connection.close()
        finally:
            await maintenance_engine.dispose()
