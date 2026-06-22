from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    evict_graph_engine,
)
from cognee.infrastructure.databases.postgres import (
    create_pg_database_if_not_exists,
    drop_pg_database_if_exists,
)
from cognee.modules.users.models import User, DatasetDatabase


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

        await create_pg_database_if_not_exists(
            graph_db_name,
            host=graph_config.graph_database_host,
            port=graph_config.graph_database_port,
            username=graph_config.graph_database_username,
            password=graph_config.graph_database_password,
        )

        engine = create_graph_engine(
            graph_database_provider="postgres",
            graph_file_path="",
            graph_database_name=graph_db_name,
            graph_database_username=graph_config.graph_database_username,
            graph_database_password=graph_config.graph_database_password,
            graph_database_host=graph_config.graph_database_host,
            graph_database_port=graph_config.graph_database_port,
        )
        await engine.initialize()

        return {
            "graph_database_provider": "postgres",
            "graph_database_url": "",
            "graph_database_name": graph_db_name,
            "graph_database_key": graph_config.graph_database_key,
            "graph_dataset_database_handler": "postgres_graph",
            "graph_database_connection_info": {
                "graph_database_host": graph_config.graph_database_host,
                "graph_database_port": graph_config.graph_database_port,
            },
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        # Credentials are never persisted in the DB; pull them from the live graph config.
        graph_config = get_graph_config()
        dataset_database.graph_database_connection_info["graph_database_username"] = (
            graph_config.graph_database_username
        )
        dataset_database.graph_database_connection_info["graph_database_password"] = (
            graph_config.graph_database_password
        )
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

        await drop_pg_database_if_exists(
            db_name,
            host=host,
            port=port,
            username=username,
            password=password,
        )
