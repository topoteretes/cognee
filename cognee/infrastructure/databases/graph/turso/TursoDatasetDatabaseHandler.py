import os
from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    evict_graph_engine,
)
from cognee.base_config import get_base_config
from cognee.modules.users.models import User, DatasetDatabase
from cognee.infrastructure.databases.dataset_database_handler.dataset_database_handler_interface import DatasetDatabaseHandlerInterface


class TursoDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for interacting with Turso Dataset databases.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "turso":
            raise ValueError(
                "TursoDatasetDatabaseHandler can only be used with turso graph database provider."
            )

        graph_db_name = f"{dataset_id}.sqlite"
        graph_db_url = graph_config.graph_database_url
        graph_db_key = graph_config.graph_database_key

        return {
            "graph_database_name": graph_db_name,
            "graph_database_url": graph_db_url,
            "graph_database_provider": graph_config.graph_database_provider,
            "graph_database_key": graph_db_key,
            "graph_dataset_database_handler": graph_config.graph_dataset_database_handler,
            "graph_database_connection_info": {},
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        base_config = get_base_config()
        databases_directory_path = os.path.join(
            base_config.system_root_directory, "databases", str(dataset_database.owner_id)
        )
        graph_file_path = os.path.join(
            databases_directory_path, dataset_database.graph_database_name
        )
        graph_config = get_graph_config()
        
        engine_kwargs = dict(
            graph_database_provider=dataset_database.graph_database_provider,
            graph_database_url=dataset_database.graph_database_url,
            graph_database_name=dataset_database.graph_database_name,
            graph_database_key=dataset_database.graph_database_key,
            graph_file_path=graph_file_path,
            graph_database_username="",
            graph_database_password="",
            graph_dataset_database_handler="",
            graph_database_port="",
        )
        
        # We don't implement delete_graph in TursoAdapter as it's not in GraphDBInterface.
        # Just evict and delete the sqlite file if it's local.
        evict_graph_engine(**engine_kwargs)

        if os.path.exists(graph_file_path):
            os.remove(graph_file_path)
