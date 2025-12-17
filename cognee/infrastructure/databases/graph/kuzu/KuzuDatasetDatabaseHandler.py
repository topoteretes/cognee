import os
from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
from cognee.base_config import get_base_config
from cognee.modules.users.models import User
from cognee.modules.users.models import DatasetDatabase
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface


class KuzuDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for interacting with Kuzu Dataset databases.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        """
        Create a new Kuzu instance for the dataset. Return connection info that will be mapped to the dataset.

        Args:
            dataset_id: Dataset UUID
            user: User object who owns the dataset and is making the request

        Returns:
            dict: Connection details for the created Kuzu instance

        """
        from cognee.infrastructure.databases.graph.config import get_graph_config

        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "kuzu":
            raise ValueError(
                "KuzuDatasetDatabaseHandler can only be used with Kuzu graph database provider."
            )

        graph_db_name = f"{dataset_id}.pkl"
        graph_db_url = graph_config.graph_database_url
        graph_db_key = graph_config.graph_database_key
        graph_db_username = graph_config.graph_database_username
        graph_db_password = graph_config.graph_database_password

        return {
            "graph_database_name": graph_db_name,
            "graph_database_url": graph_db_url,
            "graph_database_provider": graph_config.graph_database_provider,
            "graph_database_key": graph_db_key,
            "graph_dataset_database_handler": "kuzu",
            "graph_database_connection_info": {
                "graph_database_username": graph_db_username,
                "graph_database_password": graph_db_password,
            },
        }

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase):
        base_config = get_base_config()
        databases_directory_path = os.path.join(
            base_config.system_root_directory, "databases", str(dataset_database.owner_id)
        )
        graph_file_path = os.path.join(
            databases_directory_path, dataset_database.graph_database_name
        )
        graph_engine = create_graph_engine(
            graph_database_provider=dataset_database.graph_database_provider,
            graph_database_url=dataset_database.graph_database_url,
            graph_database_name=dataset_database.graph_database_name,
            graph_database_key=dataset_database.graph_database_key,
            graph_file_path=graph_file_path,
            graph_database_username=dataset_database.graph_database_connection_info.get(
                "graph_database_username", ""
            ),
            graph_database_password=dataset_database.graph_database_connection_info.get(
                "graph_database_password", ""
            ),
            graph_dataset_database_handler="",
            graph_database_port="",
        )
        await graph_engine.delete_graph()
