import os
from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
from cognee.base_config import get_base_config
from cognee.modules.users.models import User
from cognee.modules.users.models import DatasetDatabase
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface


class LadybugDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for interacting with Ladybug Dataset databases.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        """
        Create a new Ladybug instance for the dataset. Return connection info that will be mapped to the dataset.

        Args:
            dataset_id: Dataset UUID
            user: User object who owns the dataset and is making the request

        Returns:
            dict: Connection details for the created Ladybug instance

        """
        from cognee.infrastructure.databases.graph.config import get_graph_config

        graph_config = get_graph_config()

        if graph_config.graph_database_provider not in ("ladybug", "kuzu"):
            raise ValueError(
                "LadybugDatasetDatabaseHandler can only be used with Ladybug graph database provider."
            )

        graph_db_name = (
            f"{dataset_id}.pkl"
            if graph_config.graph_database_provider == "kuzu"
            else f"{dataset_id}.lbug"
        )
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
            graph_database_username="",
            graph_database_password="",
            graph_dataset_database_handler="",
            graph_database_port="",
        )
        await graph_engine.delete_graph()
