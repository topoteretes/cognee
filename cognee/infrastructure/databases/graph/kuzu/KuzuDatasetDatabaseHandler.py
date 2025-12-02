from uuid import UUID
from typing import Optional

from cognee.modules.users.models import User

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

        # TODO: Add graph file path info for kuzu (also in DatasetDatabase model)
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
            "graph_database_connection_info": {
                "graph_database_username": graph_db_username,
                "graph_database_password": graph_db_password,
            },
        }

    @classmethod
    async def delete_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]):
        pass
