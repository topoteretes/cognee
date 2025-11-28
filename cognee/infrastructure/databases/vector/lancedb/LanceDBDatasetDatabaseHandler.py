import os
from uuid import UUID
from typing import Optional

from cognee.modules.users.models import User
from cognee.base_config import get_base_config
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface


class LanceDBDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for interacting with LanceDB Dataset databases.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        vector_config = get_vectordb_config()
        base_config = get_base_config()

        if vector_config.vector_db_provider != "lancedb":
            raise ValueError(
                "LanceDBDatasetDatabaseHandler can only be used with LanceDB vector database provider."
            )

        databases_directory_path = os.path.join(
            base_config.system_root_directory, "databases", str(user.id)
        )

        vector_db_name = f"{dataset_id}.lance.db"

        return {
            "vector_database_name": vector_db_name,
            "vector_database_url": os.path.join(databases_directory_path, vector_db_name),
            "vector_database_provider": vector_config.vector_db_provider,
            "vector_database_key": vector_config.vector_db_key,
        }

    @classmethod
    async def delete_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]):
        pass
