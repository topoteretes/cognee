import os
from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.vector.create_vector_engine import (
    aevict_vector_engines_for_database,
)
from cognee.modules.users.models import User
from cognee.modules.users.models import DatasetDatabase
from cognee.base_config import get_base_config
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface
from cognee.infrastructure.files.storage.get_file_storage import get_file_storage


class LanceDBDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for interacting with LanceDB Dataset databases.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        """Create local LanceDB dataset connection details for a user's dataset."""
        vector_config = get_vectordb_config()
        base_config = get_base_config()

        if vector_config.vector_db_provider != "lancedb":
            raise ValueError(
                "LanceDBDatasetDatabaseHandler can only be used with LanceDB vector database provider."
            )

        databases_directory_path = os.path.join(
            base_config.system_root_directory, "databases", str(user.id)
        )
        await get_file_storage(databases_directory_path).ensure_directory_exists()

        vector_db_name = f"{dataset_id}.lance.db"

        return {
            "vector_database_provider": vector_config.vector_db_provider,
            "vector_database_url": os.path.join(databases_directory_path, vector_db_name),
            "vector_database_key": vector_config.vector_db_key,
            "vector_database_name": vector_db_name,
            "vector_dataset_database_handler": "lancedb",
        }

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase):
        # Never open the database to drop it: opening spawns a fresh engine
        # (in subprocess mode, a worker) that races the just-torn-down one.
        # Evict every cached engine for this database, wait for their closes
        # to fully finish, then remove the on-disk store directly.
        await aevict_vector_engines_for_database(dataset_database.vector_database_name)

        databases_directory_path = os.path.dirname(dataset_database.vector_database_url)
        file_storage = get_file_storage(databases_directory_path)
        await file_storage.remove_all(dataset_database.vector_database_name)
