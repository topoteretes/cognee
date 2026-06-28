import os
from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine
from cognee.modules.users.models import User
from cognee.modules.users.models import DatasetDatabase
from cognee.base_config import get_base_config
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface
from cognee.infrastructure.files.storage.get_file_storage import get_file_storage


class TursoVectorDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for interacting with Turso / libSQL Dataset databases.

    Embedded mode gives every dataset its own libSQL file under the user's
    databases directory (``{dataset_id}.turso.db``), mirroring the LanceDB
    handler. That per-dataset file is what isolates one user/dataset's
    vectors from another's when ``ENABLE_BACKEND_ACCESS_CONTROL=True``.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        """Create local Turso dataset connection details for a user's dataset."""
        vector_config = get_vectordb_config()
        base_config = get_base_config()

        if vector_config.vector_db_provider != "turso":
            raise ValueError(
                "TursoVectorDatasetDatabaseHandler can only be used with Turso vector database provider."
            )

        databases_directory_path = os.path.join(
            base_config.system_root_directory, "databases", str(user.id)
        )
        await get_file_storage(databases_directory_path).ensure_directory_exists()

        vector_db_name = f"{dataset_id}.turso.db"

        return {
            "vector_database_provider": vector_config.vector_db_provider,
            "vector_database_url": os.path.join(databases_directory_path, vector_db_name),
            "vector_database_key": vector_config.vector_db_key,
            "vector_database_name": vector_db_name,
            "vector_dataset_database_handler": "turso",
        }

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase):
        """Drop the dataset's vector tables by pruning its dedicated libSQL file."""
        vector_engine = create_vector_engine(
            vector_db_provider=dataset_database.vector_database_provider,
            vector_db_url=dataset_database.vector_database_url,
            vector_db_key=dataset_database.vector_database_key,
            vector_db_name=dataset_database.vector_database_name,
        )
        await vector_engine.prune()
