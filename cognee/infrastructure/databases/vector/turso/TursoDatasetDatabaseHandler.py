import os
from uuid import UUID
from typing import Optional

from cognee.modules.users.models import User
from cognee.modules.users.models import DatasetDatabase
from cognee.base_config import get_base_config
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine
from cognee.infrastructure.databases.dataset_database_handler import (
    DatasetDatabaseHandlerInterface,
)
from cognee.infrastructure.databases.relational import get_relational_config


class TursoDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """Handler for interacting with Turso/libSQL vector dataset databases.

    Follows the same pattern as LanceDBDatasetDatabaseHandler for local
    embedded databases, or PGVectorDatasetDatabaseHandler for remote Turso.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        """Create a Turso vector database for the given dataset.

        For remote Turso, uses the same database with per-dataset table prefixes.
        For local embedded libSQL, creates a per-dataset file.
        """
        vector_config = get_vectordb_config()
        relational_config = get_relational_config()

        if vector_config.vector_db_provider != "turso":
            raise ValueError(
                "TursoDatasetDatabaseHandler can only be used with Turso vector database provider."
            )

        # Build connection URL
        if relational_config.turso_url:
            # Remote Turso: use the same database URL
            vector_db_url = relational_config.turso_url
            vector_db_key = relational_config.turso_auth_token or ""
        else:
            # Local embedded: per-user directory with per-dataset file
            base_config = get_base_config()
            databases_dir = os.path.join(
                base_config.system_root_directory, "databases"
            )
            vector_db_url = databases_dir
            vector_db_key = ""

        vector_db_name = f"{dataset_id}"

        return {
            "vector_database_provider": vector_config.vector_db_provider,
            "vector_database_url": vector_db_url,
            "vector_database_key": vector_db_key,
            "vector_database_name": vector_db_name,
            "vector_dataset_database_handler": "turso",
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        """Resolve connection info — inject auth token from config at runtime."""
        relational_config = get_relational_config()
        if relational_config.turso_auth_token:
            dataset_database.vector_database_key = relational_config.turso_auth_token
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase):
        """Delete the vector data for the given dataset."""
        vector_config = get_vectordb_config()
        vector_engine = create_vector_engine(
            vector_db_provider=dataset_database.vector_database_provider,
            vector_db_url=dataset_database.vector_database_url,
            vector_db_key=dataset_database.vector_database_key,
            vector_db_name=dataset_database.vector_database_name,
        )
        await vector_engine.prune()
