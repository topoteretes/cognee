from uuid import UUID
from typing import Any, Optional

from sqlalchemy import text

from cognee.modules.users.models import User
from cognee.modules.users.models import DatasetDatabase
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface
from cognee.infrastructure.databases.vector.create_vector_engine import (
    create_vector_engine,
)
from cognee.infrastructure.databases.postgres import (
    create_pg_database_if_not_exists,
    drop_pg_database_if_exists,
)


class PGVectorDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for interacting with PGVector Dataset databases.
    """

    @classmethod
    async def create_dataset(
        cls, dataset_id: Optional[UUID], user: Optional[User], **kwargs: Any
    ) -> dict:
        if kwargs:
            raise ValueError(
                "PGVectorDatasetDatabaseHandler.create_dataset does not accept overrides; "
                f"got unsupported keys: {sorted(kwargs)}"
            )
        vector_config = get_vectordb_config()

        if vector_config.vector_db_provider != "pgvector":
            raise ValueError(
                "PGVectorDatasetDatabaseHandler can only be used with PGVector vector database provider."
            )

        vector_db_name = f"{dataset_id}"

        await create_pg_database_if_not_exists(
            vector_db_name,
            host=vector_config.vector_db_host,
            port=vector_config.vector_db_port,
            username=vector_config.vector_db_username,
            password=vector_config.vector_db_password,
        )

        new_vector_engine = create_vector_engine(
            vector_db_provider=vector_config.vector_db_provider,
            vector_db_url=vector_config.vector_db_url,
            vector_db_name=vector_db_name,
            vector_db_port=vector_config.vector_db_port,
            vector_db_key="",
            vector_db_username=vector_config.vector_db_username,
            vector_db_password=vector_config.vector_db_password,
            vector_db_host=vector_config.vector_db_host,
            vector_dataset_database_handler="pgvector",
        )
        async with new_vector_engine.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        return {
            "vector_database_provider": vector_config.vector_db_provider,
            "vector_database_url": vector_config.vector_db_url,
            "vector_database_name": vector_db_name,
            "vector_database_connection_info": {
                "port": vector_config.vector_db_port,
                "host": vector_config.vector_db_host,
            },
            "vector_dataset_database_handler": "pgvector",
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        vector_config = get_vectordb_config()
        # Note: For PGVector, we use the vector DB username/password from configuration so it's never stored in the DB
        dataset_database.vector_database_connection_info["username"] = (
            vector_config.vector_db_username
        )
        dataset_database.vector_database_connection_info["password"] = (
            vector_config.vector_db_password
        )
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase):
        dataset_database = await cls.resolve_dataset_connection_info(dataset_database)

        info = dataset_database.vector_database_connection_info
        await drop_pg_database_if_exists(
            dataset_database.vector_database_name,
            host=info["host"],
            port=info["port"],
            username=info["username"],
            password=info["password"],
        )

        vector_engine = create_vector_engine(
            vector_db_provider=dataset_database.vector_database_provider,
            vector_db_url=dataset_database.vector_database_url,
            vector_db_name=dataset_database.vector_database_name,
            vector_db_port=dataset_database.vector_database_connection_info["port"],
            vector_db_key=dataset_database.vector_database_key,
            vector_db_username=dataset_database.vector_database_connection_info["username"],
            vector_db_password=dataset_database.vector_database_connection_info["password"],
            vector_db_host=dataset_database.vector_database_connection_info["host"],
        )

        # Reset cached metadata from the vector adapter
        vector_engine.reset_metadata_cache()
