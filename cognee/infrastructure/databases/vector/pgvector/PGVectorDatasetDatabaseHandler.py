from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine
from cognee.infrastructure.databases.vector.pgvector.create_db_and_tables import delete_pg_database
from cognee.modules.users.models import User
from cognee.modules.users.models import DatasetDatabase
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface


class PGVectorDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for interacting with PGVector Dataset databases.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        vector_config = get_vectordb_config()

        if vector_config.vector_db_provider != "pgvector":
            raise ValueError(
                "PGVectorDatasetDatabaseHandler can only be used with PGVector vector database provider."
            )

        vector_db_name = f"{dataset_id}"

        new_vector_config = {
            "vector_database_provider": vector_config.vector_db_provider,
            "vector_database_url": vector_config.vector_db_url,
            "vector_database_name": vector_db_name,
            "vector_database_connection_info": {
                "port": vector_config.vector_db_port,
                "host": vector_config.vector_db_host,
            },
            "vector_dataset_database_handler": "pgvector",
        }

        from .create_db_and_tables import create_pg_database

        await create_pg_database(
            {
                "vector_db_provider": new_vector_config["vector_database_provider"],
                "vector_db_url": new_vector_config["vector_database_url"],
                "vector_db_name": new_vector_config["vector_database_name"],
                "vector_db_port": new_vector_config["vector_database_connection_info"]["port"],
                "vector_db_key": "",
                "vector_db_username": vector_config.vector_db_username,
                "vector_db_password": vector_config.vector_db_password,
                "vector_db_host": new_vector_config["vector_database_connection_info"]["host"],
                "vector_dataset_database_handler": "pgvector",
            }
        )

        return new_vector_config

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
        vector_engine = create_vector_engine(
            vector_db_provider=dataset_database.vector_database_provider,
            vector_db_url=dataset_database.vector_database_url,
            vector_db_name=dataset_database.vector_database_name,
            vector_db_port=dataset_database.vector_database_connection_info["port"],
            vector_db_username=dataset_database.vector_database_connection_info["username"],
            vector_db_password=dataset_database.vector_database_connection_info["password"],
        )
        # Prune data
        await vector_engine.prune()

        # Drop entire database
        await delete_pg_database(dataset_database)
