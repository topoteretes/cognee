from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine
from cognee.modules.users.models import User
from cognee.modules.users.models import DatasetDatabase
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface


class PGVectorDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for interacting with PGVector Dataset databases.
    """

    @classmethod
    async def _create_pg_database(cls, vector_config):
        """
        Create the necessary Postgres database, and the PGVector extension on it.
        This is defined here because the creation needs the latest vector config,
        which is not yet saved in the vector config context variable here.
        """
        from cognee.infrastructure.databases.relational.create_relational_engine import (
            create_relational_engine,
        )

        from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine
        from sqlalchemy import text

        pg_relational_engine = create_relational_engine(
            db_path="",
            db_host=vector_config["vector_db_url"],
            db_name=vector_config["vector_db_name"],
            db_port=vector_config["vector_db_port"],
            db_username=vector_config["vector_db_username"],
            db_password=vector_config["vector_db_password"],
            db_provider="postgres",
        )
        await pg_relational_engine.create_database()

        vector_engine = create_vector_engine(**vector_config)
        async with vector_engine.engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

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
                "username": vector_config.vector_db_username,
                "password": vector_config.vector_db_password,
            },
            "vector_dataset_database_handler": "pgvector",
        }

        await cls._create_pg_database(
            {
                "vector_db_provider": new_vector_config["vector_database_provider"],
                "vector_db_url": new_vector_config["vector_database_url"],
                "vector_db_name": new_vector_config["vector_database_name"],
                "vector_db_port": new_vector_config["vector_database_connection_info"]["port"],
                "vector_db_key": "",
                "vector_db_username": new_vector_config["vector_database_connection_info"][
                    "username"
                ],
                "vector_db_password": new_vector_config["vector_database_connection_info"][
                    "password"
                ],
                "vector_dataset_database_handler": "pgvector",
            }
        )

        return new_vector_config

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase):
        vector_engine = create_vector_engine(
            vector_db_provider=dataset_database.vector_database_provider,
            vector_db_url=dataset_database.vector_database_url,
            vector_db_name=dataset_database.vector_database_name,
            vector_db_port=dataset_database.vector_database_connection_info["port"],
            vector_db_username=dataset_database.vector_database_connection_info["username"],
            vector_db_password=dataset_database.vector_database_connection_info["password"],
        )
        await vector_engine.prune()
