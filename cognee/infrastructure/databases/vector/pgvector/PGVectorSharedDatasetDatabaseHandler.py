from uuid import UUID
from typing import Optional

from cognee.modules.users.models import User
from cognee.modules.users.models import DatasetDatabase
from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.relational import get_relational_config
from cognee.infrastructure.databases.dataset_database_handler import (
    DatasetDatabaseHandlerInterface,
)
from cognee.infrastructure.databases.postgres import (
    create_pg_schema_if_not_exists,
    drop_pg_schema_if_exists,
    dataset_schema_name,
)
from cognee.infrastructure.databases.vector.create_vector_engine import evict_vector_engine


class PGVectorSharedDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """Per-dataset PGVector isolation via a Postgres *schema* in one shared database.

    The default ``PGVectorDatasetDatabaseHandler`` gives every dataset its own
    Postgres database (``CREATE DATABASE "<dataset_id>"``). This handler instead
    keeps each dataset's vector collections in a dedicated schema
    (``ds_<dataset_id>``) of cognee's main Postgres database. Trade-offs:

    - No ``CREATE DATABASE`` privilege required (only ``CREATE SCHEMA``).
    - One shared connection target, so no fan-out of separate per-database pools.
    - Cleanup is a single atomic ``DROP SCHEMA ... CASCADE``.

    Isolation is enforced by pinning the per-dataset PGVector engine's
    ``search_path`` to the dataset schema (see ``PGVectorAdapter``); the shared
    database is cognee's relational database, so all credentials/host/port come
    from the relational configuration.

    Selected with ``VECTOR_DATASET_DATABASE_HANDLER=pgvector_shared`` (requires
    ``VECTOR_DB_PROVIDER=pgvector`` and ``ENABLE_BACKEND_ACCESS_CONTROL=true``).
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        vector_config = get_vectordb_config()

        if vector_config.vector_db_provider != "pgvector":
            raise ValueError(
                "PGVectorSharedDatasetDatabaseHandler can only be used with the "
                "pgvector vector database provider."
            )

        relational_config = get_relational_config()
        schema = dataset_schema_name(dataset_id)
        db_name = relational_config.db_name
        host = relational_config.db_host
        port = relational_config.db_port

        # Create the dataset schema (and ensure the pgvector extension exists) in
        # the shared database. Idempotent, so re-running add/cognify is safe.
        await create_pg_schema_if_not_exists(
            db_name,
            schema,
            host=host,
            port=port,
            username=relational_config.db_username,
            password=relational_config.db_password,
            with_vector_extension=True,
        )

        return {
            "vector_database_provider": "pgvector",
            "vector_database_url": "",
            "vector_database_name": db_name,
            "vector_database_connection_info": {
                "host": host,
                "port": port,
                "schema": schema,
            },
            "vector_dataset_database_handler": "pgvector_shared",
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        # Credentials are never persisted in the relational DB; pull them from
        # the live relational config (the shared database for this mode).
        relational_config = get_relational_config()
        dataset_database.vector_database_connection_info["username"] = relational_config.db_username
        dataset_database.vector_database_connection_info["password"] = relational_config.db_password
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        dataset_database = await cls.resolve_dataset_connection_info(dataset_database)

        info = dataset_database.vector_database_connection_info
        schema = info["schema"]

        # Drop the cached adapter pinned to this schema (its reflection cache
        # would otherwise outlive the schema) before removing the schema itself.
        evict_vector_engine(
            vector_db_provider=dataset_database.vector_database_provider,
            vector_db_url=dataset_database.vector_database_url,
            vector_db_name=dataset_database.vector_database_name,
            vector_db_port=info["port"],
            vector_db_key=dataset_database.vector_database_key,
            vector_dataset_database_handler=dataset_database.vector_dataset_database_handler,
            vector_db_username=info["username"],
            vector_db_password=info["password"],
            vector_db_host=info["host"],
            vector_db_schema=schema,
        )

        await drop_pg_schema_if_exists(
            dataset_database.vector_database_name,
            schema,
            host=info["host"],
            port=info["port"],
            username=info["username"],
            password=info["password"],
        )
