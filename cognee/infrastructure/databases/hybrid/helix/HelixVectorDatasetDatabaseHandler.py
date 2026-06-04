"""Per-dataset vector handler for HelixDB (row-level tenancy).

The vector counterpart to :class:`HelixGraphDatasetDatabaseHandler`. HelixDB
stores vectors on the same tenant-scoped nodes as the graph, so a dataset maps to
``tenant_id = str(dataset_id)`` with no provisioning. ``delete_dataset`` prunes
the tenant (idempotent with the graph handler, since both share one store).
"""

from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.modules.users.models import User, DatasetDatabase


class HelixVectorDatasetDatabaseHandler:
    """Maps a dataset to a HelixDB row-level tenant for vector data.

    Duck-typed against ``DatasetDatabaseHandlerInterface`` (not subclassed, to
    avoid a circular import with the handler registry).
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        vector_config = get_vectordb_config()

        if vector_config.vector_db_provider != "helix":
            raise ValueError(
                "HelixVectorDatasetDatabaseHandler can only be used with the helix vector provider."
            )

        return {
            "vector_database_provider": "helix",
            "vector_database_url": vector_config.vector_db_url,
            "vector_database_name": str(dataset_id),
            "vector_database_key": vector_config.vector_db_key,
            "vector_dataset_database_handler": "helix_vector",
            "vector_database_connection_info": {},
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        vector_config = get_vectordb_config()
        dataset_database.vector_database_url = vector_config.vector_db_url
        dataset_database.vector_database_key = vector_config.vector_db_key
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        dataset_database = await cls.resolve_dataset_connection_info(dataset_database)

        from cognee.infrastructure.databases.hybrid.helix.HelixHybridAdapter import (
            HelixHybridAdapter,
        )

        adapter = HelixHybridAdapter(
            base_url=dataset_database.vector_database_url,
            api_key=dataset_database.vector_database_key,
            tenant_id=dataset_database.vector_database_name,
        )
        try:
            await adapter.prune()
        finally:
            await adapter.client.aclose()
