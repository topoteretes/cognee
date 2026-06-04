"""Per-dataset graph handler for HelixDB (row-level tenancy).

HelixDB uses row-level multi-tenancy rather than per-dataset databases: every
node/edge carries a ``tenant_id`` property and reads are scoped by it. A dataset
therefore maps to ``tenant_id = str(dataset_id)`` with no provisioning step —
``create_dataset`` only records connection info + the tenant, and
``delete_dataset`` runs a tenant-scoped prune.
"""

from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.modules.users.models import User, DatasetDatabase


class HelixGraphDatasetDatabaseHandler:
    """Maps a dataset to a HelixDB row-level tenant for graph data.

    Duck-typed against ``DatasetDatabaseHandlerInterface`` (not subclassed, to
    avoid a circular import with the handler registry — same pattern as
    ``PostgresGraphDatasetDatabaseHandler``).
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "helix":
            raise ValueError(
                "HelixGraphDatasetDatabaseHandler can only be used with the helix graph provider."
            )

        # The tenant id is stored in graph_database_name and flows into the
        # adapter via the graph context config at connection time.
        return {
            "graph_database_provider": "helix",
            "graph_database_url": graph_config.graph_database_url,
            "graph_database_name": str(dataset_id),
            "graph_database_key": graph_config.graph_database_key,
            "graph_dataset_database_handler": "helix_graph",
            "graph_database_connection_info": {},
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        # URL/key are pulled from live config so secrets are never persisted.
        graph_config = get_graph_config()
        dataset_database.graph_database_url = graph_config.graph_database_url
        dataset_database.graph_database_key = graph_config.graph_database_key
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        dataset_database = await cls.resolve_dataset_connection_info(dataset_database)

        from cognee.infrastructure.databases.hybrid.helix.HelixHybridAdapter import (
            HelixHybridAdapter,
        )

        adapter = HelixHybridAdapter(
            base_url=dataset_database.graph_database_url,
            api_key=dataset_database.graph_database_key,
            tenant_id=dataset_database.graph_database_name,
        )
        try:
            await adapter.delete_graph()
        finally:
            await adapter.client.aclose()
