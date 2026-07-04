import os
from uuid import UUID
from typing import Optional

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    evict_graph_engine,
)
from cognee.modules.users.models import User, DatasetDatabase


class TursoGraphDatasetDatabaseHandler:
    """Handler for per-dataset Turso/SQLite graph databases.

    Local mode: each dataset gets its own SQLite file under
    ``<system_root>/databases/<user_id>/<dataset_id>``.

    Remote Turso: set GRAPH_DATABASE_URL to the remote libSQL URL and
    GRAPH_DATABASE_KEY to the auth token. Each dataset still gets its own local
    replica file at the same path, kept in sync via ``turso.aio.sync.connect()``,
    but (v1 limitation) all datasets sync against the same remote database/token —
    there is no per-dataset remote isolation yet.
    """

    @classmethod
    def _local_path(cls, user_id: UUID, dataset_id: UUID) -> str:
        base_config = get_base_config()
        databases_dir = os.path.join(base_config.system_root_directory, "databases", str(user_id))
        os.makedirs(databases_dir, exist_ok=True)
        return os.path.join(databases_dir, str(dataset_id))

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "turso":
            raise ValueError(
                "TursoGraphDatasetDatabaseHandler can only be used "
                "with the turso graph database provider."
            )

        local_path = cls._local_path(user.id, dataset_id)
        remote_url = graph_config.graph_database_url
        auth_token = graph_config.graph_database_key

        if auth_token:
            engine = create_graph_engine(
                graph_database_provider="turso",
                graph_file_path=local_path,
                graph_database_url=remote_url,
                graph_database_key=auth_token,
                graph_database_sync_interval=graph_config.graph_database_sync_interval,
            )
            dataset_url_field = remote_url
        else:
            engine = create_graph_engine(
                graph_database_provider="turso",
                graph_file_path="",
                graph_database_url=local_path,
                graph_database_key="",
            )
            dataset_url_field = local_path

        await engine.initialize()

        return {
            "graph_database_provider": "turso",
            "graph_database_url": dataset_url_field,
            "graph_database_name": str(dataset_id),
            "graph_database_key": auth_token,
            "graph_dataset_database_handler": "turso_graph",
            "graph_database_connection_info": {},
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        graph_config = get_graph_config()
        if graph_config.graph_database_key:
            dataset_database.graph_database_connection_info["graph_database_key"] = (
                graph_config.graph_database_key
            )
            # Backfill remote credentials for datasets created before remote sync
            # was configured, so they can adopt it without a data migration.
            if not dataset_database.graph_database_key:
                dataset_database.graph_database_key = graph_config.graph_database_key
            if not dataset_database.graph_database_url:
                dataset_database.graph_database_url = graph_config.graph_database_url
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        dataset_database = await cls.resolve_dataset_connection_info(dataset_database)

        local_path = cls._local_path(
            dataset_database.owner_id, UUID(dataset_database.graph_database_name)
        )
        auth_token = dataset_database.graph_database_key

        if auth_token:
            engine_kwargs = dict(
                graph_database_provider="turso",
                graph_file_path=local_path,
                graph_database_url=dataset_database.graph_database_url,
                graph_database_key=auth_token,
                graph_database_sync_interval=get_graph_config().graph_database_sync_interval,
            )
        else:
            engine_kwargs = dict(
                graph_database_provider="turso",
                graph_file_path="",
                graph_database_url=local_path,
                graph_database_key="",
            )

        # Explicitly empty the graph (and, in remote mode, push that empty state to
        # the shared remote) before evicting — all datasets share one remote
        # database in v1, so simply removing the local replica would otherwise
        # leave this dataset's rows permanently on the remote.
        engine = create_graph_engine(**engine_kwargs)
        try:
            await engine.delete_graph()
        finally:
            evict_graph_engine(**engine_kwargs)
            await engine.close()

        if os.path.exists(local_path):
            os.remove(local_path)
