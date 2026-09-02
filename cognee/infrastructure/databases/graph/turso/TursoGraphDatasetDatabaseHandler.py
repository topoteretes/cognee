import os
from uuid import UUID
from typing import Optional

from cognee.base_config import get_base_config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    graph_engine_cache,
)
from cognee.modules.users.models import User, DatasetDatabase


class TursoGraphDatasetDatabaseHandler:
    """Handler for per-dataset Turso/libSQL graph databases.

    Each dataset gets its own libSQL file under the system databases directory, so
    the existing multi-user permission system isolates datasets by file.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "turso":
            raise ValueError(
                "TursoGraphDatasetDatabaseHandler can only be used "
                "with the turso graph database provider."
            )

        base_config = get_base_config()
        databases_dir = os.path.join(base_config.system_root_directory, "databases")

        # This is delete_dataset's own os.path.isabs guard, evaluated at
        # creation time: a non-absolute path would be silently skipped there
        # (and by prune_system, which routes through it), leaving a file
        # nothing can remove. Runs before makedirs so a non-local root such as
        # an s3:// one creates no local directory on the way out.
        if not os.path.isabs(databases_dir):
            raise EnvironmentError(
                "Turso per-dataset graph databases need an absolute local path; set "
                f"SYSTEM_ROOT_DIRECTORY to one (got {base_config.system_root_directory!r})."
            )

        os.makedirs(databases_dir, exist_ok=True)

        dataset_url = os.path.join(databases_dir, f"graph_{dataset_id}.db")

        engine = create_graph_engine(
            graph_database_provider="turso",
            graph_file_path="",
            graph_database_url=dataset_url,
            graph_database_key="",
        )
        await engine.initialize()

        return {
            "graph_database_provider": "turso",
            "graph_database_url": dataset_url,
            "graph_database_name": str(dataset_id),
            "graph_database_key": "",
            "graph_dataset_database_handler": "turso_graph",
            "graph_database_connection_info": {},
        }

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        # A local libSQL file has no connection credentials to resolve.
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        dataset_url = str(dataset_database.graph_database_url or "")
        graph_db_name = dataset_database.graph_database_name

        # The engine cache key ensure_graph_memory_cleared's get_graph_engine()
        # resolves (built generically from graph_file_path/graph_database_name/
        # handler name by apply_database_context_variables) differs from the
        # narrower key an exact-match evict() here would target (graph_file_path=
        # "", no graph_database_name, default handler name) -- an exact-key evict
        # would miss the live cache entry and leave a stale engine with an open
        # connection to the file this method is about to remove. Evict by
        # database name instead, same pattern LadybugDatasetDatabaseHandler.
        # delete_dataset uses, and wait for any in-flight close (this adapter is
        # file-based, same reasoning as Ladybug's) before touching the file below.
        if graph_db_name:
            await graph_engine_cache.aevict_for_database(graph_db_name)

        # Remove the dataset's libSQL file and its WAL-mode companions. This
        # adapter runs PRAGMA journal_mode=WAL, so SQLite keeps write-ahead-log
        # state in "<file>-wal"/"<file>-shm" until a clean close checkpoints
        # them into the main file -- leaving them behind risks stale data
        # surviving under a same-name recreate.
        if dataset_url and os.path.isabs(dataset_url) and os.path.exists(dataset_url):
            os.remove(dataset_url)
            for suffix in ("-wal", "-shm"):
                companion_path = dataset_url + suffix
                if os.path.exists(companion_path):
                    os.remove(companion_path)
