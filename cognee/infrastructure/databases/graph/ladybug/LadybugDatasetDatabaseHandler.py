import os
from uuid import UUID
from typing import Optional

from cognee.infrastructure.databases.graph.get_graph_engine import (
    aevict_graph_engines_for_database,
)
from cognee.base_config import get_base_config
from cognee.modules.users.models import User
from cognee.modules.users.models import DatasetDatabase
from cognee.infrastructure.databases.dataset_database_handler import DatasetDatabaseHandlerInterface
from cognee.infrastructure.files.storage.get_file_storage import get_file_storage


class LadybugDatasetDatabaseHandler(DatasetDatabaseHandlerInterface):
    """
    Handler for interacting with Ladybug Dataset databases.
    """

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        """
        Create a new Ladybug instance for the dataset. Return connection info that will be mapped to the dataset.

        Args:
            dataset_id: Dataset UUID
            user: User object who owns the dataset and is making the request

        Returns:
            dict: Connection details for the created Ladybug instance

        """
        from cognee.infrastructure.databases.graph.config import get_graph_config

        graph_config = get_graph_config()

        if graph_config.graph_database_provider not in ("ladybug", "kuzu"):
            raise ValueError(
                "LadybugDatasetDatabaseHandler can only be used with Ladybug graph database provider."
            )

        graph_db_name = (
            f"{dataset_id}.pkl"
            if graph_config.graph_database_provider == "kuzu"
            else f"{dataset_id}.lbug"
        )
        graph_db_url = graph_config.graph_database_url
        graph_db_key = graph_config.graph_database_key

        return {
            "graph_database_name": graph_db_name,
            "graph_database_url": graph_db_url,
            "graph_database_provider": graph_config.graph_database_provider,
            "graph_database_key": graph_db_key,
            "graph_dataset_database_handler": graph_config.graph_dataset_database_handler,
            "graph_database_connection_info": {},
        }

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase):
        base_config = get_base_config()
        databases_directory_path = os.path.join(
            base_config.system_root_directory, "databases", str(dataset_database.owner_id)
        )
        graph_db_name = dataset_database.graph_database_name

        # Never open the database to drop it: opening spawns a fresh engine
        # (in subprocess mode, a worker that must take the on-disk file lock)
        # which races the just-torn-down one. Evict every cached engine for
        # this database — the same DB can sit under multiple cache keys —
        # wait for their in-flight closes to finish (a close deferred behind
        # an idle holder is not waited on; see aevict_graph_engines_for_database),
        # then remove the files directly. Server-backed handlers (e.g.
        # Postgres) are different on purpose: they drop the per-dataset
        # database over a connection, so no file handling applies there.
        await aevict_graph_engines_for_database(graph_db_name)

        file_storage = get_file_storage(databases_directory_path)
        if await file_storage.is_file(graph_db_name):
            await file_storage.remove(graph_db_name)
            # A clean close checkpoints and removes the WAL; the lock file and
            # a leftover WAL from a crashed worker must not survive the drop,
            # or a same-name recreate would replay stale data.
            for companion_file in (f"{graph_db_name}.lock", f"{graph_db_name}.wal"):
                if await file_storage.is_file(companion_file):
                    await file_storage.remove(companion_file)
        else:
            await file_storage.remove_all(graph_db_name)
