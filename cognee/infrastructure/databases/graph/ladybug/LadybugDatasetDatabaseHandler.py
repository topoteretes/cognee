import asyncio
import os
from uuid import UUID, NAMESPACE_OID, uuid5
from typing import Optional

from cognee.infrastructure.databases.cache.config import get_cache_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    graph_engine_cache,
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

        # In SHARED_LADYBUG_LOCK mode the Redis lock is the cross-process
        # mutex LadybugAdapter.query() takes before opening the native
        # ladybug.Database on this exact file — and LadybugAdapter.delete_graph()
        # already takes the same lock before removing files, for the same
        # reason. This handler evicts the cache and removes files directly
        # (see below), so it must hold that same lock too, or a concurrent
        # query()/search() in another process could be mid-query (holding the
        # on-disk file lock) when we delete its files, or could open the file
        # right between our drop and a later recreate. The lock key is
        # rebuilt from this dataset's own db path — identical to the path
        # get_graph_engine() resolves for this dataset (see
        # context_global_variables.apply_database_context_variables) — so it
        # is the exact same lock a concurrent query on this dataset would take.
        # Deployments that have NOT opted into SHARED_LADYBUG_LOCK get no
        # locking here (cache_config.shared_ladybug_lock is False and
        # redis_lock stays None) — that race is pre-existing for full dataset
        # deletion (datasets.delete_dataset() calls this same handler) and out
        # of scope for this fix.
        cache_config = get_cache_config()
        redis_lock = None
        held_redis_lock = None
        if cache_config.shared_ladybug_lock:
            from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine

            db_path = os.path.join(databases_directory_path, graph_db_name)
            redis_lock = get_cache_engine(
                lock_key="ladybug-lock-" + str(uuid5(NAMESPACE_OID, db_path))
            )
            # get_cache_engine() is typed Optional; shared_ladybug_lock=True means a
            # cache config was resolved, so a lock instance always comes back here
            # (same assumption LadybugAdapter.query()/delete_graph() make with this
            # exact assert before calling redis_lock.acquire_lock).
            assert redis_lock is not None
            held_redis_lock = await asyncio.to_thread(redis_lock.acquire_lock)

        try:
            # Never open the database to drop it: opening spawns a fresh engine
            # (in subprocess mode, a worker that must take the on-disk file lock)
            # which races the just-torn-down one. Evict every cached engine for
            # this database — the same DB can sit under multiple cache keys —
            # wait for their in-flight closes to finish (a close deferred behind
            # an idle holder is not waited on; see graph_engine_cache.aevict_for_database),
            # then remove the files directly. Server-backed handlers (e.g.
            # Postgres) are different on purpose: they drop the per-dataset
            # database over a connection, so no file handling applies there.
            await graph_engine_cache.aevict_for_database(graph_db_name)

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
        finally:
            if redis_lock is not None:
                await asyncio.to_thread(redis_lock.release_lock, held_redis_lock)
