"""The routing collection: one vector store for the whole deployment.

Dataset routing scores a query against every readable dataset before any
dataset is searched, so its rows cannot live inside a dataset's own vector
store. They live in the deployment's default vector store instead, the one
``VECTOR_DB_URL`` configures and every per-dataset database derives from,
which stands unused while backend access control binds each operation to its
dataset's store. The engine is built from the global config, never from the
per-dataset ContextVar, so an upsert from inside a dataset's pipeline run and
a read from search reach the same collection.

Each exact model identifier and vector width has a separate collection. The
mirror copies the dataset's precomputed index vectors into its recorded
collection; queries use only the deployment model's collection. The legacy
unsuffixed collection is never scored, since its model provenance is unknown.
Switching models reuses existing collections without comparing models' scores.

The engine is the default store's own cache entry, the one prune uses too.
It belongs to no dataset, so no dataset-queue slot would pin it against the
idle reaper, which force-closes unpinned subprocess engines and would close
it under a routing call; the queue's pin predicate therefore pins the default
store itself under access control (``DatasetQueuePinPredicate``).
"""

from __future__ import annotations

from contextvars import copy_context
from hashlib import sha256
from uuid import UUID

from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.vector.embeddings.vector_size import engine_vector_size
from cognee.infrastructure.databases.vector.get_vector_engine import _VectorEngineHandle
from cognee.shared.logging_utils import get_logger

DATASET_ROUTE_COLLECTION = "DatasetRoute_text"

logger = get_logger("dataset_routing")


def dataset_route_collection(model: str, dimensions: int) -> str:
    """A portable collection identifier for an exact model identity and width."""
    return f"{DATASET_ROUTE_COLLECTION}__{sha256(model.encode()).hexdigest()[:24]}_{dimensions}"


async def routing_collections(engine) -> list[str]:
    """Discover historical routing tables, including Turso's lowercased identifiers."""
    legacy = DATASET_ROUTE_COLLECTION.lower()
    return [
        name
        for name in await engine.list_collections()
        if name.lower() == legacy or name.lower().startswith(legacy + "__")
    ]


def _deployment_embedding_record() -> dict:
    from cognee.context_global_variables import embedding_config, llm_config
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
        embedding_model_record,
    )

    embedding_config.set(None)
    llm_config.set(None)
    return embedding_model_record()


async def deployment_embedding_record() -> dict:
    """Resolve routing identity without entering a candidate dataset's store."""
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import DIMENSIONS_KEY

    record = copy_context().run(_deployment_embedding_record)
    if record[DIMENSIONS_KEY] is None:
        engine = await get_dataset_route_engine()
        record[DIMENSIONS_KEY] = await engine_vector_size(engine.embedding_engine)
    return record


async def dataset_embedding_records(dataset_ids: list[UUID]) -> dict[UUID, dict]:
    """Batch-read registry identity; failure must never permit incompatible search."""
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.users.models import DatasetDatabase

    async with get_relational_engine().get_async_session() as session:
        rows = await session.execute(
            select(
                DatasetDatabase.dataset_id, DatasetDatabase.vector_database_connection_info
            ).where(DatasetDatabase.dataset_id.in_(dataset_ids))
        )
        return {dataset_id: info or {} for dataset_id, info in rows}


async def eligible_dataset_ids(dataset_ids: list[UUID]) -> list[UUID]:
    """Keep deployment-compatible registry identities; unknown models await first use."""
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
        DIMENSIONS_KEY,
        MODEL_KEY,
    )

    identity = await deployment_embedding_record()
    records = await dataset_embedding_records(dataset_ids)
    eligible = []
    for dataset_id in dataset_ids:
        info = records.get(dataset_id, {})
        model, width = info.get(MODEL_KEY), info.get(DIMENSIONS_KEY)
        if (model is None and width in (None, identity[DIMENSIONS_KEY])) or (
            model == identity[MODEL_KEY] and width == identity[DIMENSIONS_KEY]
        ):
            eligible.append(dataset_id)
    return eligible


class _DatasetRouteEngineHandle(_VectorEngineHandle):
    """The default store's engine, embedding with the deployment's settings.

    ``_VectorEngineHandle`` builds the adapter's embedder from the embedding
    ContextVar at creation, and the engine cache keys on the store alone. An
    engine first touched from inside a dataset pipeline carrying a per-call
    ``embedding_config`` would then keep that endpoint and key for every later
    caller of the shared collection. This handle resolves the engine in a copy
    of the current context with the per-call LLM and embedding overrides
    cleared, so it always carries the ``.env`` model.
    """

    __slots__ = ()

    def _engine(self):
        pinned = self._pinned
        if pinned is not None and self._pin_is_live(pinned):
            return pinned
        return copy_context().run(self._resolve_with_deployment_settings)

    def _resolve_with_deployment_settings(self):
        # Imported at call time: the context module's imports lead back to the
        # data methods package this module is reached from.
        from cognee.context_global_variables import embedding_config, llm_config

        embedding_config.set(None)
        llm_config.set(None)
        return _VectorEngineHandle._engine(self)


async def get_dataset_route_engine() -> _VectorEngineHandle:
    """The vector engine of the deployment's default store, outside any dataset context.

    A re-resolving handle, like ``get_vector_engine_async`` returns: the routing
    writes hold it across several awaits while dataset pipelines open and evict
    engines from the shared cache, and a bare adapter reference would then be
    closed under them. The store itself is made to exist first
    (``ensure_default_store``): every routing read and write goes through here.
    """
    await ensure_default_store()
    return _DatasetRouteEngineHandle(get_vectordb_config().to_dict())


_default_store_ensured = False


async def ensure_default_store() -> None:
    """Create the default store's database on first use, where a handler would for a dataset.

    Under access control the pgvector handler creates one database per dataset
    on the vector host and never touches the default store, which belongs to
    no dataset. With no ``VECTOR_DB_NAME`` that store is named after the
    relational database, which exists on the vector host only when the
    relational database is Postgres on the same server. With SQLite as the
    relational database (a supported pairing: the handler needs Postgres for
    vectors alone) nothing creates it, and every routing write, a document
    delete first of all, would fail against a database that does not exist. So
    the routing store creates its database the way the handler creates a
    dataset's, with the same credentials and privileges; the collection
    installs the ``vector`` extension on its first write. Once per process:
    the check is atomic in Postgres, so concurrent first calls are only a few
    extra maintenance connections.
    """
    global _default_store_ensured
    if _default_store_ensured:
        return
    # Imported at call time: the context module's imports lead back to the
    # data methods package this module is reached from.
    from cognee.context_global_variables import backend_access_control_enabled

    config = get_vectordb_config()
    if config.vector_db_provider != "pgvector" or not backend_access_control_enabled():
        _default_store_ensured = True
        return

    from cognee.infrastructure.databases.postgres.admin import create_pg_database_if_not_exists
    from cognee.infrastructure.databases.relational import get_relational_config

    # The same name create_vector_engine gives the default store.
    db_name = config.vector_db_name or get_relational_config().db_name
    if await create_pg_database_if_not_exists(
        db_name,
        host=config.vector_db_host,
        port=config.vector_db_port,
        username=config.vector_db_username,
        password=config.vector_db_password,
    ):
        logger.info("Created the default vector database '%s' for dataset routing.", db_name)
    _default_store_ensured = True
