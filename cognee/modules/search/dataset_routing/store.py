"""The routing collection: one vector store for the whole deployment.

Dataset routing scores a query against every readable dataset before any
dataset is searched, so its rows cannot live inside a dataset's own vector
store. They live in the deployment's default vector store instead, the one
``VECTOR_DB_URL`` configures and every per-dataset database derives from,
which stands unused while backend access control binds each operation to its
dataset's store. The engine is built from the global config, never from the
per-dataset ContextVar, so an upsert from inside a dataset's pipeline run and
a read from search reach the same collection.

One embedding model for the whole deployment: the collection has one vector
width, and distances from two models cannot be ranked against each other. The
rows are not embedded here at all: the mirror copies the vectors the index
build stored in the dataset's own store (``sync_dataset_routes``), so each
summary is embedded once. The query is embedded by this engine with the
``.env`` embedding settings, never with a per-call ``embedding_config``, and
when the collection's width no longer matches those settings
(``EMBEDDING_MODEL`` changed without a prune) routing steps aside with a
warning and the search covers every readable dataset, as it did before
routing. Datasets built with different models are SDK-1316.

The engine is the default store's own cache entry, the one prune uses too.
It belongs to no dataset, so no dataset-queue slot would pin it against the
idle reaper, which force-closes unpinned subprocess engines and would close
it under a routing call; the queue's pin predicate therefore pins the default
store itself under access control (``DatasetQueuePinPredicate``).
"""

from __future__ import annotations

from contextvars import copy_context

from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.vector.get_vector_engine import _VectorEngineHandle
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger

DATASET_ROUTE_COLLECTION = "DatasetRoute_text"

logger = get_logger("dataset_routing")


class DatasetRoute(DataPoint):
    """One routing row: a global context index summary, tagged with its dataset.

    It keeps the id of the ``GlobalContextSummary`` it mirrors, so the read path
    derives every row id it may score from the dataset and document ids alone.
    """

    text: str
    dataset_id: str
    is_root: bool = False
    metadata: dict = {"index_fields": ["text"]}


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


async def routing_width_matches(engine) -> bool:
    """Whether the routing collection takes the configured model's vectors.

    The collection's vector width is fixed when it is created. After an
    ``EMBEDDING_MODEL`` change without a prune every write and query on it
    would fail inside the store, so the width is compared first and a mismatch
    turns routing off with a warning. An adapter that cannot report its stored
    width (Turso, community adapters) is not checked; an empty store matches,
    the first write creates the collection at the configured width.
    """
    read_stored_size = getattr(engine, "get_stored_vector_size", None)
    if read_stored_size is None:
        return True
    stored = await read_stored_size()
    configured = engine.embedding_engine.get_vector_size()
    if stored is None or stored == configured:
        return True
    logger.warning(
        "Dataset routing is off: the routing collection holds %d-dimensional vectors and "
        "the configured embedding model produces %d. Searches cover every readable "
        "dataset until the vector stores are pruned and rebuilt with one model.",
        stored,
        configured,
    )
    return False
