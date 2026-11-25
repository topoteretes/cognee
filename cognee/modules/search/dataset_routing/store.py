"""The routing collection: one vector store for the whole deployment.

Dataset routing scores a query against every readable dataset before any
dataset is searched, so its rows cannot live inside a dataset's own vector
store. They live in the deployment's default vector store instead, the one
``VECTOR_DB_URL`` configures and every per-dataset database derives from,
which stands unused while backend access control binds each operation to its
dataset's store. The engine is built from the global config, never from the
per-dataset ContextVar, so an upsert from inside a dataset's pipeline run and
a read from search reach the same collection.

Routing summaries and queries use deployment embeddings, independently of
dataset-local embeddings and the cached storage adapter's embedder. Each
deployment model and width has its own collection; summaries are embedded
during indexing, never during query scoring. All readable datasets remain
eligible. Dataset retrieval still enforces its normal dimension guard.

The engine is the default store's own cache entry, the one prune uses too.
It belongs to no dataset, so no dataset-queue slot would pin it against the
idle reaper, which force-closes unpinned subprocess engines and would close
it under a routing call; the queue's pin predicate therefore pins the default
store itself under access control (``DatasetQueuePinPredicate``).
"""

from __future__ import annotations

import asyncio
import hashlib
from contextvars import copy_context
from dataclasses import dataclass
from types import SimpleNamespace

from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.vector.embeddings.config import (
    get_embedding_config,
    resolve_embedding_defaults,
)
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.vector.embeddings.EmbeddingGateway import EmbeddingGateway
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    get_embedding_engine,
)
from cognee.infrastructure.databases.vector.embeddings.vector_size import engine_vector_size
from cognee.infrastructure.databases.vector.get_vector_engine import _VectorEngineHandle
from cognee.infrastructure.llm.config import get_llm_config
from cognee.shared.logging_utils import get_logger

DATASET_ROUTE_COLLECTION = "DatasetRoute_text"

logger = get_logger("dataset_routing")


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
    handle = _DatasetRouteEngineHandle(get_vectordb_config().to_dict())
    await asyncio.to_thread(handle._engine)
    return handle


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


def dataset_route_collection(model: str, vector_size: int) -> str:
    """Storage-safe identity of an exact deployment model and width (under 63 bytes)."""
    if not model or type(vector_size) is not int or vector_size <= 0:
        raise ValueError("Dataset routing needs a model and a positive integer vector size")
    digest = hashlib.sha256(f"{model}\0{vector_size}".encode()).hexdigest()[:32]
    return f"{DATASET_ROUTE_COLLECTION}__{digest}"


@dataclass(frozen=True)
class DatasetRouteEmbedding:
    """Deployment embeddings resolved once for a routing read or write."""

    collection: str
    vector_size: int
    engine: EmbeddingEngine
    requests: asyncio.Semaphore


async def get_dataset_route_embedding() -> DatasetRouteEmbedding:
    """Resolve deployment settings without changing a cached storage adapter.

    Model initialization may download/load a local model, so it runs off-loop.
    Workers with an unknown width resolve it using their async protocol.
    """

    def resolve():
        from cognee.context_global_variables import embedding_config, llm_config
        from cognee.tasks.storage.index_data_points import indexing_budget

        embedding_config.set(None)
        llm_config.set(None)
        _, model, width = resolve_embedding_defaults(get_embedding_config(), get_llm_config())
        embedder = get_embedding_engine()
        budget = indexing_budget(SimpleNamespace(embedding_engine=embedder))
        return model, width, embedder, budget.requests

    model, width, embedder, requests = await asyncio.to_thread(copy_context().run, resolve)
    if width is None:
        width = await engine_vector_size(embedder)
    return DatasetRouteEmbedding(
        dataset_route_collection(model, width), width, EmbeddingGateway(embedder), requests
    )


async def dataset_route_collections(engine) -> list[str]:
    """Discover routing namespaces only, including unscored legacy collections.

    SQLite/Turso identifiers are case-insensitive. Discovery is store/schema
    scoped by the adapter; unsupported discovery fails mutations visibly.
    """
    legacy = DATASET_ROUTE_COLLECTION.lower()
    return [
        name
        for name in await engine.list_collections()
        if name.lower() == legacy
        or (
            name.lower().startswith(legacy + "__")
            and len(name) == len(legacy) + 2 + 32
            and all(char in "0123456789abcdef" for char in name.lower().split("__")[-1])
        )
    ]
