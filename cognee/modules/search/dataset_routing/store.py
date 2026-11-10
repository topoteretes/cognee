"""The routing collections: one vector store for the whole deployment, one collection per model.

Dataset routing scores a query against every readable dataset before any
dataset is searched, so its rows cannot live inside a dataset's own vector
store. They live in the deployment's default vector store instead, the one
``VECTOR_DB_URL`` configures and every per-dataset database derives from,
which stands unused while backend access control binds each operation to its
dataset's store. The engine is built from the global config, never from the
per-dataset ContextVar, so an upsert from inside a dataset's pipeline run and
a read from search reach the same collection.

One collection per embedding model and width (``dataset_route_collection_name``):
vectors from two models are not comparable and a collection's width is fixed
when it is created, so the rows of the model that built a dataset live apart
from the rest. The index build mirrors a dataset's rows into the collection of
the model that built it (``sync_dataset_routes``), the read path scores only
the configured model's collection (``route_datasets``), and switching
``EMBEDDING_MODEL`` moves to that model's collection -- the old one keeps its
rows for when the configuration comes back. A dataset whose recorded model is
not the configured one is never scored against it (``candidates.py``); within
one collection every row and every query carry the same model, which is the
one assumption ranking rests on. The rows are not embedded here at all: the
mirror copies the vectors the index build stored in the dataset's own store,
so each summary is embedded once. The query is embedded by this engine with
the ``.env`` embedding settings, never with a per-call ``embedding_config``.

The engine is the default store's own cache entry, the one prune uses too.
It belongs to no dataset, so no dataset-queue slot would pin it against the
idle reaper, which force-closes unpinned subprocess engines and would close
it under a routing call; the queue's pin predicate therefore pins the default
store itself under access control (``DatasetQueuePinPredicate``).
"""

from __future__ import annotations

import hashlib
import re
from contextvars import copy_context

from cognee.infrastructure.databases.vector import get_vectordb_config
from cognee.infrastructure.databases.vector.get_vector_engine import _VectorEngineHandle
from cognee.shared.logging_utils import get_logger

DATASET_ROUTE_COLLECTION = "DatasetRoute_text"
"""The pre-per-model collection name. Nothing writes or scores it anymore; the
cleanup paths drop it, so an upgraded deployment does not keep its old rows."""

_DATASET_ROUTE_COLLECTION_PREFIX = "DatasetRoute_text__"
# Postgres caps identifiers at 63 bytes: prefix (19) + separators (2) + a width
# of up to five digits (5) + digest (8) leaves this much of the model name.
_MODEL_NAME_LIMIT = 29

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


def route_model_record() -> dict:
    """The deployment's embedding model and width, per-call overrides cleared.

    Resolved exactly as ``embedding_model_record`` resolves them, but in a copy
    of the current context with the per-call LLM and embedding overrides
    removed -- the same resolution the routing engine itself uses. Naming the
    collection after the caller's override would point the read at a
    collection the writes never filled: the rows are mirrored under the
    deployment's model, so the query must be scored under it too.
    """
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
        embedding_model_record,
    )

    def _resolve_with_deployment_settings():
        from cognee.context_global_variables import embedding_config, llm_config

        embedding_config.set(None)
        llm_config.set(None)
        return embedding_model_record()

    return copy_context().run(_resolve_with_deployment_settings)


def dataset_route_collection_name(model: str | None, dimensions: int | None) -> str:
    """The routing collection of one embedding model and width.

    Deterministic and storage-safe: the model name is reduced to characters
    every adapter accepts (Postgres caps identifiers at 63 bytes, so the
    readable part is truncated) and a digest of the full pair keeps distinct
    models apart even when truncation or sanitization would collide. A model
    whose name the config does not record (a row from before models were
    recorded) shares the collection of no one: it reads ``unknown``.
    """
    readable = re.sub(r"[^A-Za-z0-9_-]", "_", model or "unknown")[:_MODEL_NAME_LIMIT]
    digest = hashlib.sha256(f"{model}|{dimensions}".encode()).hexdigest()[:8]
    return f"{_DATASET_ROUTE_COLLECTION_PREFIX}{readable}_{dimensions}_{digest}"


def dataset_route_collection() -> str:
    """The routing collection of the deployment's configured embedding model."""
    from cognee.infrastructure.databases.utils.ensure_embedding_model_matches import (
        DIMENSIONS_KEY,
        MODEL_KEY,
    )

    record = route_model_record()
    return dataset_route_collection_name(record[MODEL_KEY], record[DIMENSIONS_KEY])
