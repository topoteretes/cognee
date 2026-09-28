import asyncio
import copy
import inspect
import threading
import types
from collections import OrderedDict
from contextlib import asynccontextmanager
from enum import Enum
from os import path
from typing import (  # noqa: UP035 - typing.List is a distinct origin key, not an annotation
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

from filelock import AsyncFileLock
import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel

from cognee.infrastructure.background_tasks import register_background_task
from cognee.infrastructure.databases.exceptions import MissingQueryParameterError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.pgvector.serialize_data import serialize_data
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.files.storage import get_file_storage
from cognee.modules.observability import new_span
from cognee.modules.observability.tracing import (
    COGNEE_DB_SYSTEM,
    COGNEE_VECTOR_COLLECTION,
    COGNEE_VECTOR_RESULT_COUNT,
)
from cognee.modules.storage.utils import copy_model
from cognee.shared.logging_utils import get_logger
from cognee_db_workers.lancedb_compaction import (
    DEFAULT_MAX_TASKS_PER_RUN,
    DEFAULT_MAX_VERSIONS_PER_RUN,
    DEFAULT_RETENTION_SECONDS,
    DEFAULT_TARGET_ROWS_PER_FRAGMENT,
    PylanceIncompatibleError,
    compact_fragments,
    lance_core_mismatch,
    open_as_lance,
    prune_superseded_versions,
)

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.ScoredResult import ScoredResult
from ..stored_vector_size import choose_stored_vector_size
from ..vector_db_interface import VectorDBInterface

logger = get_logger("LanceDBAdapter")
_NO_DEFAULT = object()
# Anchors the background version prune each adapter starts when it first opens
# its store (asyncio keeps only weak references to tasks).
_OPEN_PRUNE_TASKS: set[asyncio.Task] = set()
_SIMPLE_TYPE_DEFAULTS = {
    str: "",
    int: 0,
    float: 0.0,
    bool: False,
    bytes: b"",
    UUID: UUID(int=0),
}
_ORIGIN_DEFAULT_FACTORIES = {
    list: list,
    List: list,  # noqa: UP006 - typing.List is a distinct origin key, not an annotation
    dict: dict,
    set: set,
    tuple: tuple,
}


class IndexSchema(DataPoint):
    """
    Represents a schema for an index data point containing an ID and text.

    Attributes:

    - id: A string representing the unique identifier for the data point.
    - text: A string representing the content of the data point.
    - metadata: A dictionary with default index fields for the schema, currently configured
    to include 'text'.
    """

    id: str
    text: str

    # Optional reference scalars carried for the search "Evidence" feature.
    # They stay None for non-chunk data points, so this schema remains
    # compatible with every indexed DataPoint type.
    document_id: str | None = None
    document_name: str | None = None
    chunk_index: int | None = None
    source_chunk_id: str | None = None
    importance_weight: float | None = 0.5
    # Document external_metadata as JSON text, copied onto chunks at ingest so
    # hybrid retrieval can surface allowlisted keys straight from the payload.
    external_metadata: str | None = None

    metadata: dict = {"index_fields": ["text"]}
    belongs_to_set: list[str] = []


class LanceDBAdapter(VectorDBInterface):
    """Vector-database adapter backed by LanceDB; implements the VectorDBInterface contract."""

    name = "LanceDB"
    # ``Optional`` because ``__init__`` accepts ``None`` for both — local
    # mode without an API key passes ``api_key=None``, and subprocess-mode
    # adapters constructed from cached state may also receive ``url=None``
    # (the ``RemoteLanceDBConnection`` carries the real URL).
    url: str | None
    api_key: str | None
    connection = None

    # Class-level memoization caches. They are shared across all adapter
    # instances because ``copy_model()`` and the LanceModel subclassing only
    # depend on the source DataPoint type + vector size — never on the
    # adapter instance itself.
    #
    # Pydantic attaches large per-class state (FieldInfo, SchemaSerializer,
    # SchemaValidator, ModelMetaclass, LazyClassAttribute) and caches it
    # globally by class identity. Without memoization, every call to
    # ``create_data_points`` mints a brand-new class for every data point,
    # and those classes are never collected — the tracemalloc profile on a
    # 2-cycle run showed +5550 FieldInfo and +879 ModelMetaclass per cycle.
    #
    # Bounded LRUs (cap 256). An unbounded dict would itself become a
    # memory-growth source if callers create many distinct DataPoint /
    # vector-size pairs over a long-running process — defeating the very
    # leak the cache is here to fix.
    _PAYLOAD_SCHEMA_CACHE_SIZE = 256
    _LANCE_DATAPOINT_CACHE_SIZE = 256
    _payload_schema_cache: "OrderedDict" = OrderedDict()
    _lance_datapoint_class_cache: "OrderedDict" = OrderedDict()
    _lance_cache_lock = threading.Lock()

    @classmethod
    def create_subprocess(
        cls,
        url: str | None,
        api_key: str | None,
        embedding_engine: "EmbeddingEngine",
    ) -> "LanceDBAdapter":
        """Create a LanceDBAdapter running in subprocess-proxy mode."""
        from .subprocess.proxy import (
            LanceDBSubprocessSession,
            RemoteLanceDBConnection,
        )

        session = LanceDBSubprocessSession.start()
        try:
            remote_conn = RemoteLanceDBConnection(session, url=url, api_key=api_key)
        except Exception:
            session.shutdown(timeout=2.0)
            raise

        return cls(
            url=url,
            api_key=api_key,
            embedding_engine=embedding_engine,
            connection=remote_conn,
            session=session,
        )

    def __init__(
        self,
        url: str | None,
        api_key: str | None,
        embedding_engine: EmbeddingEngine,
        *,
        connection: object | None = None,
        session: object | None = None,
    ):
        """
        In subprocess-proxy mode, ``connection`` is a ``RemoteLanceDBConnection``
        and ``session`` is a ``LanceDBSubprocessSession``. In local mode both
        are ``None`` and the adapter lazily creates a ``lancedb.AsyncConnection``
        on first use.

        Mixing the two — e.g. passing ``connection`` without ``session`` —
        creates an adapter no one owns: ``get_connection()`` would return a
        remote connection whose worker the adapter cannot shut down, leaving
        an orphaned subprocess on close. Reject up front.
        """
        if (connection is None) != (session is None):
            raise ValueError(
                "LanceDBAdapter requires both `connection` and `session` "
                "in subprocess mode, or neither in local mode."
            )
        # ``url`` is typed ``Optional[str]`` so callers can pass ``None``
        # in subprocess-proxy mode (the ``RemoteLanceDBConnection``
        # carries the real URL). In local mode the URL drives every
        # connection / file operation, so reject ``None`` up front
        # instead of crashing on the first ``connect_async`` /
        # ``prune`` call with a confusing ``AttributeError``.
        if connection is None and url is None:
            raise ValueError("LanceDBAdapter local mode requires a non-None `url`.")
        self.url = url
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.VECTOR_DB_LOCK = asyncio.Lock()
        # Guards lifecycle state — the ``connection``, ``_session`` and
        # ``_permanently_closed`` triple must be observed/mutated atomically
        # so a concurrent ``close()`` can't be silently overwritten by an
        # in-flight ``get_connection()`` resuming after its ``await``.
        # ``threading.Lock`` (not ``asyncio.Lock``) for cross-loop safety:
        # ``close()`` can be invoked from a foreign event loop via
        # ``closing_lru_cache._start_close`` running ``asyncio.run``, and
        # awaiting an asyncio.Lock there raises "got Future attached to a
        # different loop". The lock is held for microseconds at a time and
        # never wraps an ``await``, so it can't deadlock the event loop.
        self._lifecycle_lock = threading.Lock()
        self.connection = connection
        self._session = session
        # True iff this adapter was constructed in subprocess-proxy mode.
        # Latched at construction and NOT cleared by close()/clean(); once a
        # session is provided the adapter is forever bound to it. Combined
        # with ``_permanently_closed`` this gives a clean 3-state model:
        #   (False, False) — local mode, usable
        #   (True,  False) — subprocess mode, usable
        #   (*,     True)  — closed, not reusable in either mode
        self._subprocess_mode = session is not None
        self._permanently_closed = False
        # Compaction state (see ``compact``): tables written since their last
        # pass, a rotation cursor so a shared task budget is not always spent on
        # the same tables, a lock so concurrent passes do not overlap, and the
        # reason compaction was turned off for good, if it was.
        self._compaction_dirty: set[str] = set()
        self._compaction_cursor = 0
        self._compaction_lock = asyncio.Lock()
        self._compaction_unsupported: str | None = None
        self._lance_core_checked = False
        # The version prune started when this adapter first opens its store
        # (see ``_schedule_open_prune``), so a store that receives no more
        # writes still has its superseded versions reclaimed.
        self._open_prune_started = False
        self._open_prune_task: asyncio.Task | None = None
        # Timer for one more prune once a pass's pending versions have aged
        # (see ``_schedule_followup_prune``).
        self._followup_prune_handle: asyncio.TimerHandle | None = None

    @asynccontextmanager
    async def _write_lock(self):
        async with self.VECTOR_DB_LOCK:
            url = self.url or getattr(self.connection, "_url", None)
            if not url:
                raise RuntimeError("LanceDB URL required for interprocess writer lock")
            if url.startswith(("db://", "http://", "https://", "s3://", "gs://", "az://")):
                yield
            else:
                # ponytail: one local-store writer; partition by table if throughput requires it.
                async with AsyncFileLock(path.realpath(url) + ".writer.lock"):
                    yield

    async def get_connection(self):
        """
        Return the connection used by this adapter.

        The first successful call also starts a background prune of superseded
        table versions (see ``_schedule_open_prune``).
        """
        connection = await self._open_connection()
        self._schedule_open_prune()
        return connection

    async def _open_connection(self):
        """
        Return the connection used by this adapter.

        - Local mode: lazily constructs a ``lancedb.AsyncConnection``.
        - Subprocess mode: returns the injected ``RemoteLanceDBConnection``
          (ensuring its underlying subprocess ``lancedb`` connection is opened).

        A subprocess-mode adapter that has been closed is an error state —
        we refuse to silently fall through to a local lancedb connection.

        Race-safe under concurrent ``close()`` via ``_lifecycle_lock``: any
        connection created during an ``await`` is committed (or discarded)
        only after a re-check of the closed flag, so a closed adapter can
        never silently start handing out fresh connections again.
        """
        # Atomic state snapshot — no awaits inside the lock.
        with self._lifecycle_lock:
            if self._permanently_closed:
                raise RuntimeError(
                    "LanceDBAdapter is closed; a new adapter must be created "
                    "(subprocess-mode adapters cannot be re-initialized)."
                )
            existing = self.connection
            if existing is None and self._subprocess_mode:
                raise RuntimeError(
                    "LanceDBAdapter subprocess session is gone; adapter cannot "
                    "be re-initialized in local mode."
                )

        if existing is not None:
            # Remote connection lazily opens its own underlying lancedb handle.
            # If the first connect fails (bad URL, auth, network) the session
            # stays alive and never gets used — tear it down immediately so a
            # retry doesn't leak an orphan worker process for each failed
            # attempt. Mutate state under the lock so a concurrent ``close()``
            # observes a coherent (closed, no-session) view.
            ensure = getattr(existing, "_ensure_connected", None)
            if ensure is not None:
                try:
                    await ensure()
                except Exception:
                    with self._lifecycle_lock:
                        session = self._session
                        self._session = None
                        self.connection = None
                        self._permanently_closed = True
                    if session is not None:
                        # ``session.shutdown()`` is sync and can block for
                        # seconds (join → terminate → kill chain plus a
                        # bounded ``_rpc_lock`` acquire). Offload to a
                        # worker thread so we don't freeze the event loop
                        # during cleanup.
                        try:
                            await asyncio.to_thread(session.shutdown)
                        except Exception as teardown_err:
                            logger.warning(
                                "Error shutting down LanceDB subprocess after connect failure: %s",
                                teardown_err,
                                exc_info=True,
                            )
                    raise
            # Re-check the closed flag after the await — a concurrent
            # ``close()`` could have run while we were inside
            # ``_ensure_connected``, in which case we must not return
            # the (now-defunct) proxy.
            with self._lifecycle_lock:
                if self._permanently_closed:
                    raise RuntimeError(
                        "LanceDBAdapter is closed; a new adapter must be created "
                        "(subprocess-mode adapters cannot be re-initialized)."
                    )
            return existing

        # Local mode, lazy: create the connection outside the lock (the
        # await would otherwise block any concurrent observer), then
        # commit under the lock with a re-check. A concurrent ``close()``
        # that ran during our await must NOT be silently overwritten —
        # we discard the new connection and raise instead.
        new_conn = await lancedb.connect_async(self.url, api_key=self.api_key)
        stale = None
        # Capture the *winning* connection under the lock. Reading
        # ``self.connection`` after the lock release would race a
        # concurrent ``close()`` that nulls the field, and we'd return
        # ``None`` (or a connection that's about to be closed).
        winner = None
        with self._lifecycle_lock:
            if self._permanently_closed:
                stale = new_conn
                # winner stays None — we'll raise after closing the throwaway.
            elif self.connection is not None:
                # Lost the race — another caller already committed. Discard ours.
                stale = new_conn
                winner = self.connection
            else:
                self.connection = new_conn
                return new_conn

        # Discard the throwaway outside the lock — its close is async.
        try:
            await stale.close()
        except Exception:
            logger.debug("Ignoring exception in LanceDBAdapter.get_connection", exc_info=True)
        if winner is None:
            raise RuntimeError(
                "LanceDBAdapter is closed; a new adapter must be created "
                "(subprocess-mode adapters cannot be re-initialized)."
            )
        return winner

    # ------------------------------------------------------------------
    # Subprocess-mode conversion helpers. In local mode these are no-ops.
    # ------------------------------------------------------------------
    def _schema_for_create_table(self, lance_model_cls):
        """Return the schema value to pass to ``connection.create_table``.

        Local mode: a ``LanceModel`` class (lancedb accepts it directly).
        Subprocess mode: a ``pa.Schema`` derived from the LanceModel so the
        worker doesn't need to see pydantic.
        """
        if not self._subprocess_mode:
            return lance_model_cls
        return lance_model_cls.to_arrow_schema()

    def _records_for_write(self, records):
        """Convert LanceModel instances to a typed ``pa.Table`` in
        subprocess mode so the worker never needs to see pydantic. The
        Table carries both the data and the schema (derived from the
        LanceModel class via ``to_arrow_schema``) so LanceDB gets the
        exact types it expects.
        """
        if not records:
            return records
        if not self._subprocess_mode:
            return records

        import pyarrow as pa

        dicts = [r.model_dump() for r in records]
        schema = type(records[0]).to_arrow_schema()
        return pa.Table.from_pylist(dicts, schema=schema)

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """
        Embeds the provided textual data into vector representation.

        Uses the embedding engine to convert the list of strings into a list of float vectors.

        Parameters:
        -----------

            - data (list[str]): A list of strings representing the data to be embedded.

        Returns:
        --------

            - list[list[float]]: A list of embedded vectors corresponding to the input data.
        """
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        """
        Checks if the specified collection exists in the LanceDB.

        Returns True if the collection is present, otherwise False.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check.

        Returns:
        --------

            - bool: True if the collection exists, otherwise False.
        """
        connection = await self.get_connection()
        collection_names = await connection.table_names()
        return collection_name in collection_names

    async def get_stored_vector_size(self) -> int | None:
        """Width of the vectors already in this store, or None when nothing is stored yet.

        Every table's ``vector`` column is a fixed-size list of the embedding
        width that built it. Read once per dataset by the dataset context to
        record the width for rows that predate the recorded embedding model.

        Reads every table rather than stopping at the first: a store built
        across an ``EMBEDDING_MODEL`` change holds two widths, and which one
        gets recorded must not depend on listing order (see
        ``choose_stored_vector_size``).
        """
        connection = await self.get_connection()
        widths = []
        for table_name in await connection.table_names():
            schema = await (await connection.open_table(table_name)).schema()
            if "vector" not in schema.names:
                continue
            list_size = getattr(schema.field("vector").type, "list_size", None)
            if isinstance(list_size, int):
                widths.append(list_size)
        return choose_stored_vector_size(widths, self.name)

    async def create_collection(self, collection_name: str, payload_schema: BaseModel):
        """Create the LanceDB table for `collection_name` if it does not already exist."""
        vector_size = self.embedding_engine.get_vector_size()

        payload_schema = self.get_data_point_schema(payload_schema)
        LanceDataPoint = self._make_lance_datapoint_cls(payload_schema, vector_size)

        if not await self.has_collection(collection_name):
            async with self._write_lock():
                if not await self.has_collection(collection_name):
                    connection = await self.get_connection()
                    return await connection.create_table(
                        name=collection_name,
                        schema=self._schema_for_create_table(LanceDataPoint),
                        exist_ok=True,
                    )

    async def get_collection(self, collection_name: str):
        """Return the LanceDB table for `collection_name` or raise CollectionNotFoundError."""
        if not await self.has_collection(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found!")

        connection = await self.get_connection()
        return await connection.open_table(collection_name)

    async def create_data_points(self, collection_name: str, data_points: list[DataPoint]):
        """Upsert DataPoints into `collection_name`, merging belongs_to_set with any prior rows."""
        payload_schema = type(data_points[0])
        self._mark_written(collection_name)

        if not await self.has_collection(collection_name):
            await self.create_collection(collection_name, payload_schema)

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        vector_size = self.embedding_engine.get_vector_size()

        # One LanceDataPoint class per (payload schema, vector size), cached
        # globally. Building a new class per call — let alone per record —
        # leaks pydantic SchemaValidator/Serializer state that never gets gc'd.
        def _lance_cls_for(data_point):
            schema = self.get_data_point_schema(type(data_point))
            return self._make_lance_datapoint_cls(schema, vector_size), schema

        # The prefetch of existing `belongs_to_set` values and the subsequent
        # merge_insert must run inside the same lock section. If a second
        # upsert ran between our read and our write, merge_insert's
        # when_matched_update_all would overwrite tags we never saw and we'd
        # lose them silently. Holding VECTOR_DB_LOCK across read→build→write
        # serializes upserts but is the only way to keep tag unions honest.
        lance_data_points: list = []
        try:
            async with self._write_lock():
                collection = await self.get_collection(collection_name)
                existing_belongs_to_set: dict[str, list] = {}
                incoming_ids = [str(dp.id) for dp in data_points]
                if incoming_ids:
                    # Build the WHERE predicate explicitly with escaped string
                    # literals rather than relying on Python's tuple repr —
                    # mirrors how search() escapes `name` values to keep
                    # single-quotes from breaking the LanceDB SQL grammar.
                    escaped_ids = [id_.replace("'", "''") for id_ in incoming_ids]
                    if len(escaped_ids) == 1:
                        where_clause = f"id = '{escaped_ids[0]}'"
                    else:
                        id_list = ", ".join(f"'{id_}'" for id_ in escaped_ids)
                        where_clause = f"id IN ({id_list})"
                    try:
                        existing_rows = await collection.query().where(where_clause).to_list()
                        for row in existing_rows:
                            row_payload = row.get("payload") or {}
                            prior = row_payload.get("belongs_to_set") or []
                            if prior:
                                existing_belongs_to_set[row["id"]] = list(prior)
                    except Exception as e:
                        # Best-effort: if the lookup fails (e.g. empty table,
                        # schema mismatch the migration path will handle),
                        # fall through to the standard upsert.
                        logger.debug(
                            "belongs_to_set merge lookup failed for '%s': %s",
                            collection_name,
                            e,
                            exc_info=True,
                        )

                def create_lance_data_point(data_point: DataPoint, vector: list[float]):
                    lance_cls, payload_model = _lance_cls_for(data_point)
                    properties = payload_model.model_validate(
                        serialize_data(data_point.model_dump())
                    ).model_dump()

                    prior = existing_belongs_to_set.get(str(data_point.id))
                    if prior:
                        incoming = properties.get("belongs_to_set") or []
                        properties["belongs_to_set"] = list(
                            dict.fromkeys(list(prior) + list(incoming))
                        )

                    return lance_cls(
                        id=str(data_point.id),
                        vector=vector,
                        payload=properties,
                    )

                lance_data_points = [
                    create_lance_data_point(data_point, data_vectors[data_point_index])
                    for (data_point_index, data_point) in enumerate(data_points)
                ]

                # Dedup by id within the batch — on duplicates, union
                # `belongs_to_set` instead of keeping only the last
                # occurrence. A plain dict-collapse would drop tags that
                # only appeared on earlier siblings (mirrors the
                # batch-merge logic in PGVectorAdapter.create_data_points
                # and Neo4jAdapter.add_nodes).
                deduped_lance_points: dict = {}
                for dp in lance_data_points:
                    existing = deduped_lance_points.get(dp.id)
                    if existing is None:
                        deduped_lance_points[dp.id] = dp
                        continue
                    existing_payload = existing.payload.model_dump()
                    incoming_payload = dp.payload.model_dump()
                    existing_tags = existing_payload.get("belongs_to_set") or []
                    incoming_tags = incoming_payload.get("belongs_to_set") or []
                    if existing_tags or incoming_tags:
                        merged_tags = list(dict.fromkeys(list(existing_tags) + list(incoming_tags)))
                        incoming_payload["belongs_to_set"] = merged_tags
                        dp.payload = type(dp.payload).model_validate(incoming_payload)
                    deduped_lance_points[dp.id] = dp
                lance_data_points = list(deduped_lance_points.values())

                await (
                    collection.merge_insert("id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(self._records_for_write(lance_data_points))
                )
        except (ValueError, OSError, RuntimeError) as e:
            # Two LanceDB schema-drift failure modes are recoverable by rebuilding
            # the table via Pydantic validation (which fills defaults from the
            # current DataPoint subclass):
            #   1) "not found in target schema" — incoming payload has a field the
            #      old table schema does not know about.
            #   2) "contained null values" — old rows lack a field that the current
            #      schema now requires to be non-null. Raised from the Rust side
            #      (lance-file writer) when an old-schema row is upserted against
            #      a newer schema with a non-null field and no default in storage.
            err = str(e)
            if "not found in target schema" not in err and "contained null values" not in err:
                raise
            logger.warning(
                "Schema mismatch detected for collection '%s', migrating table: %s",
                collection_name,
                e,
            )
            await self._migrate_collection_schema(
                collection_name, collection, payload_schema, lance_data_points
            )

    async def upsert_raw_vectors(
        self,
        collection_name: str,
        points: list[dict],
        payload_schema: type[BaseModel] | None = None,
    ) -> None:
        """Upsert caller-provided vectors without invoking the embedding engine."""
        if not points:
            return
        if payload_schema is None:
            raise ValueError("payload_schema is required for LanceDB raw vector upserts")

        vector_size = self.embedding_engine.get_vector_size()
        LanceDataPoint = self._make_lance_datapoint_cls(payload_schema, vector_size)

        if not await self.has_collection(collection_name):
            async with self._write_lock():
                if not await self.has_collection(collection_name):
                    connection = await self.get_connection()
                    await connection.create_table(
                        name=collection_name,
                        schema=self._schema_for_create_table(LanceDataPoint),
                        exist_ok=True,
                    )

        raw_points = []
        for point in points:
            point_id = point.get("id")
            vector = point.get("vector")
            payload_value = point.get("payload")
            if point_id is None:
                raise ValueError("Raw vector point is missing id")
            if not isinstance(vector, list):
                raise ValueError("Raw vector point vector must be a list")
            if len(vector) != vector_size:
                raise ValueError(
                    f"Raw vector size {len(vector)} does not match expected size {vector_size}"
                )
            raw_points.append(
                LanceDataPoint(
                    id=str(point_id),
                    vector=vector,
                    payload=payload_schema.model_validate(payload_value).model_dump(),
                )
            )

        self._mark_written(collection_name)
        async with self._write_lock():
            collection = await self.get_collection(collection_name)
            await (
                collection.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(self._records_for_write(raw_points))
            )

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    _REMOTE_URL_PREFIXES = ("db://", "http://", "https://", "s3://", "gs://", "az://")

    def _store_url(self) -> str | None:
        """The LanceDB location: ``self.url`` in local mode, the remote
        connection's URL in subprocess mode (where ``url`` may be ``None``)."""
        return self.url or getattr(self.connection, "_url", None)

    def _is_remote_store(self, url: str | None) -> bool:
        return bool(url) and url.startswith(self._REMOTE_URL_PREFIXES)

    def _mark_written(self, collection_name: str) -> None:
        """Record a write so the next compaction pass serves this table first."""
        self._compaction_dirty.add(collection_name)

    def _compaction_options(self) -> dict:
        return {
            "target_rows_per_fragment": DEFAULT_TARGET_ROWS_PER_FRAGMENT,
            "retention_seconds": DEFAULT_RETENTION_SECONDS,
            "max_tasks": DEFAULT_MAX_TASKS_PER_RUN,
            "max_versions": DEFAULT_MAX_VERSIONS_PER_RUN,
        }

    def _compaction_order(self, names: list[str]) -> list[str]:
        """Tables written since their last pass first, then the rest.

        Each group is rotated by a cursor that advances every pass, so when the
        budget runs out before the end of the list, the next pass starts with
        tables this one did not reach. Called under ``_compaction_lock``.
        """
        cursor = self._compaction_cursor
        self._compaction_cursor += 1

        def rotated(group: list[str]) -> list[str]:
            if len(group) < 2:
                return group
            start = cursor % len(group)
            return group[start:] + group[:start]

        written = [name for name in names if name in self._compaction_dirty]
        untouched = [name for name in names if name not in self._compaction_dirty]
        return rotated(written) + rotated(untouched)

    def _compaction_skip_reason(self) -> str | None:
        """Why no compaction or version prune may run on this (enabled) store, or ``None``."""
        if self._is_remote_store(self._store_url()):
            return "remote_store"
        if not self._lance_core_checked:
            self._lance_core_checked = True
            mismatch = lance_core_mismatch()
            if mismatch:
                self._compaction_unsupported = "lance_core_mismatch"
                logger.warning(
                    "LanceDB compaction turned off: %s. Install the pylance release "
                    "line matching lancedb's bundled Lance core (see pyproject).",
                    mismatch,
                )
        return self._compaction_unsupported

    def _turn_off_compaction(self, exc: Exception) -> None:
        # Called from the handlers that caught ``exc``; its message names the
        # underlying error, which is all a version mismatch needs.
        self._compaction_unsupported = "pylance_incompatible"
        logger.warning(
            "LanceDB compaction turned off: pylance cannot open the tables lancedb "
            "wrote (%s). Install the pylance release line matching lancedb's "
            "bundled Lance core.",
            exc,
        )

    async def compact(self, collection_name: str | None = None) -> dict:
        """Fold the fragments cognee's upserts leave behind and prune old versions.

        Every ``merge_insert`` appends a new fragment and leaves the superseded
        rows on disk; LanceDB never reclaims either on its own, so a store in
        daily use grows without bound (issue #4684: 136 GB on disk for ~6 GB of
        live vectors). Called once at the end of every cognify run (see
        ``compact_vector_store``); a no-op on a compact table.

        Bounded, not whole-store: a pass executes at most
        ``DEFAULT_MAX_TASKS_PER_RUN`` compaction tasks and deletes
        at most ``DEFAULT_MAX_VERSIONS_PER_RUN`` old versions IN
        TOTAL across the store's tables (``0`` = no limit). A task rewrites one
        group of small fragments into one of at most
        ``target_rows_per_fragment`` rows; a version is deleted only once its
        successor has aged past the retention window, so a reader that opened
        the table a moment ago keeps working. A store with a large backlog
        therefore drains over several cognify runs instead of stalling one.
        Both budgets are spent greedily in table order: tables written since
        their last pass first, each group starting from a cursor that advances
        every pass, so no table starves. A pass ends once both are spent.

        The files a pass supersedes can only go once the retention window has
        passed, i.e. in a later pass. So that a store which receives no more
        writes still gets them back, every pass schedules one more prune for
        when the window has passed (``_schedule_followup_prune``), and every
        adapter prunes versions once when it first opens the store
        (``_schedule_open_prune``), for what a closed adapter left behind.

        Only the fragment rewrite runs under ``VECTOR_DB_LOCK``: its commit is a
        Lance "rewrite" transaction that conflicts with a concurrent upsert or
        delete on the same fragments. Version pruning commits nothing and runs
        outside the lock, so writers are never blocked behind it. Concurrent
        passes on one adapter do not queue: the second one returns at once.

        Skipped for remote stores (``s3://`` and friends), where every rewrite
        is network transfer and cleanup is thousands of object deletes, and
        turned off for the adapter's lifetime when the installed pylance is not
        on the Lance core lancedb bundles, or cannot open the tables lancedb
        wrote. Fail-open per collection: a failure is logged and reported in
        the returned stats, never raised.
        """
        options = self._compaction_options()
        reason = self._compaction_skip_reason()
        if reason:
            return {"skipped": reason}
        # This pass prunes versions itself, so a first-open prune not yet
        # started is redundant; one already running finishes first.
        self._open_prune_started = True
        await self._wait_for_open_prune()
        if self._compaction_lock.locked():
            return {"skipped": "in_progress"}

        async with self._compaction_lock:
            connection = await self.get_connection()
            if collection_name is not None:
                names = [collection_name]
            else:
                names = self._compaction_order(list(await connection.table_names()))

            task_budget = int(options["max_tasks"])  # 0 = unlimited
            version_budget = int(options["max_versions"])  # 0 = unlimited
            tasks_left, versions_left = task_budget, version_budget
            results: dict = {}
            for name in names:
                tasks_spent = task_budget > 0 and tasks_left <= 0
                versions_spent = version_budget > 0 and versions_left <= 0
                if tasks_spent and versions_spent:
                    break
                # The helpers read 0 as "no limit" and a negative value as "none".
                max_tasks = -1 if tasks_spent else (tasks_left if task_budget > 0 else 0)
                max_versions = (
                    -1 if versions_spent else (versions_left if version_budget > 0 else 0)
                )
                if not tasks_spent:
                    # Cleared before the pass, so a write that lands during it
                    # marks the table again for the next one. A table whose
                    # rewrite the budget skipped stays first in line.
                    self._compaction_dirty.discard(name)
                try:
                    stats = await self._compact_collection(name, options, max_tasks, max_versions)
                except PylanceIncompatibleError as exc:
                    self._turn_off_compaction(exc)
                    results[name] = {"error": str(exc)[:200]}
                    break
                except Exception as exc:
                    logger.warning(
                        "Compaction skipped for collection '%s': %s",
                        name,
                        exc,
                        exc_info=True,
                    )
                    results[name] = {"error": str(exc)[:200]}
                    continue
                results[name] = stats
                if task_budget > 0:
                    tasks_left -= int(stats.get("executed_tasks", 0) or 0)
                if version_budget > 0:
                    versions_left -= int(stats.get("old_versions_removed", 0) or 0)
            self._schedule_followup_prune(options)
            return results

    async def _compact_collection(
        self, collection_name: str, options: dict, max_tasks: int, max_versions: int
    ) -> dict:
        collection = await self.get_collection(collection_name)
        target_rows = options["target_rows_per_fragment"]

        async with self.VECTOR_DB_LOCK:
            if self._subprocess_mode:
                stats = await collection.compact_fragments(
                    target_rows_per_fragment=target_rows, max_tasks=max_tasks
                )
            else:
                # Plan against the version current under the lock: a write that
                # committed after the handle was opened would otherwise make
                # the rewrite conflict and fail. Hand the table to pylance, do
                # the I/O off the event loop, then move the handle to the
                # version the compaction committed.
                await collection.checkout_latest()
                dataset = await open_as_lance(collection)
                stats = await asyncio.to_thread(
                    compact_fragments,
                    dataset,
                    target_rows_per_fragment=target_rows,
                    max_tasks=max_tasks,
                )
                await collection.checkout_latest()

        pruned = await self._prune_collection(
            collection, options["retention_seconds"], max_versions
        )
        return {**(stats or {}), **(pruned or {})}

    async def _prune_collection(self, collection, retention_seconds: int, max_versions: int):
        if self._subprocess_mode:
            return await collection.prune_versions(
                retention_seconds=retention_seconds, max_versions=max_versions
            )
        dataset = await open_as_lance(collection)
        return await asyncio.to_thread(
            prune_superseded_versions,
            dataset,
            retention_seconds=retention_seconds,
            max_versions=max_versions,
        )

    # ------------------------------------------------------------------
    # Version prune on first open
    # ------------------------------------------------------------------

    def _schedule_open_prune(self) -> None:
        """Start, once per adapter, a background prune of superseded versions.

        A compaction pass can only delete what it superseded once the retention
        window has passed, i.e. in a later pass, and passes run only at the end
        of a cognify. A store that stops receiving writes would keep those
        files. Every process that opens the store -- server start, an SDK
        script, an engine re-created after the idle-engine eviction -- reclaims
        them here instead, within the same ``max_versions_per_run`` budget.
        Runs in the background so the operation that opened the store is not
        delayed; never raises.
        """
        if self._open_prune_started:
            return
        self._open_prune_started = True
        options = self._compaction_options()
        if self._compaction_skip_reason():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._start_prune_task(options)

    def _schedule_followup_prune(self, options: dict) -> None:
        """Prune once more when the versions younger than the retention window at
        this pass (the writes that preceded it, the files it superseded) have aged.

        Otherwise they wait for the next cognify, which an adapter that stays
        open (a server) may never see. A later pass replaces the timer; the
        prune itself schedules nothing, so the timer never chains."""
        if self._followup_prune_handle is not None:
            self._followup_prune_handle.cancel()
            self._followup_prune_handle = None
        if options["retention_seconds"] > 0:
            self._followup_prune_handle = asyncio.get_running_loop().call_later(
                options["retention_seconds"] + 1, self._start_followup_prune, options
            )

    def _start_followup_prune(self, options: dict) -> None:
        self._followup_prune_handle = None
        if self._permanently_closed:
            return
        self._start_prune_task(options)

    def _start_prune_task(self, options: dict) -> asyncio.Task:
        """Start a version prune in the background.

        Every prune -- first-open or follow-up -- goes through here and takes
        the single ``_open_prune_task`` slot that ``compact``, ``close`` and
        ``prune`` wait on. A prune that replaces a still-running one in the slot
        needs no explicit chaining: both take ``_compaction_lock``, which wakes
        waiters in order, so the newer one finishes last and waiting on it
        covers both. Registered as a background task, so
        ``cognee.wait_for_background_tasks()`` and the server's shutdown drain
        let it finish.
        """
        task = asyncio.ensure_future(self._prune_on_open(options))
        self._open_prune_task = task
        _OPEN_PRUNE_TASKS.add(task)
        task.add_done_callback(_OPEN_PRUNE_TASKS.discard)
        register_background_task(task)
        return task

    async def _prune_on_open(self, options: dict) -> dict:
        try:
            async with self._compaction_lock:
                return await self._prune_pass(options)
        except Exception as exc:
            logger.warning("Version prune on open skipped: %s", exc, exc_info=True)
            return {"error": str(exc)[:200]}

    async def _wait_for_open_prune(self) -> None:
        """Let a running version prune (first-open or follow-up, see
        ``_start_prune_task``) finish -- it is bounded -- before this adapter
        compacts, prunes or closes, so they never overlap and closing never tears
        the worker down under it. A prune started on another event loop cannot be
        awaited from this one and is left alone."""
        task = self._open_prune_task
        if task is None or task.done():
            return
        try:
            same_loop = task.get_loop() is asyncio.get_running_loop()
        except RuntimeError:
            return
        if same_loop:
            await asyncio.wait({task})

    async def _prune_pass(self, options: dict) -> dict:
        """Prune superseded versions across the store's tables; no fragment rewrites."""
        connection = await self.get_connection()
        budget = int(options["max_versions"])  # 0 = unlimited
        left = budget
        results: dict = {}
        for name in list(await connection.table_names()):
            if budget > 0 and left <= 0:
                break
            try:
                collection = await self.get_collection(name)
                stats = await self._prune_collection(
                    collection, options["retention_seconds"], left if budget > 0 else 0
                )
            except PylanceIncompatibleError as exc:
                self._turn_off_compaction(exc)
                results[name] = {"error": str(exc)[:200]}
                break
            except Exception as exc:
                logger.warning(
                    "Version prune skipped for collection '%s': %s", name, exc, exc_info=True
                )
                results[name] = {"error": str(exc)[:200]}
                continue
            results[name] = stats
            if budget > 0:
                left -= int(stats.get("old_versions_removed", 0) or 0)
        removed = sum(
            int(value.get("old_versions_removed", 0) or 0)
            for value in results.values()
            if isinstance(value, dict)
        )
        if removed:
            logger.info("LanceDB version prune on open: %d old version(s) removed", removed)
        return results

    async def _migrate_collection_schema(
        self,
        collection_name: str,
        old_collection,
        payload_schema: type,
        new_lance_data_points: list,
    ):
        """Migrate a LanceDB table to a new schema, preserving existing data."""
        rows = (await old_collection.to_arrow()).to_pylist()

        vector_size = self.embedding_engine.get_vector_size()
        schema_model = self.get_data_point_schema(payload_schema)
        valid_payload_fields = set(schema_model.model_fields.keys())
        defaults = self._get_payload_defaults(payload_schema)

        # Reuse the cached LanceDataPoint class rather than mint a new
        # ``MigrationLanceDataPoint`` class per migration call.
        MigrationLanceDataPoint = self._make_lance_datapoint_cls(schema_model, vector_size)

        new_ids = {dp.id for dp in new_lance_data_points}
        typed_old_rows = []
        skipped = 0
        failed_rows = []
        for row in rows:
            if row.get("id") in new_ids:
                continue

            raw_payload = row.get("payload")
            if raw_payload is None or not isinstance(raw_payload, dict):
                raw_payload = dict(defaults)

            # Strip to only fields in the new schema and fill defaults
            raw_payload = {k: v for k, v in raw_payload.items() if k in valid_payload_fields}
            for key, val in defaults.items():
                raw_payload.setdefault(key, val)

            # Convert to typed LanceModel instances to ensure exact Arrow
            # type compatibility. Using collection.add(dicts) causes LanceDB
            # to infer Arrow types from Python values, which can differ from
            # the schema's declared types and cause Rust panics on subsequent
            # vector searches.
            try:
                validated_payload = schema_model.model_validate(raw_payload).model_dump()
                typed_old_rows.append(
                    MigrationLanceDataPoint(
                        id=row["id"],
                        vector=row["vector"],
                        payload=validated_payload,
                    )
                )
            except Exception as e:
                row_id = str(row.get("id", "?"))
                logger.warning(
                    "Skipping row %s during migration (validation failed): %s",
                    row_id,
                    e,
                    exc_info=True,
                )
                skipped += 1
                failed_rows.append((row_id, str(e)))

        if skipped:
            example_failures = "; ".join(
                [f"{row_id}: {error}" for row_id, error in failed_rows[:5]]
            )
            logger.error(
                "Migration of '%s' aborted: %d rows cannot be migrated to the new schema. "
                "No data was modified. Add explicit defaults for newly required fields or make "
                "them optional. Example validation failures: %s",
                collection_name,
                skipped,
                example_failures or "<no details>",
            )
            raise RuntimeError(
                f"LanceDB migration aborted for '{collection_name}': {skipped} existing rows "
                "cannot be backfilled to the new payload schema. "
                "Add an explicit default value for newly required fields "
                "(or make them optional) and re-run migration."
            )

        async with self._write_lock():
            connection = await self.get_connection()
            await connection.drop_table(collection_name)
            await connection.create_table(
                name=collection_name,
                schema=self._schema_for_create_table(MigrationLanceDataPoint),
            )
            collection = await connection.open_table(collection_name)

            if typed_old_rows:
                await collection.add(self._records_for_write(typed_old_rows))

            if new_lance_data_points:
                await (
                    collection.merge_insert("id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(self._records_for_write(new_lance_data_points))
                )

        logger.info(
            "Migrated collection '%s' schema (%d existing rows preserved)",
            collection_name,
            len(typed_old_rows),
        )

    @classmethod
    def _resolve_collection_payload_schema(cls, collection_name: str):
        # Vector index collections store IndexSchema payloads regardless of the
        # source DataPoint type encoded in the collection name.
        return IndexSchema

    @staticmethod
    def _normalize_arrow_type(arrow_type):
        """
        Convert Arrow type objects into a recursive comparable structure.
        This allows deep comparison of field names, nullability, and nested types.
        """
        if hasattr(arrow_type, "num_fields"):
            return {
                "kind": type(arrow_type).__name__,
                "fields": [
                    {
                        "name": arrow_type.field(i).name,
                        "nullable": getattr(arrow_type.field(i), "nullable", True),
                        "type": LanceDBAdapter._normalize_arrow_type(arrow_type.field(i).type),
                    }
                    for i in range(arrow_type.num_fields)
                ],
            }

        value_field = getattr(arrow_type, "value_field", None)
        if value_field is not None:
            return {
                "kind": type(arrow_type).__name__,
                "value_nullable": getattr(value_field, "nullable", True),
                "value_type": LanceDBAdapter._normalize_arrow_type(value_field.type),
            }

        key_type = getattr(arrow_type, "key_type", None)
        item_type = getattr(arrow_type, "item_type", None)
        if key_type is not None and item_type is not None:
            return {
                "kind": type(arrow_type).__name__,
                "key_type": LanceDBAdapter._normalize_arrow_type(key_type),
                "item_type": LanceDBAdapter._normalize_arrow_type(item_type),
            }

        return {"kind": type(arrow_type).__name__, "repr": str(arrow_type)}

    def _get_target_payload_arrow_type(self, payload_schema: type):
        """
        Build a probe LanceModel and extract the expected Arrow type of the payload field.
        """
        vector_size = self.embedding_engine.get_vector_size()
        schema_model = self.get_data_point_schema(payload_schema)
        data_point_types = get_type_hints(schema_model)

        class SchemaProbeDataPoint(LanceModel):
            id: data_point_types["id"]
            vector: Vector(vector_size)
            payload: schema_model

        to_arrow_schema = getattr(SchemaProbeDataPoint, "to_arrow_schema", None)
        if not callable(to_arrow_schema):
            return None

        try:
            target_schema = to_arrow_schema()
        except TypeError:
            # Models with complex Union types (e.g. List[Union[Entity, Event,
            # tuple[Edge, Entity]]]) can't be converted to Arrow. Fall back to
            # field-name comparison in _is_payload_schema_compatible.
            return None
        payload_field_index = target_schema.get_field_index("payload")
        if payload_field_index < 0:
            return None

        return target_schema.field(payload_field_index).type

    def _is_payload_schema_compatible(self, existing_payload_type, payload_schema: type) -> bool:
        """
        Check compatibility via deep Arrow type comparison.
        Falls back to field-name comparison if target Arrow type can't be derived.
        """
        target_payload_type = self._get_target_payload_arrow_type(payload_schema)
        if target_payload_type is None:
            if not hasattr(existing_payload_type, "num_fields"):
                return False

            existing_payload_fields = {
                existing_payload_type.field(i).name for i in range(existing_payload_type.num_fields)
            }
            target_schema_model = self.get_data_point_schema(payload_schema)
            target_payload_fields = set(target_schema_model.model_fields.keys())
            return existing_payload_fields == target_payload_fields

        normalized_existing = self._normalize_arrow_type(existing_payload_type)
        normalized_target = self._normalize_arrow_type(target_payload_type)
        return normalized_existing == normalized_target

    async def run_migrations(self):
        """
        Proactively migrates all LanceDB collections that map to known DataPoint schemas.
        This is intended for startup/readiness checks so searches don't hit legacy schemas.
        """
        connection = await self.get_connection()
        collection_names = await connection.table_names()

        migrated_collections = []
        checked_collections = []
        skipped_collections = []

        for collection_name in collection_names:
            payload_schema = self._resolve_collection_payload_schema(collection_name)
            if payload_schema is None:
                skipped_collections.append(collection_name)
                continue

            checked_collections.append(collection_name)
            collection = await self.get_collection(collection_name)
            table = await collection.to_arrow()
            payload_field_index = table.schema.get_field_index("payload")

            if payload_field_index < 0:
                skipped_collections.append(collection_name)
                continue

            payload_field_type = table.schema.field(payload_field_index).type
            if not hasattr(payload_field_type, "num_fields"):
                skipped_collections.append(collection_name)
                continue

            if self._is_payload_schema_compatible(payload_field_type, payload_schema):
                continue

            logger.info(
                "Proactive LanceDB migration for '%s' due to payload schema mismatch",
                collection_name,
            )
            try:
                await self._migrate_collection_schema(
                    collection_name=collection_name,
                    old_collection=collection,
                    payload_schema=payload_schema,
                    new_lance_data_points=[],
                )
                migrated_collections.append(collection_name)
            except TypeError as e:
                # Models with fields that LanceDB can't convert to Arrow
                # (e.g. List[tuple], complex Unions) can't be migrated
                # proactively. The reactive migration in create_data_points
                # will handle them on the next write.
                logger.warning(
                    "Skipping proactive migration for '%s' (unsupported type): %s",
                    collection_name,
                    e,
                )
                skipped_collections.append(collection_name)

        return {
            "checked_collections": checked_collections,
            "migrated_collections": migrated_collections,
            "skipped_collections": skipped_collections,
        }

    def _get_payload_defaults(self, payload_schema: type) -> dict:
        """Extract default values from payload model, including inferred defaults for required fields."""
        schema_model = self.get_data_point_schema(payload_schema)
        defaults = {}
        for name, field_info in schema_model.model_fields.items():
            is_required = hasattr(field_info, "is_required") and field_info.is_required()
            if not is_required:
                default_value = field_info.get_default(call_default_factory=True)
                defaults[name] = copy.deepcopy(default_value)
                continue

            inferred_default = self._infer_default_for_annotation(field_info.annotation)
            if inferred_default is not _NO_DEFAULT:
                defaults[name] = inferred_default
        return defaults

    @staticmethod
    def _infer_default_for_annotation(annotation):
        """Infer a safe fallback default for required fields without explicit defaults."""
        origin = get_origin(annotation)
        args = get_args(annotation)

        if annotation in _SIMPLE_TYPE_DEFAULTS:
            return _SIMPLE_TYPE_DEFAULTS[annotation]

        if origin in _ORIGIN_DEFAULT_FACTORIES:
            return _ORIGIN_DEFAULT_FACTORIES[origin]()

        if str(origin).endswith("Literal"):
            return args[0] if args else _NO_DEFAULT

        if origin in (Union, types.UnionType):
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) != len(args):
                return None
            for arg in non_none_args:
                inferred = LanceDBAdapter._infer_default_for_annotation(arg)
                if inferred is not _NO_DEFAULT:
                    return inferred
            return _NO_DEFAULT

        if isinstance(annotation, type):
            if issubclass(annotation, Enum):
                members = list(annotation)
                return members[0] if members else _NO_DEFAULT
            if hasattr(annotation, "model_fields"):
                return {}

        return _NO_DEFAULT

    def _coerce_rows_to_typed_payload(self, rows: list, payload_schema: type | None) -> list:
        """Validate raw LanceDB rows through the collection's declared
        payload model so `collection.add()` writes values whose Arrow types
        match the stored schema. Without this, LanceDB infers Arrow types
        from Python values on add, and the inferred types can drift from
        the stored schema — the same class of problem _migrate_collection_schema
        guards against. Falls back to the original dicts if the schema
        can't be resolved or validation fails.
        """
        if not rows or payload_schema is None:
            return rows

        schema_model = self.get_data_point_schema(payload_schema)
        coerced: list = []
        for row in rows:
            raw_payload = row.get("payload") or {}
            if not isinstance(raw_payload, dict):
                coerced.append(row)
                continue
            try:
                validated = schema_model.model_validate(raw_payload).model_dump()
                # Re-typing must NOT introduce columns the stored row lacks. The
                # schema model may have grown optional fields since this
                # collection was created (e.g. reference scalars added to a text
                # IndexSchema); model_dump() would emit them as None, and adding
                # a field absent from the table's Arrow struct makes
                # `collection.add` reject the row ("field '...' does not exist in
                # table schema"). Keep exactly the stored key set, coercing the
                # value where the model provided one.
                validated = {key: validated.get(key, raw_payload[key]) for key in raw_payload}
            except Exception as e:
                logger.debug(
                    "_coerce_rows_to_typed_payload: validation fell back for id=%s: %s",
                    row.get("id"),
                    e,
                    exc_info=True,
                )
                coerced.append(row)
                continue
            new_row = dict(row)
            new_row["payload"] = validated
            coerced.append(new_row)
        return coerced

    async def retrieve(
        self, collection_name: str, data_point_ids: list[str], *, include_vector: bool = False
    ):
        """Return rows from `collection_name` whose id is in `data_point_ids`.

        ``include_vector`` is a LanceDB-only extension (the shared interface is
        unchanged): when True, the stored embedding is attached to each result
        under ``payload["vector"]``. The semantic memory map is the consumer;
        it needs the raw vectors the plain retrieve path drops.
        """
        if not data_point_ids:
            # No ids requested. Avoid building an "id IN ()" filter, which lance
            # rejects as a parse error; pgvector/chromadb return [] here too.
            return []
        try:
            collection = await self.get_collection(collection_name)
        except CollectionNotFoundError:
            # If collection doesn't exist, return empty list (no items to retrieve)
            return []

        # ids may be UUIDs or graph-computed deterministic strings that can
        # contain single quotes, so build the predicate with SQL escaping
        # (mirrors delete_data_points). Python's tuple repr double-quotes such
        # strings and LanceDB >= 0.38 parses double quotes as column names.
        escaped_ids = [str(id_).replace("'", "''") for id_ in data_point_ids]
        if len(escaped_ids) == 1:
            where_clause = f"id = '{escaped_ids[0]}'"
        else:
            id_list = ", ".join(f"'{id_}'" for id_ in escaped_ids)
            where_clause = f"id IN ({id_list})"
        query = collection.query().where(where_clause)

        # Convert query results to list format
        results_list = await query.to_list()

        return [
            ScoredResult(
                id=parse_id(result["id"]),
                # The copy keeps the stored row untouched; the default path is
                # byte-for-byte the previous behavior.
                payload={**result["payload"], "vector": result["vector"]}
                if include_vector
                else result["payload"],
                score=0,
            )
            for result in results_list
        ]

    SCORE_ID_BATCH_SIZE = 1000

    async def score_by_ids(
        self, collection_name: str, data_point_ids: list[str], query_vector: list[float]
    ) -> list[ScoredResult]:
        ids = list(dict.fromkeys(str(point_id) for point_id in data_point_ids))
        if not ids:
            return []
        collection = await self.get_collection(collection_name)
        scores = []
        for start in range(0, len(ids), self.SCORE_ID_BATCH_SIZE):
            batch = ids[start : start + self.SCORE_ID_BATCH_SIZE]
            literals = ", ".join("'" + point_id.replace("'", "''") + "'" for point_id in batch)
            # where() prefilters by default. Bypass ANN so even an indexed table
            # scores every requested row, including neighbors far from the query.
            rows = await (
                collection.vector_search(query_vector)
                .distance_type("cosine")
                .where(f"id IN ({literals})")
                .bypass_vector_index()
                .select(["id", "_distance"])
                .limit(len(batch))
                .to_list()
            )
            scores.extend(
                ScoredResult(id=parse_id(row["id"]), score=float(row["_distance"]), payload=None)
                for row in rows
            )
        return scores

    async def search(
        self,
        collection_name: str,
        query_text: str | None = None,
        query_vector: list[float] | None = None,
        limit: int | None = 15,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: list[str] | None = None,
        node_name_filter_operator: str = "OR",
    ):
        with new_span("cognee.db.vector.search") as otel_span:
            otel_span.set_attribute(COGNEE_DB_SYSTEM, "lancedb")
            otel_span.set_attribute(COGNEE_VECTOR_COLLECTION, collection_name)

            if query_text is None and query_vector is None:
                raise MissingQueryParameterError()

            if query_text and not query_vector:
                query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

            collection = await self.get_collection(collection_name)

            if limit is None:
                limit = await collection.count_rows()

            # LanceDB search will break if limit is 0 so we must return
            if limit <= 0:
                otel_span.set_attribute(COGNEE_VECTOR_RESULT_COUNT, 0)
                return []

            # Note: Exclude payload if not needed to optimize performance
            select_columns = (
                ["id", "vector", "payload", "_distance"]
                if include_payload
                else ["id", "vector", "_distance"]
            )

            if node_name:
                # Escape quotes to make this input safer, since it's coming from the user
                # At the time of writing this, no specific binding instructions found on LanceDB docs
                escaped_node_names = [name.replace("'", "''") for name in node_name]
                literal_node_names = (
                    "[" + ", ".join(f"'{name}'" for name in escaped_node_names) + "]"
                )

                if node_name_filter_operator == "AND":
                    node_name_filter_string = (
                        f"array_has_all(payload.belongs_to_set, {literal_node_names})"
                    )
                else:
                    node_name_filter_string = (
                        f"array_has_any(payload.belongs_to_set, {literal_node_names})"
                    )

                result_values = (
                    await collection.vector_search(query_vector)
                    .distance_type("cosine")
                    .where(node_name_filter_string)
                    .select(select_columns)
                    .limit(limit)
                    .to_list()
                )
            else:
                result_values = (
                    await collection.vector_search(query_vector)
                    .distance_type("cosine")
                    .select(select_columns)
                    .limit(limit)
                    .to_list()
                )

            if not result_values:
                otel_span.set_attribute(COGNEE_VECTOR_RESULT_COUNT, 0)
                return []

            results = [
                ScoredResult(
                    id=parse_id(result["id"]),
                    payload=result["payload"] if include_payload else None,
                    score=float(result["_distance"]),
                )
                for result in result_values
            ]

            otel_span.set_attribute(COGNEE_VECTOR_RESULT_COUNT, len(results))

            return results

    async def batch_search(
        self,
        collection_name: str,
        query_texts: list[str],
        limit: int | None = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: list[str] | None = None,
    ):
        query_vectors = await self.embedding_engine.embed_text(query_texts)

        return await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_vector=with_vectors,
                    include_payload=include_payload,
                    node_name=node_name,
                )
                for query_vector in query_vectors
            ]
        )

    supports_payload_update = True

    async def update_payload(self, collection_name: str, payload_updates: dict[str, dict]) -> None:
        """Update payload fields on existing rows WITHOUT re-embedding.

        Vector-preserving rewrite: read the matching rows (payload struct and
        stored vector), merge the field updates into each payload, and
        merge_insert them back with the SAME vector — no embedding call.
        Fields must already exist in the collection's payload schema. Runs
        under VECTOR_DB_LOCK like every other read→build→write cycle here.
        """
        if not payload_updates:
            return
        if not await self.has_collection(collection_name):
            return
        escaped_ids = [str(id_).replace("'", "''") for id_ in payload_updates]
        if len(escaped_ids) == 1:
            where_clause = f"id = '{escaped_ids[0]}'"
        else:
            id_list = ", ".join(f"'{id_}'" for id_ in escaped_ids)
            where_clause = f"id IN ({id_list})"

        async with self._write_lock():
            collection = await self.get_collection(collection_name)
            schema = await collection.schema()
            # The caller contract: every updated field already exists in the
            # payload struct. pyarrow silently DROPS unknown struct keys when
            # building against a schema, so an unknown field would be a silent
            # no-op rather than an update — refuse it instead.
            payload_type = schema.field("payload").type
            known_fields = {payload_type.field(i).name for i in range(payload_type.num_fields)}
            unknown_fields = {
                field for fields in payload_updates.values() for field in fields
            } - known_fields
            if unknown_fields:
                raise ValueError(
                    f"update_payload: fields {sorted(unknown_fields)} do not exist in the "
                    f"payload schema of collection {collection_name!r}"
                )
            rows = await collection.query().where(where_clause).to_list()
            if not rows:
                return
            records = []
            for row in rows:
                payload = dict(row.get("payload") or {})
                payload.update(payload_updates[str(row["id"])])
                records.append({"id": row["id"], "vector": list(row["vector"]), "payload": payload})
            # Build against the table's OWN arrow schema: plain dicts make
            # merge_insert re-infer types and choke on the fixed-size-list
            # vector column.
            import pyarrow

            arrow_records = pyarrow.Table.from_pylist(records, schema=schema)
            self._mark_written(collection_name)
            await collection.merge_insert("id").when_matched_update_all().execute(arrow_records)

    # Ids per `IN (...)` delete predicate. Each `collection.delete` is a
    # LanceDB commit that appends a table version, and manifest listing slows
    # down as versions accumulate — a delete per id turns a 13k-id wipe into
    # 13k increasingly slow commits (observed as a multi-hour "hang" during
    # `forget(everything=True)`). Batching bounds both the commit count and
    # the predicate size.
    DELETE_PREDICATE_BATCH_SIZE = 1000

    async def delete_data_points(self, collection_name: str, data_point_ids: list[UUID]):
        # Idempotent: a missing collection (or empty id list) is a no-op.
        if not data_point_ids:
            return

        # ids may be UUIDs or graph-computed deterministic strings; the stored
        # `id` column is a str, so match by string and escape single quotes to
        # keep the predicate injection-safe (mirrors create_data_points). The
        # sequential batches cannot commit-conflict with each other, and a
        # non-existent id no-ops. Each batch takes VECTOR_DB_LOCK like every
        # other write here: a Lance delete and a compaction rewrite of the same
        # fragments conflict at commit, and the loser would be this delete
        # (surfacing as a failed forget() or rollback). Per batch, not around
        # the loop, so a large delete does not hold off upserts for its whole
        # duration.
        self._mark_written(collection_name)
        for start in range(0, len(data_point_ids), self.DELETE_PREDICATE_BATCH_SIZE):
            batch = data_point_ids[start : start + self.DELETE_PREDICATE_BATCH_SIZE]
            escaped_ids = [str(data_point_id).replace("'", "''") for data_point_id in batch]
            id_list = ", ".join(f"'{escaped_id}'" for escaped_id in escaped_ids)
            async with self._write_lock():
                if not await self.has_collection(collection_name):
                    return
                collection = await self.get_collection(collection_name)
                await collection.delete(f"id IN ({id_list})")

    async def remove_belongs_to_set_tags(
        self,
        tags: list[str],
        node_ids: list[str] | None = None,
    ) -> None:
        """
        Strip the given tag names from `belongs_to_set` arrays in every
        table and delete rows whose array becomes empty. Used to reconcile
        surviving shared rows after a dataset/NodeSet is deleted.

        When `node_ids` is provided, the detag is scoped to rows whose id
        is in the list — used to reconcile shared rows that lose a
        dataset's anchor while that dataset still exists for other rows.

        LanceDB doesn't support in-place array mutation, so the update path
        reads rows that reference any target tag, rewrites the payload with
        the tag removed, and either re-inserts them (merge_insert) or
        deletes them when the array is empty.
        """
        if not tags:
            return

        if node_ids is not None and not node_ids:
            return

        tag_set = set(tags)
        id_set: set[str] | None = {str(nid) for nid in node_ids} if node_ids is not None else None
        connection = await self.get_connection()
        collection_names = await connection.table_names()

        for collection_name in collection_names:
            try:
                collection = await connection.open_table(collection_name)
            except (ValueError, OSError, RuntimeError) as e:
                logger.debug(
                    "remove_belongs_to_set_tags: could not open '%s': %s",
                    collection_name,
                    e,
                )
                continue

            try:
                arrow_schema = (await collection.to_arrow()).schema
            except Exception as e:
                logger.debug(
                    "remove_belongs_to_set_tags: schema read failed for '%s': %s",
                    collection_name,
                    e,
                    exc_info=True,
                )
                continue

            payload_idx = arrow_schema.get_field_index("payload")
            if payload_idx < 0:
                continue

            payload_type = arrow_schema.field(payload_idx).type
            if not hasattr(payload_type, "num_fields"):
                continue

            payload_fields = {payload_type.field(i).name for i in range(payload_type.num_fields)}
            if "belongs_to_set" not in payload_fields:
                continue

            # Resolve the DataPoint subclass that originally populated this
            # collection so we can round-trip rows through its declared
            # schema on re-add. Without this, `collection.add(dicts)` makes
            # LanceDB infer Arrow types from Python values, which can drift
            # from the stored schema (source of the Rust panics on later
            # vector searches that _migrate_collection_schema warns about).
            resolved_payload_cls = self._resolve_collection_payload_schema(collection_name)

            # Push the predicate into LanceDB so we only materialize the rows
            # that carry at least one of the target tags — mirrors the
            # `array_has_any(payload.belongs_to_set, [...])` filter used in
            # `search()`. Tags are escaped the same way to keep the literal
            # safe from `'` injection.
            escaped_tags = [tag.replace("'", "''") for tag in tag_set]
            literal_tags = "[" + ", ".join(f"'{tag}'" for tag in escaped_tags) + "]"
            where_clause = f"array_has_any(payload.belongs_to_set, {literal_tags})"
            if id_set is not None:
                escaped_ids = [str(nid).replace("'", "''") for nid in id_set]
                literal_ids = "(" + ", ".join(f"'{nid}'" for nid in escaped_ids) + ")"
                where_clause = f"({where_clause}) AND id IN {literal_ids}"

            async with self._write_lock():
                collection = await self.get_collection(collection_name)
                try:
                    rows = await collection.query().where(where_clause).to_list()
                except Exception as e:
                    logger.debug(
                        "remove_belongs_to_set_tags: row scan failed for '%s': %s",
                        collection_name,
                        e,
                        exc_info=True,
                    )
                    continue

                rows_to_delete: list[str] = []
                rows_to_update = []
                for row in rows:
                    payload = row.get("payload") or {}
                    current = payload.get("belongs_to_set") or []
                    if not any(tag in tag_set for tag in current):
                        continue
                    remaining = [tag for tag in current if tag not in tag_set]
                    if remaining:
                        payload["belongs_to_set"] = remaining
                        rows_to_update.append(row)
                    else:
                        rows_to_delete.append(row["id"])

                # Batch deletes into one predicate per bucket so each
                # collection pays two round-trips at most instead of N.
                # Ids are UUID strings produced by cognee so no escaping
                # is needed (mirrors the assumption in `retrieve()`).
                if rows_to_delete or rows_to_update:
                    self._mark_written(collection_name)
                if rows_to_delete:
                    orphan_predicate = (
                        "id IN (" + ", ".join(f"'{row_id}'" for row_id in rows_to_delete) + ")"
                    )
                    await collection.delete(orphan_predicate)

                # LanceDB merge_insert silently no-ops when given dicts whose
                # nested payload shape doesn't match the struct schema, so
                # delete + re-add is the reliable path to persist the
                # rewritten belongs_to_set. If the re-add fails we've
                # already deleted the originals — escalate to WARNING with
                # the affected ids and re-raise so the caller sees it; a
                # silent debug log would leave the collection short of rows.
                if rows_to_update:
                    update_predicate = (
                        "id IN (" + ", ".join(f"'{row['id']}'" for row in rows_to_update) + ")"
                    )
                    await collection.delete(update_predicate)

                    typed_rows = self._coerce_rows_to_typed_payload(
                        rows_to_update, resolved_payload_cls
                    )
                    try:
                        await collection.add(typed_rows)
                    except Exception as e:
                        affected_ids = [row.get("id") for row in rows_to_update]
                        logger.warning(
                            "remove_belongs_to_set_tags: re-add failed for '%s' "
                            "after deleting %d row(s) (ids=%s): %s",
                            collection_name,
                            len(rows_to_update),
                            affected_ids,
                            e,
                        )
                        raise

        return

    async def create_vector_index(self, index_name: str, index_property_name: str):
        await self.create_collection(
            f"{index_name}_{index_property_name}", payload_schema=IndexSchema
        )

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=str(data_point.id),
                    text=getattr(data_point, data_point.metadata["index_fields"][0]),
                    # Reference scalars for search "Evidence". Pulled via getattr
                    # so non-chunk data points (which lack these fields) simply
                    # fall back to None instead of raising.
                    document_id=getattr(data_point, "document_id", None),
                    document_name=getattr(data_point, "document_name", None),
                    chunk_index=getattr(data_point, "chunk_index", None),
                    source_chunk_id=getattr(data_point, "source_chunk_id", None),
                    importance_weight=getattr(data_point, "importance_weight", None),
                    external_metadata=getattr(data_point, "external_metadata", None),
                    belongs_to_set=(data_point.belongs_to_set or []),
                )
                for data_point in data_points
            ],
        )

    async def prune(self):
        # Everything is about to go, so a first-open version prune is pointless;
        # one already running, or a compaction pass (whose version cleanup runs
        # outside VECTOR_DB_LOCK), finishes before the tables are dropped under
        # it. Same lock order as ``compact``: compaction lock, then the write lock.
        self._open_prune_started = True
        if self._followup_prune_handle is not None:
            self._followup_prune_handle.cancel()
            self._followup_prune_handle = None
        await self._wait_for_open_prune()
        connection = await self.get_connection()

        async with self._compaction_lock:
            collection_names = await connection.table_names()
            async with self._write_lock():
                for collection_name in collection_names:
                    collection = await self.get_collection(collection_name)
                    await collection.delete("id IS NOT NULL")
                    await connection.drop_table(collection_name)
            self._compaction_dirty.clear()

        if self.url and not self._is_remote_store(self.url):
            db_dir_path = path.dirname(self.url)
            db_file_name = path.basename(self.url)
            await get_file_storage(db_dir_path).remove_all(db_file_name)

    def get_data_point_schema(self, model_type: BaseModel):
        """Return the storable payload schema for ``model_type``. Memoized on
        the class — repeated calls with the same DataPoint subclass reuse the
        same synthesized Pydantic class instead of re-minting one every time
        (which pydantic's SchemaValidator / SchemaSerializer cache would
        otherwise accumulate indefinitely).
        """
        with self._lance_cache_lock:
            cached = self._payload_schema_cache.get(model_type)
            if cached is not None:
                self._payload_schema_cache.move_to_end(model_type)
                return cached
        cached = self._build_data_point_schema(model_type)
        with self._lance_cache_lock:
            existing = self._payload_schema_cache.get(model_type)
            if existing is not None:
                self._payload_schema_cache.move_to_end(model_type)
                return existing
            self._payload_schema_cache[model_type] = cached
            if len(self._payload_schema_cache) > self._PAYLOAD_SCHEMA_CACHE_SIZE:
                self._payload_schema_cache.popitem(last=False)
        return cached

    @classmethod
    def _make_lance_datapoint_cls(cls, payload_schema, vector_size: int):
        """Return a concrete (non-generic) ``LanceDataPoint`` subclass for the
        given (payload_schema, vector_size) pair. Memoized globally so cognee
        workloads that keep hitting the same DataPoint types don't mint a new
        LanceModel subclass on every insert.
        """
        key = (payload_schema, int(vector_size))
        with cls._lance_cache_lock:
            cached = cls._lance_datapoint_class_cache.get(key)
            if cached is not None:
                cls._lance_datapoint_class_cache.move_to_end(key)
                return cached

        class LanceDataPoint(LanceModel):
            id: str
            vector: Vector(vector_size)
            payload: payload_schema

        with cls._lance_cache_lock:
            existing = cls._lance_datapoint_class_cache.get(key)
            if existing is not None:
                cls._lance_datapoint_class_cache.move_to_end(key)
                return existing
            cls._lance_datapoint_class_cache[key] = LanceDataPoint
            if len(cls._lance_datapoint_class_cache) > cls._LANCE_DATAPOINT_CACHE_SIZE:
                cls._lance_datapoint_class_cache.popitem(last=False)
        return LanceDataPoint

    def _build_data_point_schema(self, model_type: BaseModel):
        related_models_fields = []

        for field_name, field_config in model_type.model_fields.items():
            if hasattr(field_config, "model_fields") or hasattr(
                field_config.annotation, "model_fields"
            ):
                related_models_fields.append(field_name)

            elif (
                # `Optional[X]` / `Union[...]` have origin typing.Union; the PEP 604
                # spelling `X | None` has origin types.UnionType. Treat both alike.
                get_origin(field_config.annotation) in (Union, types.UnionType)
                or get_origin(field_config.annotation) is list
            ):
                models_list = get_args(field_config.annotation)
                if (
                    any(hasattr(model, "model_fields") for model in models_list)
                    or models_list
                    and any(get_args(model) is DataPoint for model in models_list)
                    or models_list
                    and any(submodel is DataPoint for submodel in get_args(models_list[0]))
                ):
                    related_models_fields.append(field_name)

            elif get_origin(field_config.annotation) == Optional:
                model = get_args(field_config.annotation)
                if hasattr(model, "model_fields"):
                    related_models_fields.append(field_name)

        return copy_model(
            model_type,
            include_fields={
                "id": (str, ...),
                "belongs_to_set": (list[str] | None, None),
            },
            exclude_fields=["metadata"] + related_models_fields,
        )

    async def close(self):
        """Release the underlying connection and, in subprocess mode, tear down
        the worker process. Once closed the adapter is not reusable. Idempotent.

        Snapshot lifecycle state atomically under ``_lifecycle_lock``, then
        do the slow teardown (``connection.close()`` is async, ``session.shutdown()``
        can take seconds) outside the lock. The flag is flipped first so a
        concurrent ``get_connection`` that reads the snapshot sees the
        closed state immediately — no new connections after this point.
        """
        if self._followup_prune_handle is not None:
            self._followup_prune_handle.cancel()
            self._followup_prune_handle = None
        await self._wait_for_open_prune()
        with self._lifecycle_lock:
            if self._permanently_closed:
                return  # idempotent
            self._permanently_closed = True
            connection = self.connection
            session = self._session
            self.connection = None
            self._session = None

        # Local-mode connection.close() releases the underlying LanceDB
        # native handles. In subprocess mode the connection is a thin
        # proxy whose lifecycle is owned by ``session.shutdown()`` —
        # closing it separately would just bounce more RPCs to a worker
        # we're about to kill.
        if connection is not None and not self._subprocess_mode:
            try:
                close_result = connection.close()
                if inspect.isawaitable(close_result):
                    await close_result
            except Exception as e:
                logger.warning("Error closing LanceDB connection: %s", e, exc_info=True)
        if session is not None:
            # ``session.shutdown()`` is sync and joins/terminates/kills the
            # worker process — can take seconds. Offload to a worker thread
            # so awaiting ``close()`` doesn't freeze the calling event loop.
            try:
                await asyncio.to_thread(session.shutdown)
            except Exception as e:
                logger.warning("Error shutting down LanceDB subprocess: %s", e, exc_info=True)
