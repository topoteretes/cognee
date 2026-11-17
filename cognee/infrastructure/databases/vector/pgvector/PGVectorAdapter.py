import asyncio
import hashlib
import json
import re
import threading
from typing import Any
from uuid import UUID

import asyncpg
from asyncpg import DeadlockDetectedError, DuplicateTableError, UniqueViolationError
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    Column,
    MetaData,
    Table,
    TextClause,
    Uuid,
    cast,
    delete,
    exc,
    func,
    select,
    text,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, ProgrammingError, SQLAlchemyError
from sqlalchemy.inspection import inspect
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.exceptions import MissingQueryParameterError
from cognee.infrastructure.databases.postgres.deadlock import is_deadlock
from cognee.infrastructure.databases.relational import get_relational_config, get_relational_engine
from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.locks.loop_agnostic_lock import LoopAgnosticLock
from cognee.modules.graph.methods.sanitize_relational_payload import sanitize_relational_payload
from cognee.shared.logging_utils import get_logger

from ...relational.sqlalchemy.SqlAlchemyAdapter import (
    DEFAULT_MAX_OVERFLOW,
    DEFAULT_POOL_SIZE,
    SQLAlchemyAdapter,
)
from ..embeddings.EmbeddingGateway import EmbeddingGateway
from ..exceptions import CollectionNotFoundError, PgvectorVersionError, SharedDatabasePruneError
from ..models.ScoredResult import ScoredResult
from ..stored_vector_size import choose_stored_vector_size
from ..vector_db_interface import VectorDBInterface, validate_raw_vector_points
from .serialize_data import serialize_data

logger = get_logger("PGVectorAdapter")
QUERY_BATCH_SIZE = 1000

# Default pool sizing for per-dataset PGVector engines when ENABLE_BACKEND_ACCESS_CONTROL=True.
# Lean pool_size limits connection fan-out across datasets (only pool_size connections are
# retained while idle); the large max_overflow keeps burst headroom, since overflow
# connections close on release instead of idling.
_ACCESS_CONTROL_DEFAULT_POOL_ARGS = {"pool_size": 2, "max_overflow": 20}

# pgvector builds HNSW indexes on vector up to 2,000 dimensions and on halfvec up to
# 4,000; a larger embedding stays unindexed and is searched by a full scan.
_HNSW_VECTOR_MAX_DIMENSIONS = 2000
_HNSW_HALFVEC_MAX_DIMENSIONS = 4000
# The halfvec type arrived in pgvector 0.7.0 (HNSW in 0.5.0): new collections store
# halfvec, so the adapter requires it (see ``_pgvector_version``).
_MIN_PGVECTOR = (0, 7, 0)
# Links per node (m) and build-time candidate list (ef_construction) of new HNSW
# indexes. pgvector's defaults are m = 16 and ef_construction = 64; only m is changed.
# With many concurrent writers the index insert dominates a write, and its cost grows
# with m: every new row is linked to up to 2*m neighbours on the base layer, and each of
# those rewrites its own neighbour list. Measured locally on 99,328 stored text
# embeddings of 3072 dimensions, inserted by 24 concurrent connections into an indexed
# table: 36.6 s with m = 8 against 80.3 s with m = 16. The price is recall: 0.951 against
# 0.974 (recall@10 against an exact scan, 200 queries, ef_search 40), which searching
# more candidates buys back, see _HNSW_DEFAULT_EF_SEARCH. Existing indexes are not rebuilt
# (CREATE INDEX IF NOT EXISTS) and keep the parameters they were built with.
_HNSW_M = 8
_HNSW_EF_CONSTRUCTION = 64
# An HNSW scan examines hnsw.ef_search candidates and returns at most that many rows.
# Searches use at least 100 (pgvector's own default is 40). In the measurement above that
# gives an index with m = 8 a recall@10 of 0.971, where m = 16 had 0.974 at 40, for a
# median query of 4.8 ms against 2.9 ms; an index built with m = 16 reaches 0.989 at 100.
# A larger limit raises the setting. pgvector caps it at 1,000, so a limit above that
# keeps the exact full scan.
_HNSW_DEFAULT_EF_SEARCH = 100
_HNSW_MAX_EF_SEARCH = 1000
# Postgres truncates identifiers to NAMEDATALEN - 1 bytes, with only a NOTICE.
_PG_MAX_IDENTIFIER_BYTES = 63
# Vectors reach Postgres in binary through an asyncpg pool of the adapter's own
# (``_get_write_pool``): connections open on demand and close after this long idle.
# Its size follows the adapter's pool args (``_write_pool_size``).
_WRITE_POOL_IDLE_SECONDS = 60.0
# SQLAlchemy's asyncpg dialect consumes these connect_args itself; asyncpg.connect
# does not accept them.
_SQLALCHEMY_ONLY_CONNECT_ARGS = ("prepared_statement_cache_size", "prepared_statement_name_func")


def _is_write_deadlock(error: BaseException) -> bool:
    """A deadlock abort from the binary write pool (a bare asyncpg error) or behind
    SQLAlchemy's wrapping (every other statement of this adapter)."""
    return isinstance(error, DeadlockDetectedError) or is_deadlock(error)


def _write_pool_size(pool_args: dict) -> int:
    """Most connections the binary write pool may open: ``pool_size + max_overflow``.

    The same ceiling the adapter's SQLAlchemy pool has, read from the same pool args
    (``VECTOR_POOL_ARGS``, else ``POOL_ARGS``, else the adapter's defaults), so one
    setting sizes both pools. The write pool is a pool of its own, so an adapter can
    hold up to this many connections in each. SQLAlchemy's "no limit" values
    (``pool_size`` 0, a negative ``max_overflow``) have no asyncpg equivalent.
    """
    pool_size = pool_args.get("pool_size", DEFAULT_POOL_SIZE)
    max_overflow = pool_args.get("max_overflow", DEFAULT_MAX_OVERFLOW)
    if pool_size <= 0 or max_overflow < 0:
        raise ValueError(
            "PGVector writes need a bounded connection pool: set a positive pool_size and "
            "a non-negative max_overflow in VECTOR_POOL_ARGS (or POOL_ARGS), got "
            f"pool_size={pool_size}, max_overflow={max_overflow}."
        )
    return pool_size + max_overflow


def _row_order(row_id: Any) -> str:
    """The order rows are written in, so concurrent upserts never deadlock.

    A multi-row ``INSERT ... ON CONFLICT DO UPDATE`` locks each conflicting row as it
    reaches it, in ``VALUES`` order. Two statements that touch the same ids in
    different orders can each hold a row the other waits for, and Postgres then
    aborts one with a deadlock. Written in one global order (the uuid's canonical
    text, which is also its btree order) every statement waits only for rows after
    the ones it holds, so no cycle can form, and the per-collection lock that used to
    serialize every insert of a collection is not needed: inserts into one
    collection run concurrently, and pgvector's HNSW index takes them concurrently.
    """
    return str(row_id)


def _batches(rows: list[dict]) -> list[list[dict]]:
    """``rows`` in consecutive slices of ``QUERY_BATCH_SIZE``, one statement each."""
    return [
        rows[start : start + QUERY_BATCH_SIZE] for start in range(0, len(rows), QUERY_BATCH_SIZE)
    ]


PgvectorVersion = tuple[int, ...]

# Collection tables in the engine's current schema (the dataset's own schema in
# shared-database mode): a ``vector`` column of pgvector's ``vector`` or ``halfvec``
# type, with that type's name and its ``atttypmod``, the declared dimension count (-1
# when undeclared). A partitioned collection is its parent table ('p'); its partitions
# are not collections.
_VECTOR_COLLECTIONS_SQL = """
SELECT c.relname, t.typname, a.atttypmod
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_type t ON t.oid = a.atttypid
WHERE n.nspname = current_schema()
  AND c.relkind IN ('r', 'p')
  AND NOT c.relispartition
  AND a.attname = 'vector'
  AND t.typname IN ('vector', 'halfvec')
  AND NOT a.attisdropped
"""
# An HNSW build is much slower once its graph outgrows maintenance_work_mem (12k
# 3072-dimensional rows on a 1 CU Neon compute: 22.6 s at its 67 MB default, 12.4 s at
# 256 MB). Raised for the build's own transaction only, and never lowered.
_BUILD_MAINTENANCE_WORK_MEM_KB = 256 * 1024
_RAISE_MAINTENANCE_WORK_MEM_SQL = """
SELECT set_config('maintenance_work_mem', :target, true)
FROM pg_settings
WHERE name = 'maintenance_work_mem' AND setting::bigint < :target_kb
"""
# A parallel build keeps the graph in a dynamic shared memory segment of
# maintenance_work_mem bytes, which Postgres backs with /dev/shm. Docker caps /dev/shm at
# 64 MB by default (the shipped docker-compose.yml included), so at the raised setting a
# parallel build fails with "could not resize shared memory segment ... No space left on
# device" and the collection stays unindexed, searched by a full scan. A serial build
# keeps the graph in the backend's own memory and needs no segment; it was not slower at
# the sizes cognee collections reach (24k 3072-dimensional rows: 15.9 s serial, 17.4 s
# with the default two workers). Set for the build's own transaction only.
_SERIAL_BUILD_SQL = "SET LOCAL max_parallel_maintenance_workers = 0"


def parse_pgvector_version(version: str | None) -> PgvectorVersion | None:
    """``pg_extension.extversion`` ("0.8.0") as a comparable tuple, or None if unknown."""
    if not version:
        return None
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts[:3]) if parts else None


def hnsw_index_name(table_name: str) -> str:
    """Index name for a collection, unique per schema even for long collection names.

    A name past 63 bytes would be truncated by Postgres, and two collections sharing their
    first bytes would get the same index name, so the second one would be silently skipped
    by ``IF NOT EXISTS``. Long names get a hash of the full table name instead.
    """
    name = f"{table_name}_vector_hnsw"
    if len(name.encode()) <= _PG_MAX_IDENTIFIER_BYTES:
        return name
    digest = hashlib.sha1(table_name.encode()).hexdigest()[:12]
    prefix = table_name.encode()[:40].decode(errors="ignore")
    return f"{prefix}_{digest}_hnsw"


def partition_name(table_name: str, remainder: int) -> str:
    """Name of one hash partition of a collection's table: ``<collection>_p<remainder>``.

    Hashed like ``hnsw_index_name`` past 63 bytes, so the partitions of two long
    collection names never truncate to the same name.
    """
    name = f"{table_name}_p{remainder}"
    if len(name.encode()) <= _PG_MAX_IDENTIFIER_BYTES:
        return name
    digest = hashlib.sha1(table_name.encode()).hexdigest()[:12]
    prefix = table_name.encode()[:40].decode(errors="ignore")
    return f"{prefix}_{digest}_p{remainder}"


# The partitions of partitioned tables, as (schema, table). Reflection lists them as
# tables of their own, but they are part of their parent's collection.
_PARTITIONS_SQL = """
SELECT n.nspname, c.relname
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relispartition
"""


def _stored_type(vector_column) -> str:
    """``halfvec`` or ``vector``: the type of a collection's (reflected) vector column."""
    from pgvector.sqlalchemy import HALFVEC

    return "halfvec" if isinstance(vector_column.type, HALFVEC) else "vector"


def _indexed_type(dimensions: int, column_type: str) -> str | None:
    """The type the HNSW index of a ``column_type`` column uses (``vector`` or
    ``halfvec``), or None when pgvector cannot index it.

    New collections store ``halfvec`` and are indexed as they are. A collection created
    earlier with a ``vector`` column keeps it (nothing converts it): indexed as ``vector``
    up to 2,000 dimensions and through a halfvec cast above.
    """
    if dimensions <= 0:
        return None
    if column_type == "vector" and dimensions <= _HNSW_VECTOR_MAX_DIMENSIONS:
        return "vector"
    if dimensions <= _HNSW_HALFVEC_MAX_DIMENSIONS:
        return "halfvec"
    return None


def hnsw_index_statement(
    table_name: str, dimensions: int, *, column_type: str, schema: str | None = None
) -> str | None:
    """CREATE INDEX for a collection's vector column, or None when pgvector cannot index it.

    Without an index every search is a full scan that reads every stored vector; a
    3072-dimensional collection of 12k rows is ~160 MB of TOAST pages per query, which
    took 20 s on a Neon compute whose cache was empty.

    The index is on the column itself when it has the indexed type, and on a halfvec
    cast of it when a ``vector`` column is too wide to index (``_indexed_type``).

    With ``schema`` the table is named in that schema, so a schema-pinned adapter
    indexes its own collection and never a same-named one earlier on the search path.
    """
    indexed_type = _indexed_type(dimensions, column_type)
    if indexed_type is None:
        return None
    indexed = "vector" if indexed_type == column_type else f"(vector::halfvec({dimensions}))"
    table = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
    return (
        f'CREATE INDEX IF NOT EXISTS "{hnsw_index_name(table_name)}" '
        f"ON {table} USING hnsw ({indexed} {indexed_type}_cosine_ops) "
        f"WITH (m = {_HNSW_M}, ef_construction = {_HNSW_EF_CONSTRUCTION})"
    )


def _indexed_distance(vector_column, query_vector: list[float], indexed_type: str, dimensions: int):
    """The distance expression the HNSW index orders by (see ``hnsw_index_statement``)."""
    if _stored_type(vector_column) == indexed_type:
        return vector_column.cosine_distance(query_vector)
    from pgvector.sqlalchemy import HALFVEC

    return cast(vector_column, HALFVEC(dimensions)).cosine_distance(query_vector)


def _vector_values(stored) -> list[float]:
    """A stored vector as floats: pgvector returns a ``vector`` column as a numpy array
    and a ``halfvec`` column as a ``HalfVector``."""
    from pgvector.sqlalchemy import HalfVector

    values = stored.to_list() if isinstance(stored, HalfVector) else stored
    return [float(v) for v in values]


class IndexSchema(DataPoint):
    """
    Define a schema for indexing data points with a text field.

    This class inherits from the DataPoint class and specifies the structure of a single
    data point that includes a text attribute. It also includes a metadata field that
    indicates which fields should be indexed.
    """

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


class PGVectorAdapter(SQLAlchemyAdapter, VectorDBInterface):
    """Vector-database adapter backed by Postgres + pgvector; implements VectorDBInterface."""

    name = "PGVector"
    # Postgres schema this adapter is pinned to ("" = the connection's default schema).
    # Class-level defaults so a bare instance (tests build them with object.__new__)
    # reads as unpinned and without a write pool.
    schema: str = ""
    _write_pool: asyncpg.Pool | None = None
    _write_pool_loop = None

    def __init__(
        self,
        connection_string: str,
        api_key: str | None,
        embedding_engine: EmbeddingGateway,
        schema: str = "",
    ):
        """Initialize the adapter and, when possible, reuse the relational engine.

        When ``schema`` is set (shared-database isolation mode), this adapter is
        pinned to a single Postgres schema via the connection ``search_path``:
        every collection table is created and queried inside that schema, so
        many datasets coexist in one database with no table-name collisions and
        no per-dataset database. In that mode the adapter always owns a
        dedicated engine (it must never mutate the search_path of the shared
        relational engine, which has to stay on ``public``).
        """
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.db_uri: str = connection_string
        # Postgres schema this adapter is pinned to ("" = default/public search path).
        # Read by schema-scoped overrides of get_table_names()/delete_database().
        self.schema: str = schema or ""
        # Loop-agnostic: the adapter is cached across event loops, and an
        # asyncio.Lock binds to the first loop that awaits it.
        self.VECTOR_DB_LOCK = LoopAgnosticLock()
        # pgvector's extversion, read on first use (see _pgvector_version).
        self._pgvector_version_cache: PgvectorVersion | None = None
        # Per-collection locks for the delete paths. Inserts do not take them: their
        # rows are written in ``_row_order``, which rules out deadlocks between them.
        self._write_locks: dict[str, LoopAgnosticLock] = {}
        # Guards the check-then-insert in ``_get_write_lock``: callers on
        # different threads/loops must get the same lock for a collection.
        self._write_locks_guard = threading.Lock()
        self._metadata = MetaData()
        # True when this adapter created its own engine and must dispose it on close().
        # False when the engine is borrowed from the relational adapter.
        self._owns_engine: bool = False

        relational_db = get_relational_engine()
        relational_config = get_relational_config()
        vector_config = get_vectordb_config()
        # How many hash partitions a collection created by this adapter gets
        # (PGVECTOR_COLLECTION_PARTITIONS, see ``create_collection``).
        self.collection_partitions: int = vector_config.pgvector_collection_partitions

        # Resolve effective pool_args for any new PGVector engine we create:
        # 1. Explicit VECTOR_POOL_ARGS always wins.
        # 2. Then the relational POOL_ARGS, when configured — an operator who
        #    sized the pool explicitly outranks our built-in default.
        # 3. Otherwise, when access control is on, each dataset gets its own
        #    engine — use a small default to avoid connection fan-out
        #    (N datasets × pool_size).
        if vector_config.vector_pool_args is not None:
            effective_pool_args = dict(vector_config.vector_pool_args)
        elif relational_config.pool_args:
            effective_pool_args = dict(relational_config.pool_args)
        elif backend_access_control_enabled():
            effective_pool_args = _ACCESS_CONTROL_DEFAULT_POOL_ARGS
        else:
            effective_pool_args = {}

        # A per-dataset PGVector engine may connect to managed
        # Postgres (Neon) which requires SSL. Reuse the relational connect_args
        # (e.g. {"ssl": "require"} from DATABASE_CONNECT_ARGS); {} for in-cluster.
        effective_connect_args = (
            dict(relational_config.database_connect_args)
            if relational_config.database_connect_args
            else None
        )

        # Connect args for every connection this adapter opens itself: the SQLAlchemy
        # engine (when it is not borrowed) and the binary write pool. In
        # shared-database schema-isolation mode they are pinned to this dataset's
        # schema via search_path; ``public`` is kept on the path so the pgvector
        # extension's ``vector`` type and operators (installed in public) resolve.
        # Collection tables are named with the schema in every statement, so neither
        # reads nor writes can fall through to a same-named table in public.
        connect_args = dict(effective_connect_args) if effective_connect_args else {}
        if self.schema:
            server_settings = dict(connect_args.get("server_settings") or {})
            server_settings["search_path"] = f"{self.schema}, public"
            connect_args["server_settings"] = server_settings
        self._write_connect_args = {
            key: value
            for key, value in connect_args.items()
            if key not in _SQLALCHEMY_ONLY_CONNECT_ARGS
        }
        self._write_pool = None
        self._write_pool_loop = None
        self._write_pool_lock = LoopAgnosticLock()
        self._write_pool_max_size = _write_pool_size(effective_pool_args)

        # Reuse engine and sessionmaker if the relational engine is provided and is the same database as the one configured for pgvector
        db_name1 = make_url(relational_db.db_uri).database
        db_name2 = make_url(self.db_uri).database
        if self.schema:
            # Shared-database schema-isolation mode: always a dedicated engine, since
            # the shared relational engine's search_path has to stay on ``public``.
            super().__init__(
                connection_string=self.db_uri,
                connect_args=connect_args,
                pool_args=effective_pool_args,
            )
            self._owns_engine = True
        elif db_name1 != db_name2:
            # A different database name is sufficient isolation regardless of access-control
            # mode, so honor VECTOR_DB_NAME instead of borrowing the relational engine.
            super().__init__(
                connection_string=self.db_uri,
                connect_args=effective_connect_args,
                pool_args=effective_pool_args,
            )
            self._owns_engine = True
        elif relational_db.engine.dialect.name == "postgresql":
            # Same PostgreSQL database as the relational engine: reuse its engine and sessionmaker
            self.engine = relational_db.engine
            self.sessionmaker = relational_db.sessionmaker
        else:
            # If not postgreSQL and not backend access control create new instances of engine and sessionmaker
            super().__init__(
                connection_string=self.db_uri,
                connect_args=effective_connect_args,
                pool_args=effective_pool_args,
            )
            self._owns_engine = True

        # Has to be imported at class level
        # Functions reading tables from database need to know what a Vector column type is
        from pgvector.sqlalchemy import Vector

        self.Vector = Vector

    def reset_metadata_cache(self):
        """
        Reset SQLAlchemy metadata reflection cache for this adapter instance.
        """
        self._metadata = MetaData()

    def _table_key(self, collection_name: str) -> str:
        """The key a collection's reflected table has in ``self._metadata``."""
        return f"{self.schema}.{collection_name}" if self.schema else collection_name

    def _qualified_table(self, collection_name: str) -> str:
        """The collection's table as SQL names it: schema-qualified when pinned.

        An unqualified name resolves along the search path, where ``public`` follows
        the pinned schema, so a same-named collection there would be read and written
        instead of this dataset's.
        """
        quoted = f'"{collection_name}"'
        return f'"{self.schema}".{quoted}' if self.schema else quoted

    async def get_table_names(self) -> list[str]:
        """List collection tables, scoped to this adapter's schema when pinned.

        In shared-database mode this adapter's engine is pinned to a single
        Postgres schema (``self.schema``); reflecting every schema (the base
        adapter's behavior) would leak other datasets' collections into
        schema-wide operations such as ``remove_belongs_to_set_tags``. Falls
        back to the base (all non-system schemas) when not schema-pinned.
        """
        if not self.schema:
            table_names = await super().get_table_names()
        else:
            table_names = []
            async with self.engine.begin() as connection:
                metadata = MetaData()
                await connection.run_sync(metadata.reflect, schema=self.schema)
                table_names.extend(metadata.tables.keys())

        # A partitioned collection is listed once, as its parent table. Reflection keys
        # are "schema.table", or a bare "table" in the connection's default schema.
        async with self.engine.connect() as connection:
            partitions = (await connection.execute(text(_PARTITIONS_SQL))).all()
            default_schema = (await connection.execute(text("SELECT current_schema()"))).scalar()
        hidden = {f"{schema}.{name}" for schema, name in partitions}
        return [
            name
            for name in table_names
            if (name if "." in name else f"{default_schema}.{name}") not in hidden
        ]

    async def delete_database(self):
        """Drop this dataset's tables, scoped to its schema when pinned.

        In shared-database mode ``prune`` must only drop the pinned dataset
        schema's tables — never ``public`` (which holds cognee's shared
        relational tables) or other datasets' schemas. Falls back to the base
        public-schema behavior when not schema-pinned.
        """
        if not self.schema:
            return await super().delete_database()

        async with self.engine.begin() as connection:
            metadata = MetaData()
            await connection.run_sync(metadata.reflect, schema=self.schema)
            for table in metadata.sorted_tables:
                await connection.execute(
                    text(f'DROP TABLE IF EXISTS "{self.schema}"."{table.name}" CASCADE')
                )

    async def close(self) -> None:
        """
        Release connection-pool resources held by this adapter.

        Only disposes the engine when this adapter owns it. When the engine is
        borrowed from the relational adapter (shared-engine mode), disposal is
        left to the relational adapter so the pool is not torn down while the
        relational layer is still using it.

        Called automatically by ``closing_lru_cache`` when this adapter is
        evicted from ``_create_vector_engine``'s cache.
        """
        pool, self._write_pool = self._write_pool, None
        pool_loop, self._write_pool_loop = self._write_pool_loop, None
        try:
            if pool is not None:
                if pool_loop is asyncio.get_running_loop():
                    try:
                        await asyncio.wait_for(pool.close(), timeout=10)
                    except asyncio.TimeoutError:
                        self._terminate_write_pool(pool, pool_loop)
                else:
                    self._terminate_write_pool(pool, pool_loop)
        finally:
            if self._owns_engine:
                await self.engine.dispose(close=True)

    @staticmethod
    def _terminate_write_pool(pool, loop) -> None:
        try:
            pool.terminate()
        except RuntimeError:
            # asyncpg aborts transports on their owning loop. Once that loop is
            # closed it cannot schedule transport cleanup; the detached pool must
            # be reclaimed instead. Do not hide errors on a live loop.
            if loop is None or not loop.is_closed():
                raise

    def _get_write_lock(self, collection_name: str) -> LoopAgnosticLock:
        with self._write_locks_guard:
            lock = self._write_locks.get(collection_name)
            if lock is None:
                lock = LoopAgnosticLock()
                self._write_locks[collection_name] = lock
            return lock

    def _asyncpg_dsn(self) -> str:
        """This adapter's URL for asyncpg itself, without the SQLAlchemy driver suffix."""
        return (
            make_url(self.db_uri).set(drivername="postgresql").render_as_string(hide_password=False)
        )

    async def _init_write_connection(self, connection) -> None:
        """Register pgvector's binary codecs on a new write-pool connection.

        The codec is looked up in the schema holding the extension (``public`` unless a
        deployment installed it elsewhere). ``create_collection`` installs the extension
        before the first write, so a missing one here is a configuration error.
        """
        from pgvector.asyncpg import register_vector

        extension_schema = await connection.fetchval(
            "SELECT n.nspname FROM pg_extension e "
            "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'vector'"
        )
        if extension_schema is None:
            raise RuntimeError(
                "The pgvector extension is not installed in the vector database; "
                "create a collection first, which installs it."
            )
        await register_vector(connection, schema=extension_schema)

    async def _get_write_pool(self) -> asyncpg.Pool:
        """The asyncpg pool vectors are written through, opened on first use per event loop.

        Vectors go to Postgres in binary: pgvector's asyncpg codec turns a row's vector
        into the column's bytes in C (half floats for ``halfvec``, float32 for a
        ``vector`` column of an earlier release), where the SQLAlchemy bind processor
        formatted every coordinate as text in Python, about half of a large write's time.
        The pool is
        separate from the SQLAlchemy engine because a codec is registered per
        connection and would change how the engine's own statements, which bind
        vectors as text, encode them. Connections open on demand and close after
        ``_WRITE_POOL_IDLE_SECONDS`` idle, so outside a write the pool holds none; at
        most ``pool_size + max_overflow`` of the adapter's pool args are open at once
        (``_write_pool_size``).

        asyncpg connections belong to the loop that opened them, and this adapter is
        cached across loops (the CLI and the tests run several), so a pool from an
        earlier loop is dropped and a new one opened.
        """
        loop = asyncio.get_running_loop()
        if self._write_pool is not None and self._write_pool_loop is loop:
            return self._write_pool
        async with self._write_pool_lock:
            if self._write_pool is not None and self._write_pool_loop is loop:
                return self._write_pool
            if self._write_pool is not None:
                stale, self._write_pool = self._write_pool, None
                stale_loop, self._write_pool_loop = self._write_pool_loop, None
                self._terminate_write_pool(stale, stale_loop)
            pool = await asyncpg.create_pool(
                self._asyncpg_dsn(),
                min_size=0,
                max_size=self._write_pool_max_size,
                max_inactive_connection_lifetime=_WRITE_POOL_IDLE_SECONDS,
                init=self._init_write_connection,
                **self._write_connect_args,
            )
            self._write_pool = pool
            self._write_pool_loop = loop
            return pool

    async def _write_rows(self, collection_name: str, rows: list[dict], statement_for) -> None:
        """Upsert ``rows`` (``id``, ``payload``, ``vector`` each), in their order, in one
        transaction on the write pool: one statement per row, prepared once per
        connection and pipelined by asyncpg. ``statement_for(table, collection_name)``
        gives the statement; its ``$3`` is the vector, encoded by the binary codec.
        """
        if not rows:
            return
        statement = statement_for(self._qualified_table(collection_name), collection_name)
        params = [(row["id"], json.dumps(row["payload"]), row["vector"]) for row in rows]
        pool = await self._get_write_pool()
        async with pool.acquire() as connection, connection.transaction():
            await connection.executemany(statement, params)

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """
        Embed a list of texts into vectors using the specified embedding engine.

        Parameters:
        -----------

            - data (list[str]): A list of strings to be embedded into vectors.

        Returns:
        --------

            - list[list[float]]: A list of lists of floats representing embedded vectors.
        """
        return await self.embedding_engine.embed_text(data)

    async def get_stored_vector_size(self) -> int | None:
        """Width of the vectors already in this store, or None when nothing is stored yet.

        Every table's ``vector`` column is declared ``halfvec(N)`` (``vector(N)`` in
        collections created by earlier releases) with the embedding width that built
        it. Read once per dataset by the dataset context to record the width for rows
        that predate the recorded embedding model.

        Reads every such column rather than the first: a schema built across an
        ``EMBEDDING_MODEL`` change holds two widths, and which one gets recorded
        must not depend on catalog order (see ``choose_stored_vector_size``).
        """
        async with self.get_async_session() as session:
            result = await session.execute(
                text(
                    "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                    "JOIN pg_class c ON c.oid = a.attrelid "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = :schema AND a.attname = 'vector' "
                    "AND NOT a.attisdropped"
                ),
                {"schema": self.schema or "public"},
            )
            declared_types = result.scalars().all()

        widths = []
        for declared_type in declared_types:
            match = re.fullmatch(r"(?:vector|halfvec)\((\d+)\)", declared_type or "")
            if match:
                widths.append(int(match.group(1)))
        return choose_stored_vector_size(widths, self.name)

    async def _pgvector_version(self, executor=None) -> PgvectorVersion | None:
        """The installed pgvector version; raises ``PgvectorVersionError`` below 0.7.0.

        Cached once known and supported. None (the extension is not installed yet) is not
        cached, so the first collection created after ``CREATE EXTENSION`` still checks
        the version; an unsupported one is not cached, so an upgraded extension
        (``ALTER EXTENSION vector UPDATE``) is seen without a restart.
        """
        if self._pgvector_version_cache is not None:
            return self._pgvector_version_cache
        statement = text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        if executor is not None:  # an open connection or session: no extra checkout
            raw = (await executor.execute(statement)).scalar_one_or_none()
        else:
            async with self.engine.connect() as own_connection:
                raw = (await own_connection.execute(statement)).scalar_one_or_none()
        version = parse_pgvector_version(raw)
        if version is not None and version < _MIN_PGVECTOR:
            raise PgvectorVersionError(raw, ".".join(map(str, _MIN_PGVECTOR)))
        self._pgvector_version_cache = version
        return version

    async def existing_collections(self, collection_names: list[str]) -> list[str]:
        """Return the given collections that exist, read in one round trip.

        A missing table is never cached by ``get_table``, so a search over many
        collection names would otherwise reflect each missing one on its own
        connection, every search. The list holds the tables of this adapter's schema
        (the pinned one, or the connection's default), the same resolution
        ``get_table`` uses.
        """
        schema = self.schema or None
        async with self.engine.connect() as connection:
            table_names = set(
                await connection.run_sync(
                    lambda sync_conn: inspect(sync_conn).get_table_names(schema=schema)
                )
            )
        return [name for name in collection_names if name in table_names]

    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if a specified collection exists in this adapter's schema.

        A schema-pinned adapter looks only in its own schema. ``public`` stays on its
        search path for pgvector's types, and a same-named collection there used to
        count as this dataset's, so its rows went to that table.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check for existence.

        Returns:
        --------

            - bool: Returns True if the collection exists, False otherwise.
        """
        key = self._table_key(collection_name)
        if key in self._metadata.tables:
            return True

        try:
            async with self.engine.begin() as connection:
                await connection.run_sync(
                    self._metadata.reflect, only=[collection_name], schema=self.schema or None
                )
        except exc.InvalidRequestError:
            return False

        return key in self._metadata.tables

    async def delete_collection(self, collection_name: str) -> None:
        """Drop the table for ``collection_name``; a missing table is not an error.

        Named with the schema when pinned, like every other statement here, so a
        schema-pinned adapter drops its own schema's table.
        """
        async with self.engine.begin() as connection:
            await connection.execute(
                text(f"DROP TABLE IF EXISTS {self._qualified_table(collection_name)} CASCADE")
            )
        table = self._metadata.tables.get(self._table_key(collection_name))
        if table is not None:
            self._metadata.remove(table)

    @retry(
        retry=retry_if_exception_type(
            (DuplicateTableError, UniqueViolationError, ProgrammingError)
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=6),
    )
    async def create_collection(self, collection_name: str, payload_schema=None):
        """Create the pgvector table for `collection_name` if it does not already exist.

        With ``PGVECTOR_COLLECTION_PARTITIONS`` above 1 the table is hash-partitioned
        by ``id`` into that many partitions, and Postgres builds the HNSW index once
        per partition. Concurrent inserts are then spread over that many index graphs
        instead of meeting on the pages of one, which is what limits many writers of
        one collection. The price is on the read side: a search reads every
        partition's index. The setting only decides how a new collection is created;
        an existing one keeps its layout and is read and written the same either way.
        """
        vector_size = self.embedding_engine.get_vector_size()
        # The primary key (``id``) is the partition key, as Postgres requires of a
        # partitioned table's primary key, so ``ON CONFLICT (id)`` works unchanged.
        partition_args = (
            {"postgresql_partition_by": "HASH (id)"} if self.collection_partitions > 1 else {}
        )

        if not await self.has_collection(collection_name):
            async with self.VECTOR_DB_LOCK:
                if not await self.has_collection(collection_name):
                    from pgvector.sqlalchemy import HALFVEC

                    # A plain Table, not a declarative class: one class per collection
                    # used to be registered on the shared declarative Base, and warned
                    # on every repeat. Same DDL: a UUID primary key, a JSON payload and
                    # a HALFVEC(size) column, in the pinned schema when there is one.
                    # halfvec: half the bytes per row of vector to store and write, and
                    # the only type HNSW indexes above 2,000 dimensions.
                    table = Table(
                        collection_name,
                        MetaData(),
                        Column("id", Uuid, primary_key=True),
                        Column("payload", JSON),
                        Column("vector", HALFVEC(vector_size)),
                        schema=self.schema or None,
                        **partition_args,
                    )

                    async with self.engine.begin() as connection:
                        # The dataset handlers install the extension in the databases
                        # they create; the deployment's default database (dataset
                        # routing) has no handler, so the collection installs it.
                        # A no-op where it is installed.
                        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                        # After CREATE EXTENSION, so an outdated pgvector fails here with
                        # its own error, before the halfvec column is created.
                        await self._pgvector_version(connection)
                        await connection.run_sync(table.metadata.create_all, tables=[table])
                        for remainder in range(self.collection_partitions if partition_args else 0):
                            partition = self._qualified_table(
                                partition_name(collection_name, remainder)
                            )
                            await connection.execute(
                                text(
                                    f"CREATE TABLE {partition} PARTITION OF "
                                    f"{self._qualified_table(collection_name)} FOR VALUES WITH "
                                    f"(MODULUS {self.collection_partitions}, "
                                    f"REMAINDER {remainder})"
                                )
                            )
                        # On a partitioned table the statement below creates the index
                        # on every partition and attaches them to the parent's.
                        # Same transaction as the table: if the index fails, the table
                        # rolls back with it and the retry starts over, instead of the
                        # retry finding the table and leaving it unindexed.
                        index_statement = hnsw_index_statement(
                            collection_name,
                            vector_size,
                            column_type="halfvec",
                            schema=self.schema or None,
                        )
                        if index_statement is not None:
                            await connection.execute(text(index_statement))
                    # Reflect AFTER the DDL transaction commits so
                    # _metadata is never populated for a table that
                    # might be rolled back.
                    async with self.engine.begin() as connection:
                        await connection.run_sync(
                            self._metadata.reflect,
                            only=[collection_name],
                            schema=self.schema or None,
                        )

    @staticmethod
    def _merged_payload_sql(collection_name: str, incoming: str) -> str:
        """The upserted payload as jsonb SQL: the incoming one with ``belongs_to_set``
        unioned with the stored row's, so an upsert never drops a tag the row already carries.

        ``incoming`` names the relation holding the incoming row's ``payload`` column:
        ``EXCLUDED`` in the upsert, the incoming rows in ``_classify_sql``, so the two
        cannot disagree on what an upsert would write.

        A side whose ``belongs_to_set`` is absent or JSON ``null`` (a DataPoint's default)
        counts as empty: ``COALESCE`` alone leaves a JSON ``null`` in place, and
        ``'null' || '["a"]'`` is ``[null, "a"]``.
        """
        quoted_table = f'"{collection_name}"'
        stored_tags = f"NULLIF({quoted_table}.payload::jsonb->'belongs_to_set', 'null'::jsonb)"
        incoming_tags = f"NULLIF({incoming}.payload::jsonb->'belongs_to_set', 'null'::jsonb)"
        return f"""
            jsonb_set(
                {incoming}.payload::jsonb,
                '{{belongs_to_set}}',
                (
                    SELECT COALESCE(jsonb_agg(DISTINCT val), '[]'::jsonb)
                    FROM jsonb_array_elements_text(
                        COALESCE({stored_tags}, '[]'::jsonb)
                        || COALESCE({incoming_tags}, '[]'::jsonb)
                    ) AS val
                )
            )
            """

    @staticmethod
    def _vector_if_text_changed_sql(collection_name: str) -> str:
        """``ON CONFLICT`` vector: the new embedding when the row's text changed (a summary
        keeps its id across re-extraction), else the stored one, so a metadata-only change
        leaves the indexed column alone and can stay a HOT update.
        """
        table = f'"{collection_name}"'
        return (
            f"CASE WHEN {table}.payload::jsonb->'text' IS DISTINCT FROM "
            f"EXCLUDED.payload::jsonb->'text' THEN EXCLUDED.vector ELSE {table}.vector END"
        )

    @staticmethod
    def _payload_changed_sql(collection_name: str, incoming: str) -> str:
        """SQL condition: the merged payload differs from the stored one. The upsert's
        ``ON CONFLICT`` condition and ``_classify_sql``'s filter (``incoming`` as in
        ``_merged_payload_sql``).

        The timestamps are stamped on every write and read by no one, so they are left
        out: an unchanged re-upsert writes no new row version and no HNSW index entry.
        """
        volatile = "- 'created_at' - 'updated_at'"
        merged = PGVectorAdapter._merged_payload_sql(collection_name, incoming)
        stored = f'"{collection_name}".payload::jsonb'
        return f"(({merged}) {volatile}) IS DISTINCT FROM ({stored} {volatile})"

    @classmethod
    def _classify_sql(cls, table: str, collection_name: str) -> str:
        """The incoming rows an upsert would write, each with whether its id is missing.

        A plain ``SELECT``: it takes no row lock and waits for none. An existing row
        the upsert would leave as it is (``_payload_changed_sql`` false) is not returned.
        ``$1`` and ``$2`` are parallel arrays: the ids, and the payloads as JSON text.
        ``table`` is the schema-qualified name; the conditions refer to it by its bare
        name, as the upsert's conflict clause does.
        """
        stored = f'"{collection_name}"'
        return (
            f"SELECT incoming.id, {stored}.id IS NULL AS missing "
            "FROM unnest($1::uuid[], $2::text[]) AS incoming(id, payload) "
            f"LEFT JOIN {table} ON {stored}.id = incoming.id "
            f"WHERE {stored}.id IS NULL "
            f"OR {cls._payload_changed_sql(collection_name, 'incoming')}"
        )

    async def _classify(
        self, connection, collection_name: str, rows: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Split ``rows`` into (missing, changed) against the table as ``connection``
        sees it, dropping rows whose upsert would change nothing. Keeps the input order
        within each list."""
        statement = self._classify_sql(self._qualified_table(collection_name), collection_name)
        is_missing: dict[str, bool] = {}
        for batch in _batches(rows):
            records = await connection.fetch(
                statement,
                [row["id"] for row in batch],
                [json.dumps(row["payload"]) for row in batch],
            )
            is_missing.update((str(record["id"]), record["missing"]) for record in records)
        missing = [row for row in rows if is_missing.get(str(row["id"])) is True]
        changed = [row for row in rows if is_missing.get(str(row["id"])) is False]
        return missing, changed

    @staticmethod
    def _data_point_insert_sql(table: str) -> str:
        """Insert a row unless its id exists, returning the id only when inserted.

        ``DO NOTHING`` takes no lock on an existing row; it waits only while another
        transaction's insert of the same id is uncommitted.
        """
        return (
            f"INSERT INTO {table} (id, payload, vector) VALUES ($1, $2, $3) "
            "ON CONFLICT (id) DO NOTHING RETURNING id"
        )

    @classmethod
    def _data_point_upsert_sql(cls, table: str, collection_name: str) -> str:
        """The statement ``create_data_points`` runs per changed row: insert, or on a
        conflicting id merge ``belongs_to_set``, keep the vector unless the text changed,
        and skip the update altogether when nothing but the timestamps differs. ``table``
        is the schema-qualified name; the conflict clause refers to the target by its
        bare name.
        """
        return (
            f"INSERT INTO {table} (id, payload, vector) VALUES ($1, $2, $3) "
            "ON CONFLICT (id) DO UPDATE SET "
            f"payload = ({cls._merged_payload_sql(collection_name, 'EXCLUDED')})::json, "
            f"vector = {cls._vector_if_text_changed_sql(collection_name)} "
            f"WHERE {cls._payload_changed_sql(collection_name, 'EXCLUDED')}"
        )

    @classmethod
    def _raw_vector_upsert_sql(cls, table: str, collection_name: str) -> str:
        """The statement ``upsert_raw_vectors`` runs per row: a conflicting id takes the
        new payload (tags merged) and the new vector."""
        return (
            f"INSERT INTO {table} (id, payload, vector) VALUES ($1, $2, $3) "
            "ON CONFLICT (id) DO UPDATE SET "
            f"payload = ({cls._merged_payload_sql(collection_name, 'EXCLUDED')})::json, "
            "vector = EXCLUDED.vector"
        )

    @retry(
        retry=retry_if_exception(_is_write_deadlock),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=6),
        reraise=True,
    )
    async def create_data_points(self, collection_name: str, data_points: list[DataPoint]):
        """Upsert DataPoints into `collection_name`, merging belongs_to_set on conflict.

        Only the rows an upsert would write are embedded: a re-run over data already
        stored would otherwise embed every row and then write none of them.

        The rows reach Postgres as plain ``(id, payload, vector)`` tuples over the binary
        write pool: no ORM object per row, and the vector in the column's binary form
        instead of a text literal built coordinate by coordinate.
        """
        if not data_points:
            return
        if not await self.has_collection(collection_name):
            await self.create_collection(
                collection_name=collection_name,
                payload_schema=type(data_points[0]),
            )

        rows, texts = self._data_point_rows(data_points)

        # Which rows would an upsert write at all? Asked in a short read of its own, on a
        # connection given back before the embedding request, so no transaction stays
        # open across it. A changed row whose text is the same is embedded too, although
        # the upsert keeps its stored vector.
        pool = await self._get_write_pool()
        async with pool.acquire() as connection:
            missing, changed = await self._classify(connection, collection_name, rows)
        if not missing and not changed:
            return

        # The write classifies every row again inside its transaction. A row left out
        # here that was deleted or changed during the embedding request comes back
        # without a vector: the transaction is left without writing, those rows are
        # embedded, and the write runs again. Each round embeds only rows that have no
        # vector yet, so the rounds end once every row has one.
        vectors: dict[str, Any] = {}
        to_embed = missing + changed
        while to_embed:
            embedded = await self.embed_data([texts[str(row["id"])] for row in to_embed])
            # zip() below stops at the shorter list, so a provider that answers with
            # fewer vectors than inputs would store part of the batch and report success.
            if len(embedded) != len(to_embed):
                raise ValueError(
                    f"The embedding provider returned {len(embedded)} vectors for "
                    f"{len(to_embed)} inputs; nothing was written to {collection_name!r}."
                )
            vectors.update(zip([str(row["id"]) for row in to_embed], embedded))
            to_embed = await self._write_data_point_rows(collection_name, rows, vectors)

    @staticmethod
    def _data_point_rows(data_points: list[DataPoint]) -> tuple[list[dict], dict[str, str]]:
        """One ``id``/``payload`` row per id, in ``_row_order``, and each id's text to
        embed (keyed by the id's text).

        The same id twice would otherwise be two upserts of one row, and entity
        extraction may emit same-UUID5 rows in one batch. The last duplicate's payload
        and text win, with the duplicates' belongs_to_set arrays unioned so a tag that
        only appears on one duplicate isn't dropped (mirrors the in-batch tag merge in
        Neo4jAdapter.add_nodes). The rows are written in ``_row_order`` so concurrent
        batches cannot deadlock.
        """
        rows_by_id: dict[str, dict] = {}
        texts: dict[str, str] = {}
        for data_point in data_points:
            # Strip NUL bytes: the json column accepts \u0000 on insert, but
            # the payload::jsonb casts in search/merge queries reject it.
            payload = sanitize_relational_payload(serialize_data(data_point.model_dump()))
            key = str(data_point.id)
            existing = rows_by_id.get(key)
            if existing is not None:
                existing_tags = (existing["payload"] or {}).get("belongs_to_set") or []
                incoming_tags = (payload or {}).get("belongs_to_set") or []
                if existing_tags or incoming_tags:
                    merged_tags = list(dict.fromkeys(list(existing_tags) + list(incoming_tags)))
                    payload = {**(payload or {}), "belongs_to_set": merged_tags}
            rows_by_id[key] = {"id": data_point.id, "payload": payload}
            texts[key] = DataPoint.get_embeddable_data(data_point)
        rows = sorted(rows_by_id.values(), key=lambda row: _row_order(row["id"]))
        return rows, texts

    async def _write_data_point_rows(
        self, collection_name: str, rows: list[dict], vectors: dict[str, Any]
    ) -> list[dict]:
        """Write ``rows`` (``id``, ``payload``) with ``vectors`` (by the id's text),
        locking only the rows it changes, in one transaction on the write pool.

        Returns the rows the write needs but ``vectors`` lacks, having written nothing
        (an empty list once the write is committed): the caller embeds them and calls
        again, so no transaction stays open across an embedding request.

        ``ON CONFLICT DO UPDATE`` locks every conflicting row until commit, even when
        its ``WHERE`` is false, so an id many concurrent writers share (the same entity
        extracted from many documents) queued every writer behind every other one,
        although nearly all of them changed nothing. So only the rows that need it
        take a row lock: a lock-free classify drops the unchanged rows, the missing
        ones are inserted with ``DO NOTHING`` (which locks no existing row), and only
        what is left goes through the upsert. All inserts of the call precede all its
        upserts, each phase in ``_row_order``: a writer waiting for another's insert
        holds no updated row's lock yet, so the two phases form no wait cycle.
        """

        def params(batch: list[dict]) -> list[tuple]:
            return [
                (row["id"], json.dumps(row["payload"]), vectors[str(row["id"])]) for row in batch
            ]

        table = self._qualified_table(collection_name)
        pool = await self._get_write_pool()
        async with pool.acquire() as connection, connection.transaction():
            missing, changed = await self._classify(connection, collection_name, rows)
            # A lost-race row (below) was missing here, so it has a vector too.
            unembedded = [row for row in missing + changed if str(row["id"]) not in vectors]
            if unembedded:
                return unembedded

            inserted_ids: set[str] = set()
            if missing:
                inserted = await connection.fetchmany(
                    self._data_point_insert_sql(table), params(missing)
                )
                inserted_ids = {str(record["id"]) for record in inserted}

            # An id another writer inserted after the classify: classified again, it
            # is now an existing row, written below only if the upsert would change it.
            lost_race = [row for row in missing if str(row["id"]) not in inserted_ids]
            still_missing, changed_since = await self._classify(
                connection, collection_name, lost_race
            )

            to_upsert = sorted(
                changed + still_missing + changed_since, key=lambda row: _row_order(row["id"])
            )
            if to_upsert:
                await connection.executemany(
                    self._data_point_upsert_sql(table, collection_name), params(to_upsert)
                )
        return []

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """Create the underlying index collection (table) for the given name/property pair."""
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        """Write index rows derived from `data_points` into the `{index}_{property}` table."""
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=data_point.id,
                    text=DataPoint.get_embeddable_data(data_point),
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

    async def get_table(self, collection_name: str) -> Table:
        """
        Dynamically loads a table using the given collection name
        with an async engine.
        """
        key = self._table_key(collection_name)
        if key in self._metadata.tables:
            return self._metadata.tables[key]

        try:
            async with self.engine.begin() as connection:
                await connection.run_sync(
                    self._metadata.reflect, only=[collection_name], schema=self.schema or None
                )
        except exc.InvalidRequestError:
            raise CollectionNotFoundError(
                f"Collection '{collection_name}' not found!",
            )

        if key in self._metadata.tables:
            return self._metadata.tables[key]

        raise CollectionNotFoundError(
            f"Collection '{collection_name}' not found!",
        )

    supports_payload_update = True

    async def update_payload(
        self, collection_name: str, payload_updates: dict[str, dict[str, Any]]
    ) -> None:
        """Update payload fields on existing rows WITHOUT re-embedding.

        Read-modify-write on the JSON payload column only — the vector column
        is never touched, so no embedding call happens. Missing ids skip.
        """
        if not payload_updates:
            return
        try:
            PGVectorDataPoint = await self.get_table(collection_name)
        except CollectionNotFoundError:
            return

        ids = list(payload_updates.keys())
        async with self.get_async_session() as session:
            rows = (
                await session.execute(
                    select(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(ids))
                )
            ).all()
            for row in rows:
                payload = dict(row.payload or {})
                updates = payload_updates[str(row.id)]
                # Caller contract: the fields already exist in the payload. The
                # JSON column would accept anything, so an unknown field would
                # drift the schema silently — refuse it instead.
                unknown_fields = set(updates) - set(payload)
                if unknown_fields:
                    raise ValueError(
                        f"update_payload: fields {sorted(unknown_fields)} do not exist in the "
                        f"payload of {collection_name!r} row {row.id}"
                    )
                payload.update(updates)
                await session.execute(
                    PGVectorDataPoint.update()
                    .where(PGVectorDataPoint.c.id == row.id)
                    .values(payload=payload)
                )
            await session.commit()

    @retry(
        retry=retry_if_exception(_is_write_deadlock),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=6),
        reraise=True,
    )
    async def upsert_raw_vectors(
        self,
        collection_name: str,
        points: list[dict],
        payload_schema: type[BaseModel] | None = None,
    ) -> None:
        """Upsert caller-provided vectors without invoking the embedding engine.

        Payloads are stored as given (the JSON column has no schema; ``payload_schema``
        only names the collection's type for ``create_collection``), so a row read with
        ``retrieve(..., include_vector=True)`` lands here unchanged. A conflicting id
        takes the new payload and the new vector, except ``belongs_to_set``, which is
        unioned with the stored row's as in ``create_data_points``. The same id twice
        in one call is written once, see ``validate_raw_vector_points``. Rows are
        written in ``_row_order``, like ``create_data_points``, so concurrent upserts
        of the same ids cannot deadlock, and over the same binary write pool.
        """
        if not points:
            return

        vector_size = self.embedding_engine.get_vector_size()
        # Every check runs before the table exists, so a rejected call creates nothing.
        rows = sorted(
            (
                {
                    "id": row["id"],
                    "vector": row["vector"],
                    "payload": sanitize_relational_payload(serialize_data(row["payload"])),
                }
                for row in validate_raw_vector_points(points, vector_size).values()
            ),
            key=lambda row: _row_order(row["id"]),
        )

        if not await self.has_collection(collection_name):
            await self.create_collection(collection_name, payload_schema=payload_schema)

        await self._write_rows(collection_name, rows, self._raw_vector_upsert_sql)

    async def retrieve(
        self, collection_name: str, data_point_ids: list[str], *, include_vector: bool = False
    ):
        """Return rows from `collection_name` matching any of `data_point_ids`.

        With ``include_vector`` the stored embedding rides along as ``payload["vector"]``.
        """
        # Get PGVectorDataPoint Table from database
        try:
            PGVectorDataPoint = await self.get_table(collection_name)
        except CollectionNotFoundError:
            # If collection doesn't exist, return empty list (no items to retrieve)
            return []

        async with self.get_async_session() as session:
            results = []
            for start_index in range(0, len(data_point_ids), QUERY_BATCH_SIZE):
                id_batch = data_point_ids[start_index : start_index + QUERY_BATCH_SIZE]
                batch_results = await session.execute(
                    select(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(id_batch))
                )
                results.extend(batch_results.all())

            seen_ids = set()
            unique_results = []
            for result in results:
                if result.id in seen_ids:
                    continue
                seen_ids.add(result.id)
                unique_results.append(result)

            return [
                ScoredResult(
                    id=parse_id(result.id),
                    # pgvector hands a vector column back as a numpy array and a halfvec
                    # one as a HalfVector; callers get floats.
                    payload={**result.payload, "vector": _vector_values(result.vector)}
                    if include_vector
                    else result.payload,
                    score=0,
                )
                for result in unique_results
            ]

    async def score_by_ids(
        self, collection_name: str, data_point_ids: list[str], query_vector: list[float]
    ) -> list[ScoredResult]:
        ids = list(dict.fromkeys(str(point_id) for point_id in data_point_ids))
        if not ids:
            return []
        table = await self.get_table(collection_name)
        scores = []
        async with self.get_async_session() as session:
            for start in range(0, len(ids), QUERY_BATCH_SIZE):
                batch = ids[start : start + QUERY_BATCH_SIZE]
                rows = await session.execute(
                    select(
                        table.c.id, table.c.vector.cosine_distance(query_vector).label("distance")
                    ).where(table.c.id.in_(batch))
                )
                scores.extend(
                    ScoredResult(id=parse_id(str(row.id)), score=float(row.distance), payload=None)
                    for row in rows.all()
                )
        return scores

    async def score_by_ids_batch(
        self, collection_name: str, data_point_ids: list[str], query_vectors: list[list[float]]
    ) -> list[list[ScoredResult]]:
        """One statement per id batch, with a distance column per query vector.

        The rows are visited once and the distances are computed in Postgres, so no
        vector leaves the database: fetching 3,000 vectors to score them locally costs
        more than the per-query statements it would replace.
        """
        ids = list(dict.fromkeys(str(point_id) for point_id in data_point_ids))
        if not ids or not query_vectors:
            return [[] for _ in query_vectors]
        table = await self.get_table(collection_name)
        distances = [
            table.c.vector.cosine_distance(query_vector).label(f"distance_{index}")
            for index, query_vector in enumerate(query_vectors)
        ]
        scores: list[list[ScoredResult]] = [[] for _ in query_vectors]
        async with self.get_async_session() as session:
            for start in range(0, len(ids), QUERY_BATCH_SIZE):
                batch = ids[start : start + QUERY_BATCH_SIZE]
                rows = await session.execute(
                    select(table.c.id, *distances).where(table.c.id.in_(batch))
                )
                for row in rows.all():
                    row_id = parse_id(str(row.id))
                    for index, query_scores in enumerate(scores):
                        query_scores.append(
                            ScoredResult(id=row_id, score=float(row[index + 1]), payload=None)
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
    ) -> list[ScoredResult]:
        """Run a cosine-distance similarity search, optionally filtered by NodeSet tag."""
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        # Get PGVectorDataPoint Table from database
        PGVectorDataPoint = await self.get_table(collection_name)

        # A search without a limit wants every row. The approximate index does not
        # guarantee that, so it keeps the exact scan whatever the row count turns out to be.
        exact_scan_required = limit is None
        if limit is None:
            async with self.get_async_session() as session:
                query = select(func.count()).select_from(PGVectorDataPoint)
                result = await session.execute(query)
                limit = result.scalar_one()

        # If limit is still 0, no need to do the search, just return empty results
        if limit <= 0:
            return []

        # NOTE: This needs to be initialized in case search doesn't return a value
        closest_items = []

        # Never select the vector column: results carry only id, payload and distance,
        # and the distance is computed in the database. Fetching vectors made a
        # limit=None search (node-set filtered retrieval reads whole collections)
        # decode every stored embedding in Python - several GB for ~25k 3072-dim rows.
        select_columns = (
            [PGVectorDataPoint.c.id, PGVectorDataPoint.c.payload]
            if include_payload
            else [PGVectorDataPoint.c.id]
        )
        # Use async session to connect to the database
        async with self.get_async_session() as session:
            # The column was sized by this engine in create_collection.
            dimensions = self.embedding_engine.get_vector_size()
            await self._pgvector_version(session)  # fails fast on an outdated pgvector
            # The collection's own column type: one created as ``vector`` keeps it.
            column_type = _stored_type(PGVectorDataPoint.c.vector)
            indexed_type = _indexed_type(dimensions, column_type)
            use_index = (
                indexed_type is not None
                and not node_name
                and not exact_scan_required
                and limit <= _HNSW_MAX_EF_SEARCH
            )
            if indexed_type == column_type and not use_index:
                # When the index is on the column itself (every indexed halfvec column, a
                # vector column up to 2,000 dimensions), Postgres would also pick it for
                # an exact ORDER BY and return at most hnsw.ef_search rows, before any
                # node-set filter is applied. Searches that must be exact keep the full
                # scan.
                await session.execute(text("SET LOCAL enable_indexscan = off"))
            if node_name:
                if node_name_filter_operator == "AND":
                    filter_operator = "?&"
                else:
                    filter_operator = "?|"

                from sqlalchemy import bindparam, cast
                from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TEXT

                target = bindparam("target", value=node_name, type_=ARRAY(TEXT()))
                query = (
                    select(
                        *select_columns,
                        PGVectorDataPoint.c.vector.cosine_distance(query_vector).label(
                            "similarity"
                        ),
                    )
                    .where(
                        cast(PGVectorDataPoint.c.payload, JSONB)
                        .op("->")("belongs_to_set")
                        .op(filter_operator)(target)
                    )
                    .order_by("similarity")
                )
            else:
                query = select(
                    *select_columns,
                    PGVectorDataPoint.c.vector.cosine_distance(query_vector).label("similarity"),
                )
                if use_index:
                    # Order by the indexed expression so the HNSW index serves the scan.
                    # The planner underestimates a full scan of TOASTed vectors and picks
                    # it even when the index exists, so it is steered off it here; the
                    # returned similarity stays the exact cosine distance, and the rows are
                    # re-sorted by it below.
                    query = query.order_by(
                        _indexed_distance(
                            PGVectorDataPoint.c.vector, query_vector, indexed_type, dimensions
                        )
                    )
                    await session.execute(text("SET LOCAL enable_seqscan = off"))
                    await session.execute(
                        text(f"SET LOCAL hnsw.ef_search = {max(limit, _HNSW_DEFAULT_EF_SEARCH)}")
                    )
                else:
                    query = query.order_by("similarity")

            if limit > 0:
                query = query.limit(limit)

            # Find closest vectors to query_vector
            closest_items = await session.execute(query)

        vector_list = []

        # Extract distances and find min/max for normalization
        for vector in closest_items.all():
            vector_list.append(
                {
                    "id": parse_id(str(vector.id)),
                    "payload": vector.payload if include_payload else None,
                    "_distance": vector.similarity,
                }
            )

        if len(vector_list) == 0:
            return []

        # An index scan returns rows in the index's order (half precision through a
        # halfvec cast); callers expect them sorted by the distance they carry.
        vector_list.sort(key=lambda row: row["_distance"])

        # Return backend raw cosine distance as score (lower is better)
        for i in range(len(vector_list)):
            vector_list[i]["score"] = float(vector_list[i]["_distance"])

        # Create and return ScoredResult objects
        return [
            ScoredResult(
                id=row.get("id"),
                payload=row.get("payload") if include_payload else None,
                score=row.get("score"),
            )
            for row in vector_list
        ]

    async def batch_search(
        self,
        collection_name: str,
        query_texts: list[str],
        limit: int | None = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: list[str] | None = None,
    ):
        """Run `search` concurrently for each query text and return a list of result lists."""
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

    @retry(
        retry=retry_if_exception(is_deadlock),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=6),
        reraise=True,
    )
    async def delete_data_points(self, collection_name: str, data_point_ids: list[UUID]):
        """Delete rows from `collection_name` whose id is in `data_point_ids`.

        Inserts no longer take the collection lock, so a delete can meet a concurrent
        upsert of the same rows; Postgres aborts one side of such a deadlock and the
        delete is retried.
        """
        # Skip deletion if collection doesn't exist
        if not await self.has_collection(collection_name):
            return None

        async with self._get_write_lock(collection_name):
            # Resolve the table BEFORE opening the session. get_table() checks out
            # its own connection; doing it inside the session would hold two pooled
            # connections at once and deadlock the pool under concurrency (same
            # class as #4197). Mirrors retrieve()/search().
            PGVectorDataPoint = await self.get_table(collection_name)
            async with self.get_async_session() as session:
                results = None
                if not data_point_ids:
                    results = await session.execute(
                        delete(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(data_point_ids))
                    )
                else:
                    for start_index in range(0, len(data_point_ids), QUERY_BATCH_SIZE):
                        id_batch = data_point_ids[start_index : start_index + QUERY_BATCH_SIZE]
                        results = await session.execute(
                            delete(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(id_batch))
                        )
                await session.commit()
                return results

    async def remove_belongs_to_set_tags(
        self,
        tags: list[str],
        node_ids: list[str] | None = None,
    ) -> None:
        """
        Strip the given tag names from `belongs_to_set` arrays in every
        vector collection and delete rows whose array is now empty. Used to
        reconcile surviving shared rows when a dataset/NodeSet is deleted.

        When `node_ids` is provided, the detag only applies to rows whose
        id is in the list — used to reconcile shared rows that lose a
        dataset's anchor while that dataset still exists for other rows.

        Cognee vector collections follow the `{PascalCaseType}_{field}`
        naming convention (e.g. `Entity_name`, `DocumentChunk_text`) and
        coexist in the same schema as lowercase relational tables — filter
        to collections by requiring an uppercase first character.
        """
        if not tags:
            return

        if node_ids is not None and not node_ids:
            return

        # `get_table_names()` returns the raw SQLAlchemy reflection keys; for
        # Postgres those may be schema-qualified (`schema.table`) when the
        # reflected schema isn't the search-path default, while SQLite and
        # same-schema Postgres return plain table names. Strip any leading
        # `schema.` prefix first, then keep only PascalCase names — cognee's
        # vector collections are `Entity_name` / `DocumentChunk_text` etc.,
        # whereas relational tables (`users`, `nodes`, …) are snake_case.
        def _table_only(name: str) -> str:
            """Strip any leading `schema.` prefix from a reflected table name."""
            return name.rpartition(".")[2] if "." in name else name

        candidate_tables: list[str] = []
        for name in await self.get_table_names():
            if not name:
                continue
            table_only = _table_only(name)
            if table_only and table_only[0].isupper():
                candidate_tables.append(table_only)

        id_scope_clause = "AND id = ANY(:node_ids)" if node_ids is not None else ""
        bind_params: dict[str, Any] = {"tags": list(tags)}
        if node_ids is not None:
            bind_params["node_ids"] = [str(nid) for nid in node_ids]

        first_error: SQLAlchemyError | None = None
        for table_name in candidate_tables:
            quoted_table = self._qualified_table(table_name)
            # Can't use a data-modifying CTE (UPDATE … RETURNING then DELETE
            # against the CTE): Postgres runs both statements under the
            # same snapshot, so the outer DELETE doesn't observe the
            # updated row versions and silently no-ops. Instead, capture
            # the id set up front, run the UPDATE scoped to those ids,
            # then DELETE only the ones that ended up empty. The captured
            # set ensures the DELETE can't drop rows that were already
            # empty before the call.
            select_targets_sql = text(
                f"""
                SELECT id FROM {quoted_table}
                WHERE payload::jsonb ? 'belongs_to_set'
                  {id_scope_clause}
                  AND EXISTS (
                      SELECT 1
                      FROM jsonb_array_elements_text(
                          payload::jsonb->'belongs_to_set'
                      ) v
                      WHERE v = ANY(:tags)
                  )
                """
            )
            update_sql = text(
                f"""
                UPDATE {quoted_table}
                SET payload = jsonb_set(
                    payload::jsonb,
                    '{{belongs_to_set}}',
                    (
                        SELECT COALESCE(jsonb_agg(val), '[]'::jsonb)
                        FROM jsonb_array_elements_text(
                            COALESCE(payload::jsonb->'belongs_to_set', '[]'::jsonb)
                        ) AS val
                        WHERE val <> ALL(:tags)
                    )
                )::json
                WHERE id = ANY(:target_ids)
                """
            )
            delete_empties_sql = text(
                f"""
                DELETE FROM {quoted_table}
                WHERE id = ANY(:target_ids)
                  AND payload::jsonb->'belongs_to_set' = '[]'::jsonb
                """
            )
            # Each table is its own transaction, so one table's failure never
            # undoes or prevents another's. A PascalCase table that is not a
            # vector collection has no `payload::jsonb` column (ProgrammingError)
            # and is skipped. Any other database failure is logged with the
            # table, the remaining tables are still swept, and the first failure
            # is raised afterwards: the caller sees it, and the log names every
            # table that kept the tag.
            try:
                await self._remove_tags_from_table(
                    table_name, select_targets_sql, update_sql, delete_empties_sql, bind_params
                )
            except ProgrammingError as e:
                logger.debug(
                    "remove_belongs_to_set_tags skipped '%s': %s",
                    table_name,
                    e,
                )
            except SQLAlchemyError as e:
                logger.warning(
                    "remove_belongs_to_set_tags failed on '%s', its rows keep the tags: %s",
                    table_name,
                    e,
                )
                if first_error is None:
                    first_error = e

        if first_error is not None:
            raise first_error

    @retry(
        retry=retry_if_exception(is_deadlock),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=6),
        reraise=True,
    )
    async def _remove_tags_from_table(
        self,
        table_name: str,
        select_targets_sql: TextClause,
        update_sql: TextClause,
        delete_empties_sql: TextClause,
        bind_params: dict[str, Any],
    ) -> None:
        """One table's detag in one transaction.

        Inserts do not take the collection lock, so the UPDATE can meet a concurrent
        upsert of the same rows in another order; Postgres aborts one side of that
        deadlock and the sweep is retried from the target selection.
        """
        async with self._get_write_lock(table_name), self.get_async_session() as session:
            target_rows = await session.execute(select_targets_sql, bind_params)
            target_ids = [row[0] for row in target_rows.all()]
            if not target_ids:
                await session.commit()
                return

            scoped_params: dict[str, Any] = {
                "tags": bind_params["tags"],
                "target_ids": target_ids,
            }
            await session.execute(update_sql, scoped_params)
            await session.execute(delete_empties_sql, {"target_ids": target_ids})
            await session.commit()

    async def prune(self):
        """Drop all vector collection tables and reset cached reflection metadata."""
        if not self._owns_engine:
            raise SharedDatabasePruneError(
                "PGVector cannot be pruned independently while it shares the relational "
                "PostgreSQL database. Use prune_system(metadata=True) to delete the shared "
                "database (this also deletes users, datasets and permissions), or set "
                "VECTOR_DB_NAME to a different, dedicated database."
            )

        self._metadata.clear()
        await self.delete_database()

    async def run_migrations(self):
        """Build the HNSW index of any collection that is missing it.

        The runner calls this on every cognee version change. The
        ``pgvector_hnsw_indexes`` migration builds the indexes once, and logs and skips
        a table whose build fails (a ``statement_timeout``, a full disk), so this is
        where such a table is retried. ``IF NOT EXISTS`` makes an indexed table a
        catalog lookup. A failed build is logged and skipped here too: the runner
        blocks writes to the dataset when this raises, and an unindexed collection is
        still searched correctly, by a full scan.
        """
        async with self.engine.connect() as connection:
            collections = (await connection.execute(text(_VECTOR_COLLECTIONS_SQL))).all()
            await self._pgvector_version(connection)  # fails fast on an outdated pgvector

        indexed, failed = [], []
        for table_name, column_type, dimensions in collections:
            statement = hnsw_index_statement(table_name, dimensions, column_type=column_type)
            if statement is None:
                continue
            try:
                async with self.engine.begin() as connection:
                    await connection.execute(
                        text(_RAISE_MAINTENANCE_WORK_MEM_SQL),
                        {
                            "target": f"{_BUILD_MAINTENANCE_WORK_MEM_KB}kB",
                            "target_kb": _BUILD_MAINTENANCE_WORK_MEM_KB,
                        },
                    )
                    await connection.execute(text(_SERIAL_BUILD_SQL))
                    await connection.execute(text(statement))
                indexed.append(table_name)
            except SQLAlchemyError as error:
                failed.append(table_name)
                logger.warning(
                    "Could not build the HNSW index on %s (%s); it stays searchable by a "
                    "full scan and is retried on the next cognee version change: %s",
                    table_name,
                    type(error).__name__,
                    error,
                )
        return {"hnsw_index_checked": indexed, "hnsw_index_failed": failed}
