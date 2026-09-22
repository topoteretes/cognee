"""Vector-database adapter backed by Turso, the Rust rewrite of SQLite (``pyturso``).

One table per collection: ``(id TEXT PRIMARY KEY, payload TEXT, vector F32_BLOB(n))``.
Similarity search is an exact ``vector_distance_cos`` scan ordered by distance; the
engine has no ``libsql_vector_idx`` / ``vector_top_k`` approximate index.

Engine constraints that shape the SQL here: no scalar subquery inside
``ON CONFLICT DO UPDATE SET`` (so ``belongs_to_set`` merges happen in Python before
a plain upsert), and no bind parameter inside a nested ``json_each`` subquery (so
tag removal rewrites payloads in Python and writes them back with plain binds).

Concurrency: one synchronous driver connection per adapter, used only inside
``asyncio.to_thread`` under ``self._connection_lock``. That lock is load-bearing —
a pyturso connection used from two threads at once aborts the process. Under
``TURSO_JOURNAL_MODE=mvcc`` writes run as ``BEGIN CONCURRENT`` and retry on
``Write-write conflict``.
"""

import asyncio
import json
import threading
from typing import Any
from uuid import UUID

from cognee.infrastructure.databases.exceptions import MissingQueryParameterError
from cognee.infrastructure.databases.turso import (
    begin_statement,
    connect_pragmas,
    get_turso_config,
    retry_on_conflict,
)
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.shared.logging_utils import get_logger

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..exceptions import CollectionNotFoundError
from ..models.ScoredResult import ScoredResult
from ..pgvector.serialize_data import serialize_data
from ..vector_db_interface import VectorDBInterface

logger = get_logger("TursoVectorAdapter")

QUERY_BATCH_SIZE = 1000


class IndexSchema(DataPoint):
    """Schema for the rows written by ``index_data_points`` (mirrors PGVector)."""

    text: str

    # Optional reference scalars carried for the search "Evidence" feature.
    # They stay None for non-chunk data points, so this schema remains
    # compatible with every indexed DataPoint type.
    document_id: str | None = None
    document_name: str | None = None
    chunk_index: int | None = None
    source_chunk_id: str | None = None
    importance_weight: float | None = 0.5

    metadata: dict = {"index_fields": ["text"]}
    belongs_to_set: list[str] = []


def _is_remote_url(url: str) -> bool:
    """True when ``url`` points at a Turso server rather than a local file."""
    return url.startswith(("libsql://", "http://", "https://", "ws://", "wss://"))


def _union_tags(*tag_lists) -> list[str]:
    """Order-preserving union of ``belongs_to_set`` lists (None-safe)."""
    merged: dict[str, None] = {}
    for tags in tag_lists:
        for tag in tags or []:
            merged.setdefault(tag, None)
    return list(merged)


def _vector_literal(vector: list[float]) -> str:
    """Render an embedding as the JSON-array text ``vector32()`` expects."""
    return json.dumps([float(value) for value in vector])


class TursoVectorAdapter(VectorDBInterface):
    """Vector-database adapter backed by the Turso rewrite engine; implements VectorDBInterface."""

    name = "Turso"

    def __init__(
        self,
        url: str,
        api_key: str | None,
        embedding_engine: EmbeddingEngine,
        database_name: str | None = None,
    ):
        if _is_remote_url(url):
            raise OSError(
                "Remote Turso databases are not supported by the Turso vector backend in this "
                f"version (VECTOR_DB_URL={url!r}). Point VECTOR_DB_URL at a local database "
                "file path instead."
            )
        self.url = url
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.database_name = database_name
        self.turso_config = get_turso_config()

        # One lock serializes access to the shared sync Turso connection. It
        # is a threading.Lock (not an asyncio.Lock) held inside the
        # asyncio.to_thread worker: this adapter is cached process-globally, so
        # a loop-bound asyncio.Lock would raise "bound to a different event
        # loop" the moment a second event loop (e.g. a later asyncio.run)
        # contends it. A threading.Lock is loop-agnostic — the same reason
        # LanceDBAdapter uses one for its lifecycle state. It is also what keeps
        # the process alive: pyturso aborts on truly concurrent use of one
        # connection, so every driver call below runs with this lock held.
        self._connection_lock = threading.Lock()

        # Reflected collection names; refreshed lazily by has_collection().
        self._known_collections: set[str] = set()
        self._connection = None

    # ------------------------------------------------------------------ #
    # Connection + low-level execution.
    #
    # The sync pyturso DB-API connection runs via asyncio.to_thread. It is
    # touched only here (and in the _run* helpers), keeping the async contract
    # easy to re-align to turso.aio later.
    # ------------------------------------------------------------------ #
    def _get_connection(self):
        """Lazily open the Turso connection to the local database file."""
        if self._connection is not None:
            return self._connection

        import turso

        config = self.turso_config
        # mvcc: driver autocommit so _run controls BEGIN CONCURRENT / COMMIT itself.
        connect_kwargs = {"isolation_level": None} if config.concurrent_writes else {}
        connection = turso.connect(self.url, **connect_kwargs)
        for statement in connect_pragmas(config):
            # Step the PRAGMA: pyturso runs a statement when its cursor is read.
            connection.execute(statement).fetchall()
        self._connection = connection
        return self._connection

    def _run(
        self,
        sql: str,
        params: list[Any] | None = None,
        *,
        fetch: bool = False,
        commit: bool = False,
        ddl: bool = False,
    ):
        """Run one statement synchronously. Called only inside asyncio.to_thread."""
        with self._connection_lock:
            connection = self._get_connection()
            begin = begin_statement(self.turso_config, ddl=ddl) if commit else None
            if begin:
                connection.execute(begin)
            try:
                cursor = connection.execute(sql, tuple(params) if params else ())
                rows = cursor.fetchall() if fetch else None
                if commit:
                    self._commit(connection, begin)
                return rows
            except Exception:
                if begin:
                    self._rollback(connection)
                raise

    def _commit(self, connection, begin: str | None) -> None:
        # With an explicit BEGIN (mvcc) the connection is in driver autocommit and
        # COMMIT must be a statement; otherwise the driver's own transaction ends
        # with commit().
        if begin:
            connection.execute("COMMIT")
        else:
            connection.commit()

    @staticmethod
    def _rollback(connection) -> None:
        # A write conflict aborts the MVCC transaction on the engine side already;
        # the ROLLBACK then reports "no transaction is active", which is fine.
        try:
            connection.execute("ROLLBACK")
        except Exception:  # nothing to recover; the caller re-raises the cause
            logger.debug("Turso rollback after a failed write", exc_info=True)

    def _run_write(self, statements: list[tuple[str, tuple]]) -> None:
        """Execute ``statements`` inside one committed transaction (sync, locked)."""
        with self._connection_lock:
            connection = self._get_connection()
            begin = begin_statement(self.turso_config)
            if begin:
                connection.execute(begin)
            try:
                for sql, params in statements:
                    connection.execute(sql, params)
                self._commit(connection, begin)
            except Exception:
                self._rollback(connection)
                raise

    async def _execute(
        self,
        sql: str,
        params: list[Any] | None = None,
        *,
        fetch: bool = False,
        commit: bool = False,
        ddl: bool = False,
    ):
        def run():
            return asyncio.to_thread(self._run, sql, params, fetch=fetch, commit=commit, ddl=ddl)

        return await retry_on_conflict(run) if commit else await run()

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #
    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors using the configured engine."""
        return await self.embedding_engine.embed_text(data)

    # ------------------------------------------------------------------ #
    # Collections
    # ------------------------------------------------------------------ #
    async def has_collection(self, collection_name: str) -> bool:
        """Return True when a table named ``collection_name`` exists."""
        if collection_name in self._known_collections:
            return True

        # The engine stores quoted identifiers lowercased in sqlite_master (stock
        # SQLite keeps their case), so compare case-insensitively; name resolution
        # in FROM clauses is case-insensitive either way.
        rows = await self._execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND lower(name) = lower(?)",
            [collection_name],
            fetch=True,
        )
        exists = bool(rows)
        if exists:
            self._known_collections.add(collection_name)
        return exists

    async def create_collection(self, collection_name: str, payload_schema=None):
        """Create the table for ``collection_name`` if it does not exist."""
        vector_size = self.embedding_engine.get_vector_size()

        if not await self.has_collection(collection_name):
            await self._execute(
                f'CREATE TABLE IF NOT EXISTS "{collection_name}" '
                f"(id TEXT PRIMARY KEY, payload TEXT, vector F32_BLOB({vector_size}))",
                commit=True,
                ddl=True,
            )
            self._known_collections.add(collection_name)

    async def _is_collection(self, table_name: str) -> bool:
        """True when ``table_name`` has the vector-collection columns."""
        rows = await self._execute(f'PRAGMA table_info("{table_name}")', fetch=True)
        columns = {row[1] for row in rows or []}
        return {"id", "payload", "vector"} <= columns

    async def get_table_names(self) -> list[str]:
        """Return every table name in the database (used by prune / detag / tests)."""
        rows = await self._execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
            fetch=True,
        )
        return [row[0] for row in rows] if rows else []

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    async def create_data_points(self, collection_name: str, data_points: list[DataPoint]):
        """Upsert DataPoints, merging ``belongs_to_set`` on id conflict."""
        if not data_points:
            return

        if not await self.has_collection(collection_name):
            await self.create_collection(collection_name, payload_schema=type(data_points[0]))

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        # Same id twice in one batch: the last payload wins and the tags are
        # unioned, so a tag present on only one duplicate is never dropped
        # (PGVector does this dedup in Python too).
        rows: dict[str, dict[str, Any]] = {}
        for index, data_point in enumerate(data_points):
            row_id = str(data_point.id)
            payload = serialize_data(data_point.model_dump())
            previous = rows.get(row_id)
            if previous is not None:
                payload["belongs_to_set"] = _union_tags(
                    previous["payload"].get("belongs_to_set"), payload.get("belongs_to_set")
                )
            rows[row_id] = {"payload": payload, "vector": _vector_literal(data_vectors[index])}

        await retry_on_conflict(lambda: asyncio.to_thread(self._upsert_rows, collection_name, rows))

    def _upsert_rows(self, collection_name: str, rows: dict[str, dict[str, Any]]) -> None:
        """Upsert ``rows`` in one transaction, merging ``belongs_to_set`` with stored rows.

        The engine rejects a scalar subquery inside ``ON CONFLICT DO UPDATE SET``, so
        the stored tag arrays are read first and unioned here; the upsert itself
        then only assigns ``excluded.payload``. As in PGVector, a conflicting row
        keeps its stored vector.
        """
        with self._connection_lock:
            connection = self._get_connection()
            begin = begin_statement(self.turso_config)
            if begin:
                connection.execute(begin)
            try:
                existing = connection.execute(
                    f"SELECT id, json_extract(payload, '$.belongs_to_set') FROM \"{collection_name}\" "
                    f"WHERE id IN (SELECT value FROM json_each(?))",
                    (json.dumps(list(rows)),),
                ).fetchall()
                for row_id, stored_tags in existing:
                    payload = rows[row_id]["payload"]
                    payload["belongs_to_set"] = _union_tags(
                        json.loads(stored_tags) if stored_tags else None,
                        payload.get("belongs_to_set"),
                    )
                insert_sql = (
                    f'INSERT INTO "{collection_name}" (id, payload, vector) '
                    f"VALUES (?, ?, vector32(?)) "
                    f"ON CONFLICT(id) DO UPDATE SET payload = excluded.payload"
                )
                for row_id, row in rows.items():
                    connection.execute(
                        insert_sql, (row_id, json.dumps(row["payload"]), row["vector"])
                    )
                self._commit(connection, begin)
            except Exception:
                self._rollback(connection)
                raise

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """Create the index collection (table) for the given name/property pair."""
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        """Write index rows derived from ``data_points`` into the {index}_{property} table."""
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=data_point.id,
                    text=DataPoint.get_embeddable_data(data_point),
                    document_id=getattr(data_point, "document_id", None),
                    document_name=getattr(data_point, "document_name", None),
                    chunk_index=getattr(data_point, "chunk_index", None),
                    source_chunk_id=getattr(data_point, "source_chunk_id", None),
                    importance_weight=getattr(data_point, "importance_weight", None),
                    belongs_to_set=(data_point.belongs_to_set or []),
                )
                for data_point in data_points
            ],
        )

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    supports_payload_update = True

    async def update_payload(self, collection_name: str, payload_updates: dict) -> None:
        """Update payload fields on existing rows WITHOUT re-embedding.

        Read-modify-write on the JSON payload text column only — the vector
        blob is never touched, so no embedding call happens. Missing ids skip.
        """
        if not payload_updates:
            return
        if not await self.has_collection(collection_name):
            return
        for data_point_id, fields in payload_updates.items():
            rows = await self._execute(
                f'SELECT payload FROM "{collection_name}" WHERE id = ?',
                [str(data_point_id)],
                fetch=True,
            )
            if not rows:
                continue
            payload = json.loads(rows[0][0]) if rows[0][0] else {}
            # Caller contract: the fields already exist in the payload (see
            # PGVectorAdapter.update_payload).
            unknown_fields = set(fields) - set(payload)
            if unknown_fields:
                raise ValueError(
                    f"update_payload: fields {sorted(unknown_fields)} do not exist in the "
                    f"payload of {collection_name!r} row {data_point_id}"
                )
            payload.update(fields)
            await self._execute(
                f'UPDATE "{collection_name}" SET payload = ? WHERE id = ?',
                [json.dumps(payload), str(data_point_id)],
                commit=True,
            )

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        """Return rows from ``collection_name`` matching any of ``data_point_ids``."""
        if not await self.has_collection(collection_name):
            return []

        results = []
        seen_ids = set()
        ids = [str(data_point_id) for data_point_id in data_point_ids]
        for start in range(0, len(ids), QUERY_BATCH_SIZE):
            id_batch = ids[start : start + QUERY_BATCH_SIZE]
            placeholders = ",".join("?" for _ in id_batch)
            rows = await self._execute(
                f'SELECT id, payload FROM "{collection_name}" WHERE id IN ({placeholders})',
                id_batch,
                fetch=True,
            )
            for row in rows or []:
                if row[0] in seen_ids:
                    continue
                seen_ids.add(row[0])
                results.append(
                    ScoredResult(
                        id=parse_id(row[0]),
                        payload=json.loads(row[1]) if row[1] else {},
                        score=0,
                    )
                )
        return results

    async def score_by_ids(
        self, collection_name: str, data_point_ids: list[str], query_vector: list[float]
    ) -> list[ScoredResult]:
        ids = list(dict.fromkeys(str(point_id) for point_id in data_point_ids))
        if not ids:
            return []
        if not await self.has_collection(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found!")
        scores = []
        for start in range(0, len(ids), QUERY_BATCH_SIZE):
            batch = ids[start : start + QUERY_BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            rows = await self._execute(
                f"SELECT id, vector_distance_cos(vector, vector32(?)) AS distance "
                f'FROM "{collection_name}" WHERE id IN ({placeholders})',
                [_vector_literal(query_vector), *batch],
                fetch=True,
            )
            scores.extend(
                ScoredResult(id=parse_id(str(row[0])), score=float(row[1]), payload=None)
                for row in rows or []
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

        if not await self.has_collection(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found!")

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        if limit is None:
            rows = await self._execute(f'SELECT count(*) FROM "{collection_name}"', fetch=True)
            limit = rows[0][0] if rows else 0

        if limit <= 0:
            return []

        params: list[Any] = [_vector_literal(query_vector)]
        where_clause = ""
        if node_name:
            placeholders = ",".join("?" for _ in node_name)
            if node_name_filter_operator == "AND":
                where_clause = (
                    f" WHERE (SELECT count(DISTINCT je.value) FROM json_each("
                    f"\"{collection_name}\".payload, '$.belongs_to_set') je "
                    f"WHERE je.value IN ({placeholders})) = ?"
                )
                params.extend(node_name)
                params.append(len(set(node_name)))
            else:
                where_clause = (
                    f" WHERE EXISTS (SELECT 1 FROM json_each("
                    f"\"{collection_name}\".payload, '$.belongs_to_set') je "
                    f"WHERE je.value IN ({placeholders}))"
                )
                params.extend(node_name)

        params.append(limit)
        # Skip the payload column unless the caller needs it (mirrors PGVector):
        # the graph/RAG hot path wants only id + distance, and payloads are large
        # chunk JSON. Bind order matches the statement: SELECT's vector32(?), then
        # any NodeSet placeholders (+ the AND count), then LIMIT.
        payload_column = "payload" if include_payload else "NULL"
        rows = await self._execute(
            f"SELECT id, {payload_column}, "
            f"vector_distance_cos(vector, vector32(?)) AS _distance "
            f'FROM "{collection_name}"{where_clause} ORDER BY _distance ASC LIMIT ?',
            params,
            fetch=True,
        )

        return [
            ScoredResult(
                id=parse_id(str(row[0])),
                payload=json.loads(row[1]) if (include_payload and row[1]) else None,
                score=float(row[2]),
            )
            for row in rows or []
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
        """Run ``search`` for each query text and return a list of result lists."""
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

    # ------------------------------------------------------------------ #
    # Deletes
    # ------------------------------------------------------------------ #
    async def delete_data_points(self, collection_name: str, data_point_ids: list[UUID]):
        """Delete rows whose id is in ``data_point_ids``."""
        if not await self.has_collection(collection_name):
            return

        ids = [str(data_point_id) for data_point_id in data_point_ids]
        if not ids:
            return

        for start in range(0, len(ids), QUERY_BATCH_SIZE):
            id_batch = ids[start : start + QUERY_BATCH_SIZE]
            placeholders = ",".join("?" for _ in id_batch)
            await self._execute(
                f'DELETE FROM "{collection_name}" WHERE id IN ({placeholders})',
                id_batch,
                commit=True,
            )
        return

    async def remove_belongs_to_set_tags(
        self,
        tags: list[str],
        node_ids: list[str] | None = None,
    ) -> None:
        """Strip ``tags`` from belongs_to_set arrays and delete rows left empty.

        Only tables with the collection schema (``id, payload, vector``) are
        touched. The engine reports table names lowercased in ``sqlite_master``,
        so the PascalCase naming convention cannot be used to tell collections
        from relational tables the way the PGVector adapter does.
        """
        if not tags:
            return
        if node_ids is not None and not node_ids:
            return

        candidate_tables = [
            name for name in await self.get_table_names() if await self._is_collection(name)
        ]

        tags_json = json.dumps(list(tags))
        node_ids_list = [str(node_id) for node_id in node_ids] if node_ids is not None else None

        for table_name in candidate_tables:
            id_scope = ""
            scope_params: list[Any] = []
            if node_ids_list is not None:
                placeholders = ",".join("?" for _ in node_ids_list)
                id_scope = f" AND id IN ({placeholders})"
                scope_params = list(node_ids_list)

            # Capture the rows that actually contain one of the removed tags
            # FIRST. Only these rows are rewritten or deleted-when-empty,
            # otherwise a row that was already stored with an empty
            # belongs_to_set (e.g. an untagged index row) would be deleted as
            # collateral on any unrelated tag removal. Mirrors PGVector.
            select_sql = (
                f'SELECT id, payload FROM "{table_name}" '
                f"WHERE json_type(payload, '$.belongs_to_set') = 'array' "
                f"AND EXISTS (SELECT 1 FROM json_each(payload, '$.belongs_to_set') je "
                f"WHERE je.value IN (SELECT value FROM json_each(?))){id_scope}"
            )
            # The SELECT doubles as the "is this a vector collection?" probe: a
            # PascalCase relational table without a JSON belongs_to_set payload
            # errors here and is skipped quietly.
            try:
                rows = await self._execute(select_sql, [tags_json] + scope_params, fetch=True)
            except Exception as error:  # not a vector collection; skip
                logger.debug(
                    "remove_belongs_to_set_tags skipped '%s': %s", table_name, error, exc_info=True
                )
                continue

            if not rows:
                continue

            # The engine cannot bind a parameter inside the nested json_each()
            # a SQL-side rewrite needs, so filter the arrays here and write the
            # payloads back with plain binds: UPDATE the survivors, DELETE the
            # rows whose array became empty, all in one transaction.
            tag_set = set(tags)
            statements: list[tuple[str, tuple]] = []
            for row_id, payload_text in rows:
                payload = json.loads(payload_text) if payload_text else {}
                remaining = [tag for tag in payload.get("belongs_to_set", []) if tag not in tag_set]
                if remaining:
                    payload["belongs_to_set"] = remaining
                    statements.append(
                        (
                            f'UPDATE "{table_name}" SET payload = ? WHERE id = ?',
                            (json.dumps(payload), row_id),
                        )
                    )
                else:
                    statements.append((f'DELETE FROM "{table_name}" WHERE id = ?', (row_id,)))
            # A write failure once we know the table is a real collection is a
            # genuine error: surface it at warning rather than hiding it at
            # debug, but keep going so one table can't abort the rest.
            try:
                await retry_on_conflict(
                    lambda statements=statements: asyncio.to_thread(self._run_write, statements)
                )
            except Exception as error:  # surface, but continue other tables
                logger.warning(
                    "remove_belongs_to_set_tags failed to update '%s': %s",
                    table_name,
                    error,
                    exc_info=True,
                )

        return

    async def prune(self):
        """Drop every collection table and reset cached reflection state."""
        for table_name in await self.get_table_names():
            await self._execute(f'DROP TABLE IF EXISTS "{table_name}"', commit=True, ddl=True)
        self._known_collections.clear()

    async def run_migrations(self):
        """Run Turso adapter migrations (currently no-op)."""
        return

    def reset_metadata_cache(self):
        """Reset cached collection names for this adapter instance."""
        self._known_collections.clear()

    async def close(self) -> None:
        """Close the Turso connection. Driven by closing_lru_cache on eviction."""
        await asyncio.to_thread(self._close)

    def _close(self) -> None:
        """Close the connection under the lock so it can't race an in-flight _run."""
        with self._connection_lock:
            if self._connection is not None:
                connection, self._connection = self._connection, None
                connection.close()
