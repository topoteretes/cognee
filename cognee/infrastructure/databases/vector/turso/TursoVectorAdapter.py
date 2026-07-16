"""Vector-database adapter backed by Turso / libSQL."""

import json
import asyncio
import threading
from typing import Any, List, Optional
from uuid import UUID

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.exceptions import MissingQueryParameterError

from ..models.ScoredResult import ScoredResult
from ..exceptions import CollectionNotFoundError
from ..vector_db_interface import VectorDBInterface
from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..pgvector.serialize_data import serialize_data

logger = get_logger("TursoVectorAdapter")

QUERY_BATCH_SIZE = 1000


class IndexSchema(DataPoint):
    """Schema for the rows written by ``index_data_points`` (mirrors PGVector)."""

    text: str

    # Optional reference scalars carried for the search "Evidence" feature.
    # They stay None for non-chunk data points, so this schema remains
    # compatible with every indexed DataPoint type.
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    chunk_index: Optional[int] = None
    source_chunk_id: Optional[str] = None
    importance_weight: Optional[float] = 0.5

    metadata: dict = {"index_fields": ["text"]}
    belongs_to_set: List[str] = []


def _is_remote_url(url: str) -> bool:
    """True when ``url`` points at a libSQL server rather than a local file."""
    return url.startswith(("libsql://", "http://", "https://", "ws://", "wss://"))


def _vector_literal(vector: List[float]) -> str:
    """Render an embedding as the JSON-array text ``vector32()`` expects."""
    return json.dumps([float(value) for value in vector])


class TursoVectorAdapter(VectorDBInterface):
    """Vector-database adapter backed by Turso / libSQL; implements VectorDBInterface."""

    name = "Turso"

    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        embedding_engine: EmbeddingEngine,
        database_name: Optional[str] = None,
    ):
        self.url = url
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.database_name = database_name

        # One lock serializes access to the shared sync libSQL connection. It
        # is a threading.Lock (not an asyncio.Lock) held inside the
        # asyncio.to_thread worker: this adapter is cached process-globally, so
        # a loop-bound asyncio.Lock would raise "bound to a different event
        # loop" the moment a second event loop (e.g. a later asyncio.run)
        # contends it. A threading.Lock is loop-agnostic — the same reason
        # LanceDBAdapter uses one for its lifecycle state.
        self._connection_lock = threading.Lock()

        # Reflected collection names; refreshed lazily by has_collection().
        self._known_collections: set[str] = set()
        self._connection = None

    # ------------------------------------------------------------------ #
    # Connection + low-level execution.
    #
    # libsql-experimental is sync but is the only client with native vector
    # support in embedded mode, so DB calls run via asyncio.to_thread. The sync
    # client is touched only here (and in _run / _run_many), keeping the async
    # contract easy to re-align to a native-async client later.
    # ------------------------------------------------------------------ #
    def _get_connection(self):
        """Lazily open the libSQL connection (embedded file or remote server)."""
        if self._connection is not None:
            return self._connection

        import libsql_experimental as libsql

        if _is_remote_url(self.url):
            self._connection = libsql.connect(
                database=self.url,
                auth_token=self.api_key or "",
                check_same_thread=False,
            )
        else:
            self._connection = libsql.connect(self.url, check_same_thread=False)

        return self._connection

    def _run(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        *,
        fetch: bool = False,
        commit: bool = False,
    ):
        """Run one statement synchronously. Called only inside asyncio.to_thread."""
        with self._connection_lock:
            connection = self._get_connection()
            cursor = connection.execute(sql, tuple(params) if params else ())
            rows = cursor.fetchall() if fetch else None
            if commit:
                connection.commit()
            return rows

    async def _execute(
        self,
        sql: str,
        params: Optional[List[Any]] = None,
        *,
        fetch: bool = False,
        commit: bool = False,
    ):
        return await asyncio.to_thread(self._run, sql, params, fetch=fetch, commit=commit)

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #
    async def embed_data(self, data: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors using the configured engine."""
        return await self.embedding_engine.embed_text(data)

    # ------------------------------------------------------------------ #
    # Collections
    # ------------------------------------------------------------------ #
    async def has_collection(self, collection_name: str) -> bool:
        """Return True when a table named ``collection_name`` exists."""
        if collection_name in self._known_collections:
            return True

        rows = await self._execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            [collection_name],
            fetch=True,
        )
        exists = bool(rows)
        if exists:
            self._known_collections.add(collection_name)
        return exists

    async def create_collection(self, collection_name: str, payload_schema=None):
        """Create the libSQL table for ``collection_name`` if it does not exist."""
        vector_size = self.embedding_engine.get_vector_size()

        if not await self.has_collection(collection_name):
            await self._execute(
                f'CREATE TABLE IF NOT EXISTS "{collection_name}" '
                f"(id TEXT PRIMARY KEY, payload TEXT, vector F32_BLOB({vector_size}))",
                commit=True,
            )
            self._known_collections.add(collection_name)

    async def get_table_names(self) -> List[str]:
        """Return every table name in the database (used by prune / detag / tests)."""
        rows = await self._execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
            fetch=True,
        )
        return [row[0] for row in rows] if rows else []

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        """Upsert DataPoints, merging ``belongs_to_set`` on id conflict."""
        if not data_points:
            return

        if not await self.has_collection(collection_name):
            await self.create_collection(collection_name, payload_schema=type(data_points[0]))

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        # One INSERT ... ON CONFLICT(id) DO UPDATE per row, all in one
        # transaction. libSQL reads its own uncommitted writes, so duplicate ids
        # within a single batch merge in SQL exactly like cross-call upserts do:
        # the last row's payload wins, the stored and incoming belongs_to_set
        # arrays are unioned (keeping the existing vector), so a tag present on
        # only one duplicate is never dropped. (PGVector needs a separate
        # in-Python dedup pass only because it sends the whole batch as a single
        # multi-row VALUES statement, which cannot touch the same id twice.)
        insert_sql = (
            f'INSERT INTO "{collection_name}" (id, payload, vector) '
            f"VALUES (?, ?, vector32(?)) "
            f"ON CONFLICT(id) DO UPDATE SET payload = json_set("
            f"  excluded.payload, '$.belongs_to_set',"
            f"  (SELECT json_group_array(value) FROM ("
            f'    SELECT value FROM json_each(json_extract("{collection_name}".payload, '
            f"'$.belongs_to_set'))"
            f"    UNION"
            f"    SELECT value FROM json_each(json_extract(excluded.payload, '$.belongs_to_set'))"
            f"  ))"
            f")"
        )

        params = [
            [
                str(data_point.id),
                json.dumps(serialize_data(data_point.model_dump())),
                _vector_literal(data_vectors[index]),
            ]
            for index, data_point in enumerate(data_points)
        ]

        await asyncio.to_thread(self._run_many, insert_sql, params)

    def _run_many(self, sql: str, params: List[List[Any]]):
        """Execute one write per row inside a single committed transaction."""
        with self._connection_lock:
            connection = self._get_connection()
            for row in params:
                connection.execute(sql, tuple(row))
            connection.commit()

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """Create the index collection (table) for the given name/property pair."""
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: List[DataPoint]
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
    async def retrieve(self, collection_name: str, data_point_ids: List[str]):
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

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: Optional[int] = 15,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
    ) -> List[ScoredResult]:
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

        params: List[Any] = [_vector_literal(query_vector)]
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
        query_texts: List[str],
        limit: int = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
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
    async def delete_data_points(self, collection_name: str, data_point_ids: List[UUID]):
        """Delete rows whose id is in ``data_point_ids``."""
        if not await self.has_collection(collection_name):
            return None

        ids = [str(data_point_id) for data_point_id in data_point_ids]
        if not ids:
            return None

        for start in range(0, len(ids), QUERY_BATCH_SIZE):
            id_batch = ids[start : start + QUERY_BATCH_SIZE]
            placeholders = ",".join("?" for _ in id_batch)
            await self._execute(
                f'DELETE FROM "{collection_name}" WHERE id IN ({placeholders})',
                id_batch,
                commit=True,
            )
        return None

    async def remove_belongs_to_set_tags(
        self,
        tags: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> None:
        """Strip ``tags`` from belongs_to_set arrays and delete rows left empty.

        cognee vector collections follow the ``{PascalCaseType}_{field}``
        naming convention and coexist with snake_case relational tables, so
        only PascalCase-named tables are touched.
        """
        if not tags:
            return None
        if node_ids is not None and not node_ids:
            return None

        candidate_tables = [
            name for name in await self.get_table_names() if name and name[0].isupper()
        ]

        tags_json = json.dumps(list(tags))
        node_ids_list = [str(node_id) for node_id in node_ids] if node_ids is not None else None

        for table_name in candidate_tables:
            id_scope = ""
            scope_params: List[Any] = []
            if node_ids_list is not None:
                placeholders = ",".join("?" for _ in node_ids_list)
                id_scope = f" AND id IN ({placeholders})"
                scope_params = list(node_ids_list)

            # Capture the rows that actually contain one of the removed tags
            # FIRST. The UPDATE + delete-when-empty must only touch these rows,
            # otherwise a row that was already stored with an empty
            # belongs_to_set (e.g. an untagged index row) would be deleted as
            # collateral on any unrelated tag removal. Mirrors PGVector.
            select_sql = (
                f'SELECT id FROM "{table_name}" '
                f"WHERE json_type(payload, '$.belongs_to_set') = 'array' "
                f"AND EXISTS (SELECT 1 FROM json_each(payload, '$.belongs_to_set') je "
                f"WHERE je.value IN (SELECT value FROM json_each(?))){id_scope}"
            )
            # The SELECT doubles as the "is this a vector collection?" probe: a
            # PascalCase relational table without a JSON belongs_to_set payload
            # errors here and is skipped quietly.
            try:
                rows = await self._execute(select_sql, [tags_json] + scope_params, fetch=True)
            except Exception as error:  # noqa: BLE001 - not a vector collection; skip
                logger.debug("remove_belongs_to_set_tags skipped '%s': %s", table_name, error)
                continue

            target_ids = [row[0] for row in rows or []]
            if not target_ids:
                continue

            id_placeholders = ",".join("?" for _ in target_ids)
            # Strip the tags from exactly those rows.
            update_sql = (
                f"UPDATE \"{table_name}\" SET payload = json_set(payload, '$.belongs_to_set', ("
                f"  SELECT json_group_array(value) FROM json_each(payload, '$.belongs_to_set')"
                f"  WHERE value NOT IN (SELECT value FROM json_each(?))"
                f")) WHERE id IN ({id_placeholders})"
            )
            # Delete only the captured rows that are now empty.
            delete_sql = (
                f'DELETE FROM "{table_name}" WHERE id IN ({id_placeholders}) '
                f"AND json_array_length(payload, '$.belongs_to_set') = 0"
            )
            # A write failure once we know the table is a real collection is a
            # genuine error: surface it at warning rather than hiding it at
            # debug, but keep going so one table can't abort the rest.
            try:
                await self._execute(update_sql, [tags_json] + target_ids, commit=True)
                await self._execute(delete_sql, target_ids, commit=True)
            except Exception as error:  # noqa: BLE001 - surface, but continue other tables
                logger.warning(
                    "remove_belongs_to_set_tags failed to update '%s': %s", table_name, error
                )

        return None

    async def prune(self):
        """Drop every collection table and reset cached reflection state."""
        for table_name in await self.get_table_names():
            await self._execute(f'DROP TABLE IF EXISTS "{table_name}"', commit=True)
        self._known_collections.clear()

    async def run_migrations(self):
        """Run Turso adapter migrations (currently no-op)."""
        return None

    def reset_metadata_cache(self):
        """Reset cached collection names for this adapter instance."""
        self._known_collections.clear()

    async def close(self) -> None:
        """Close the libSQL connection. Driven by closing_lru_cache on eviction."""
        await asyncio.to_thread(self._close)

    def _close(self) -> None:
        """Close the connection under the lock so it can't race an in-flight _run."""
        with self._connection_lock:
            if self._connection is not None:
                connection, self._connection = self._connection, None
                connection.close()
