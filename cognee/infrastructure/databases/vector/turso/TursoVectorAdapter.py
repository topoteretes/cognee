import json
import asyncio
from typing import Any, List, Optional
from uuid import UUID

from sqlalchemy import text

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.exceptions import MissingQueryParameterError

from ...relational.sqlalchemy.SqlAlchemyAdapter import SQLAlchemyAdapter
from ..models.ScoredResult import ScoredResult
from ..exceptions import CollectionNotFoundError
from ..vector_db_interface import VectorDBInterface
from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..pgvector.serialize_data import serialize_data

logger = get_logger("TursoVectorAdapter")
QUERY_BATCH_SIZE = 1000


class IndexSchema(DataPoint):
    """Schema for vector index data points, mirroring PGVectorAdapter's IndexSchema."""

    text: str

    # Optional reference scalars for search "Evidence" feature.
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    chunk_index: Optional[int] = None
    source_chunk_id: Optional[str] = None
    importance_weight: Optional[float] = 0.5

    metadata: dict = {"index_fields": ["text"]}
    belongs_to_set: List[str] = []


class TursoVectorAdapter(SQLAlchemyAdapter, VectorDBInterface):
    """Vector-database adapter backed by Turso/libSQL; implements VectorDBInterface.

    Uses libSQL's native F32_BLOB vector type and vector_distance_cos() for
    similarity search. Extends SQLAlchemyAdapter to inherit async engine and
    session management, mirroring PGVectorAdapter's architecture.
    """

    name = "Turso"

    def __init__(
        self,
        connection_string: str,
        api_key: Optional[str],
        embedding_engine: EmbeddingEngine,
    ):
        """Initialize the Turso vector adapter.

        Parameters
        ----------
        connection_string : str
            libSQL connection string. Either:
            - Remote: ``libsql://host?authToken=xxx&secure=true``
            - Local: ``sqlite+libsql:///path/to/db``
        api_key : str, optional
            API key / auth token (stored for compatibility).
        embedding_engine : EmbeddingEngine
            Engine used to convert text to vector embeddings.
        """
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.db_uri: str = connection_string
        self.VECTOR_DB_LOCK = asyncio.Lock()
        self._write_locks: dict[str, asyncio.Lock] = {}

        # Initialize the SQLAlchemy async engine via parent class
        super().__init__(connection_string)

    async def close(self) -> None:
        """Release connection-pool resources held by this adapter."""
        await self.engine.dispose(close=True)

    def _get_write_lock(self, collection_name: str) -> asyncio.Lock:
        if collection_name not in self._write_locks:
            self._write_locks[collection_name] = asyncio.Lock()
        return self._write_locks[collection_name]

    # ── Embedding ────────────────────────────────────────────────────────

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors using the configured embedding engine."""
        return await self.embedding_engine.embed_text(data)

    # ── Collection management ────────────────────────────────────────────

    async def has_collection(self, collection_name: str) -> bool:
        """Check if a collection (table) exists in the database."""
        async with self.get_async_session() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type='table' AND name=:name"
                ),
                {"name": collection_name},
            )
            count = result.scalar()
            return count > 0

    async def create_collection(
        self,
        collection_name: str,
        payload_schema: Optional[Any] = None,
    ):
        """Create a vector collection table using libSQL's F32_BLOB type.

        Schema: (id TEXT PRIMARY KEY, payload JSON, vector F32_BLOB(N))
        """
        if await self.has_collection(collection_name):
            return

        vector_size = self.embedding_engine.get_vector_size()

        async with self.VECTOR_DB_LOCK:
            if await self.has_collection(collection_name):
                return

            # Use double-quoting for the table name to handle special chars
            ddl = text(
                f'CREATE TABLE IF NOT EXISTS "{collection_name}" '
                f"(id TEXT PRIMARY KEY, payload JSON, vector F32_BLOB({vector_size}))"
            )

            async with self.get_async_session() as session:
                await session.execute(ddl)
                await session.commit()

    # ── Data point operations ────────────────────────────────────────────

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        """Upsert DataPoints into the collection, merging belongs_to_set on conflict."""
        if not await self.has_collection(collection_name):
            await self.create_collection(
                collection_name=collection_name,
                payload_schema=type(data_points[0]),
            )

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        # Build rows: (id, payload_json, vector_json)
        rows = []
        for i, data_point in enumerate(data_points):
            payload = serialize_data(data_point.model_dump())
            rows.append(
                {
                    "id": str(data_point.id),
                    "payload": json.dumps(payload),
                    "vector": json.dumps(data_vectors[i]),
                }
            )

        # Deduplicate by id within the batch, merging belongs_to_set arrays
        deduped: dict[str, dict] = {}
        for row in rows:
            existing = deduped.get(row["id"])
            if existing is None:
                deduped[row["id"]] = row
                continue
            # Merge belongs_to_set tags
            existing_payload = json.loads(existing["payload"])
            incoming_payload = json.loads(row["payload"])
            existing_tags = existing_payload.get("belongs_to_set") or []
            incoming_tags = incoming_payload.get("belongs_to_set") or []
            if existing_tags or incoming_tags:
                merged_tags = list(dict.fromkeys(list(existing_tags) + list(incoming_tags)))
                incoming_payload["belongs_to_set"] = merged_tags
                row["payload"] = json.dumps(incoming_payload)
            deduped[row["id"]] = row

        unique_rows = list(deduped.values())

        async with self._get_write_lock(collection_name):
            async with self.get_async_session() as session:
                for start in range(0, len(unique_rows), QUERY_BATCH_SIZE):
                    batch = unique_rows[start : start + QUERY_BATCH_SIZE]
                    for row in batch:
                        # Check if the row already exists to merge belongs_to_set
                        existing = await session.execute(
                            text(f'SELECT payload FROM "{collection_name}" WHERE id = :id'),
                            {"id": row["id"]},
                        )
                        existing_row = existing.fetchone()

                        if existing_row and existing_row[0]:
                            # Merge belongs_to_set from existing row
                            try:
                                old_payload = (
                                    json.loads(existing_row[0])
                                    if isinstance(existing_row[0], str)
                                    else existing_row[0]
                                )
                            except (json.JSONDecodeError, TypeError):
                                old_payload = {}

                            new_payload = json.loads(row["payload"])
                            old_tags = old_payload.get("belongs_to_set") or []
                            new_tags = new_payload.get("belongs_to_set") or []
                            if old_tags or new_tags:
                                merged = list(dict.fromkeys(list(old_tags) + list(new_tags)))
                                new_payload["belongs_to_set"] = merged
                                row["payload"] = json.dumps(new_payload)

                        await session.execute(
                            text(
                                f'INSERT OR REPLACE INTO "{collection_name}" '
                                f"(id, payload, vector) VALUES (:id, json(:payload), :vector)"
                            ),
                            {
                                "id": row["id"],
                                "payload": row["payload"],
                                "vector": row["vector"],
                            },
                        )
                await session.commit()

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """Create the underlying index collection table."""
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        """Write index rows derived from data_points into the {index}_{property} table."""
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

    # ── Retrieval ────────────────────────────────────────────────────────

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        """Retrieve data points from a collection by their IDs."""
        if not await self.has_collection(collection_name):
            return []

        results = []

        async with self.get_async_session() as session:
            for start in range(0, len(data_point_ids), QUERY_BATCH_SIZE):
                id_batch = data_point_ids[start : start + QUERY_BATCH_SIZE]
                placeholders = ", ".join([f":id_{i}" for i in range(len(id_batch))])
                params = {f"id_{i}": str(uid) for i, uid in enumerate(id_batch)}

                batch_results = await session.execute(
                    text(
                        f'SELECT id, payload FROM "{collection_name}" '
                        f"WHERE id IN ({placeholders})"
                    ),
                    params,
                )
                results.extend(batch_results.all())

        seen_ids = set()
        unique_results = []
        for row in results:
            if row[0] in seen_ids:
                continue
            seen_ids.add(row[0])
            unique_results.append(row)

        scored = []
        for row in unique_results:
            payload = row[1]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError):
                    payload = None
            scored.append(
                ScoredResult(id=parse_id(row[0]), payload=payload, score=0)
            )
        return scored

    # ── Search ───────────────────────────────────────────────────────────

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
        """Run cosine-distance similarity search using vector_distance_cos()."""
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        if not await self.has_collection(collection_name):
            return []

        if limit is None:
            async with self.get_async_session() as session:
                result = await session.execute(
                    text(f'SELECT COUNT(*) FROM "{collection_name}"')
                )
                limit = result.scalar()

        if limit <= 0:
            return []

        query_vector_json = json.dumps(query_vector)

        # Build SELECT columns
        select_cols = "id, vector_distance_cos(vector, :qvec) AS distance"
        if include_payload:
            select_cols = "id, payload, vector_distance_cos(vector, :qvec) AS distance"

        params: dict[str, Any] = {"qvec": query_vector_json, "limit": limit}

        # Build WHERE clause for belongs_to_set filtering
        where_clause = ""
        if node_name:
            # Filter rows whose payload->belongs_to_set contains any/all of the given names
            # Using json_each() to check array membership
            if node_name_filter_operator == "AND":
                # All node_names must be present
                conditions = []
                for i, name in enumerate(node_name):
                    param_key = f"nn_{i}"
                    conditions.append(
                        f"EXISTS (SELECT 1 FROM json_each("
                        f'json_extract(payload, \'$.belongs_to_set\')) '
                        f"WHERE value = :{param_key})"
                    )
                    params[param_key] = name
                where_clause = "WHERE " + " AND ".join(conditions)
            else:
                # OR: any node_name must be present
                conditions = []
                for i, name in enumerate(node_name):
                    param_key = f"nn_{i}"
                    conditions.append(
                        f"EXISTS (SELECT 1 FROM json_each("
                        f'json_extract(payload, \'$.belongs_to_set\')) '
                        f"WHERE value = :{param_key})"
                    )
                    params[param_key] = name
                where_clause = "WHERE (" + " OR ".join(conditions) + ")"

        query_sql = text(
            f'SELECT {select_cols} FROM "{collection_name}" '
            f"{where_clause} "
            f"ORDER BY distance ASC LIMIT :limit"
        )

        async with self.get_async_session() as session:
            closest_items = await session.execute(query_sql, params)
            rows = closest_items.all()

        if not rows:
            return []

        results = []
        for row in rows:
            if include_payload:
                row_id, payload_raw, distance = row
                payload = payload_raw
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except (json.JSONDecodeError, TypeError):
                        payload = None
            else:
                row_id, distance = row
                payload = None

            results.append(
                ScoredResult(
                    id=parse_id(str(row_id)),
                    payload=payload if include_payload else None,
                    score=float(distance) if distance is not None else 0.0,
                )
            )

        return results

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
    ):
        """Run search concurrently for each query text."""
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

    # ── Deletion ─────────────────────────────────────────────────────────

    async def delete_data_points(self, collection_name: str, data_point_ids: list[UUID]):
        """Delete rows from the collection whose id is in data_point_ids."""
        if not await self.has_collection(collection_name):
            return None

        async with self._get_write_lock(collection_name):
            async with self.get_async_session() as session:
                for start in range(0, len(data_point_ids), QUERY_BATCH_SIZE):
                    id_batch = data_point_ids[start : start + QUERY_BATCH_SIZE]
                    placeholders = ", ".join([f":id_{i}" for i in range(len(id_batch))])
                    params = {f"id_{i}": str(uid) for i, uid in enumerate(id_batch)}

                    await session.execute(
                        text(
                            f'DELETE FROM "{collection_name}" '
                            f"WHERE id IN ({placeholders})"
                        ),
                        params,
                    )
                await session.commit()

    async def remove_belongs_to_set_tags(
        self,
        tags: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> None:
        """Strip given tag names from belongs_to_set arrays in all vector collections.

        Deletes rows whose belongs_to_set becomes empty after tag removal.
        Uses SQLite/libSQL JSON functions for array manipulation.
        """
        if not tags:
            return None

        if node_ids is not None and not node_ids:
            return None

        # Get all tables, filter to PascalCase (vector collection convention)
        all_tables = await self.get_table_names()
        candidate_tables = []
        for name in all_tables:
            if name and name[0].isupper():
                candidate_tables.append(name)

        for table_name in candidate_tables:
            try:
                async with self._get_write_lock(table_name):
                    async with self.get_async_session() as session:
                        # Build node_ids filter
                        id_filter = ""
                        id_params: dict[str, str] = {}
                        if node_ids is not None:
                            id_placeholders = ", ".join(
                                [f":nid_{i}" for i in range(len(node_ids))]
                            )
                            id_params = {
                                f"nid_{i}": str(nid) for i, nid in enumerate(node_ids)
                            }
                            id_filter = f" AND id IN ({id_placeholders})"

                        # Find rows that have any of the target tags
                        tag_conditions = []
                        tag_params: dict[str, str] = {}
                        for i, tag in enumerate(tags):
                            tag_key = f"tag_{i}"
                            tag_conditions.append(
                                f"EXISTS (SELECT 1 FROM json_each("
                                f"json_extract(payload, '$.belongs_to_set')) "
                                f"WHERE value = :{tag_key})"
                            )
                            tag_params[tag_key] = tag

                        where_tags = " OR ".join(tag_conditions)
                        all_params = {**tag_params, **id_params}

                        # Select target rows
                        select_sql = text(
                            f'SELECT id, payload FROM "{table_name}" '
                            f"WHERE ({where_tags}){id_filter}"
                        )
                        result = await session.execute(select_sql, all_params)
                        target_rows = result.all()

                        if not target_rows:
                            await session.commit()
                            continue

                        tags_set = set(tags)
                        ids_to_delete = []

                        for row_id, payload_raw in target_rows:
                            payload = payload_raw
                            if isinstance(payload, str):
                                try:
                                    payload = json.loads(payload)
                                except (json.JSONDecodeError, TypeError):
                                    continue

                            if not isinstance(payload, dict):
                                continue

                            bts = payload.get("belongs_to_set") or []
                            new_bts = [t for t in bts if t not in tags_set]

                            if not new_bts:
                                ids_to_delete.append(row_id)
                            else:
                                payload["belongs_to_set"] = new_bts
                                await session.execute(
                                    text(
                                        f'UPDATE "{table_name}" '
                                        f"SET payload = json(:payload) WHERE id = :id"
                                    ),
                                    {"payload": json.dumps(payload), "id": row_id},
                                )

                        # Delete rows with empty belongs_to_set
                        if ids_to_delete:
                            placeholders = ", ".join(
                                [f":del_{i}" for i in range(len(ids_to_delete))]
                            )
                            del_params = {
                                f"del_{i}": str(uid) for i, uid in enumerate(ids_to_delete)
                            }
                            await session.execute(
                                text(
                                    f'DELETE FROM "{table_name}" '
                                    f"WHERE id IN ({placeholders})"
                                ),
                                del_params,
                            )

                        await session.commit()

            except Exception as e:
                logger.debug(
                    "remove_belongs_to_set_tags skipped '%s': %s",
                    table_name,
                    e,
                )

        return None

    # ── Pruning ──────────────────────────────────────────────────────────

    async def prune(self):
        """Drop all vector collection tables."""
        all_tables = await self.get_table_names()
        async with self.get_async_session() as session:
            for table_name in all_tables:
                await session.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
            await session.commit()

    async def run_migrations(self):
        """No-op for Turso adapter."""
        return None
