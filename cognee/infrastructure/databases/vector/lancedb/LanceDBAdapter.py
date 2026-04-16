import asyncio
from os import path
from uuid import UUID
import lancedb
from pydantic import BaseModel
from lancedb.pydantic import LanceModel, Vector
from typing import Generic, List, Optional, TypeVar, Union, get_args, get_origin, get_type_hints

from cognee.infrastructure.databases.exceptions import MissingQueryParameterError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.files.storage import get_file_storage
from cognee.modules.storage.utils import copy_model
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.pgvector.serialize_data import serialize_data
from cognee.shared.logging_utils import get_logger

from ..embeddings.EmbeddingEngine import EmbeddingEngine
from ..models.ScoredResult import ScoredResult
from ..vector_db_interface import VectorDBInterface

from cognee.modules.observability import new_span
from cognee.modules.observability.tracing import (
    COGNEE_DB_SYSTEM,
    COGNEE_VECTOR_COLLECTION,
    COGNEE_VECTOR_RESULT_COUNT,
)

logger = get_logger("LanceDBAdapter")


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

    metadata: dict = {"index_fields": ["text"]}
    belongs_to_set: List[str] = []


class LanceDBAdapter(VectorDBInterface):
    """Vector-database adapter backed by LanceDB; implements the VectorDBInterface contract."""

    name = "LanceDB"
    url: str
    api_key: str
    connection: lancedb.AsyncConnection = None

    def __init__(
        self,
        url: Optional[str],
        api_key: Optional[str],
        embedding_engine: EmbeddingEngine,
    ):
        """Store connection params; connection is lazily established in get_connection()."""
        self.url = url
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.VECTOR_DB_LOCK = asyncio.Lock()

    async def get_connection(self):
        """
        Establishes and returns a connection to the LanceDB.

        If a connection already exists, it will return the existing connection.

        Returns:
        --------

            - lancedb.AsyncConnection: An active connection to the LanceDB.
        """
        if self.connection is None:
            self.connection = await lancedb.connect_async(self.url, api_key=self.api_key)

        return self.connection

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

    async def create_collection(self, collection_name: str, payload_schema: BaseModel):
        vector_size = self.embedding_engine.get_vector_size()

        payload_schema = self.get_data_point_schema(payload_schema)
        data_point_types = get_type_hints(payload_schema)

        class LanceDataPoint(LanceModel):
            """
            Represents a data point in the Lance model with an ID, vector, and associated payload.

            The class inherits from LanceModel and defines the following public attributes:
            - id: A unique identifier for the data point.
            - vector: A vector representing the data point in a specified dimensional space.
            - payload: Additional data or metadata associated with the data point.
            """

            id: data_point_types["id"]
            vector: Vector(vector_size)
            payload: payload_schema

        if not await self.has_collection(collection_name):
            async with self.VECTOR_DB_LOCK:
                if not await self.has_collection(collection_name):
                    connection = await self.get_connection()
                    return await connection.create_table(
                        name=collection_name,
                        schema=LanceDataPoint,
                        exist_ok=True,
                    )

    async def get_collection(self, collection_name: str):
        """Return the LanceDB table for `collection_name` or raise CollectionNotFoundError."""
        if not await self.has_collection(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found!")

        connection = await self.get_connection()
        return await connection.open_table(collection_name)

    async def create_data_points(self, collection_name: str, data_points: list[DataPoint]):
        payload_schema = type(data_points[0])

        if not await self.has_collection(collection_name):
            async with self.VECTOR_DB_LOCK:
                if not await self.has_collection(collection_name):
                    await self.create_collection(
                        collection_name,
                        payload_schema,
                    )

        collection = await self.get_collection(collection_name)

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        )

        # Fetch existing `belongs_to_set` values for rows we are about to
        # upsert so the same DataPoint cognified into multiple datasets
        # accumulates every dataset tag. Without this, merge_insert's
        # when_matched_update_all overwrites the prior dataset's tags.
        existing_belongs_to_set: dict[str, list] = {}
        incoming_ids = [str(dp.id) for dp in data_points]
        if incoming_ids:
            # Build the WHERE predicate explicitly with escaped string literals
            # rather than relying on Python's tuple repr — mirrors how search()
            # escapes `name` values to keep single-quotes from breaking the
            # LanceDB SQL grammar.
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
                # Best-effort: if the lookup fails (e.g. empty table, schema
                # mismatch that the migration path will handle), fall through
                # to the standard upsert.
                logger.debug(
                    "belongs_to_set merge lookup failed for '%s': %s",
                    collection_name,
                    e,
                )

        IdType = TypeVar("IdType")
        PayloadSchema = TypeVar("PayloadSchema")
        vector_size = self.embedding_engine.get_vector_size()

        class LanceDataPoint(LanceModel, Generic[IdType, PayloadSchema]):
            """
            Represents a data point in the Lance model with an ID, vector, and payload.

            This class encapsulates a data point consisting of an identifier, a vector representing
            the data, and an associated payload, allowing for operations and manipulations specific
            to the Lance data structure.
            """

            id: IdType
            vector: Vector(vector_size)
            payload: PayloadSchema

        def create_lance_data_point(data_point: DataPoint, vector: list[float]) -> LanceDataPoint:
            """Build a typed LanceDataPoint from a DataPoint, merging any prior belongs_to_set tags."""
            payload_model = self.get_data_point_schema(type(data_point))
            properties = payload_model.model_validate(
                serialize_data(data_point.model_dump())
            ).model_dump()

            prior = existing_belongs_to_set.get(str(data_point.id))
            if prior:
                incoming = properties.get("belongs_to_set") or []
                properties["belongs_to_set"] = list(dict.fromkeys(list(prior) + list(incoming)))

            return LanceDataPoint[str, self.get_data_point_schema(type(data_point))](
                id=str(data_point.id),
                vector=vector,
                payload=properties,
            )

        lance_data_points = [
            create_lance_data_point(data_point, data_vectors[data_point_index])
            for (data_point_index, data_point) in enumerate(data_points)
        ]

        # Dedup by id within the batch — on duplicates, union `belongs_to_set`
        # instead of keeping only the last occurrence. A plain dict-collapse
        # would drop tags that only appeared on earlier siblings (mirrors the
        # batch-merge logic in PGVectorAdapter.create_data_points and
        # Neo4jAdapter.add_nodes).
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

        try:
            async with self.VECTOR_DB_LOCK:
                await (
                    collection.merge_insert("id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(lance_data_points)
                )
        except (ValueError, OSError, RuntimeError) as e:
            if "not found in target schema" not in str(e):
                raise
            logger.warning(
                "Schema mismatch detected for collection '%s', migrating table: %s",
                collection_name,
                e,
            )
            await self._migrate_collection_schema(
                collection_name, collection, payload_schema, lance_data_points
            )

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
        data_point_types = get_type_hints(schema_model)
        valid_payload_fields = set(schema_model.model_fields.keys())
        defaults = self._get_payload_defaults(payload_schema)

        class MigrationLanceDataPoint(LanceModel):
            id: data_point_types["id"]
            vector: Vector(vector_size)
            payload: schema_model

        new_ids = {dp.id for dp in new_lance_data_points}
        typed_old_rows = []
        skipped = 0
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
                logger.warning(
                    "Skipping row %s during migration (validation failed): %s",
                    row.get("id", "?"),
                    e,
                )
                skipped += 1

        if skipped:
            logger.warning(
                "Migration of '%s': skipped %d rows due to validation errors",
                collection_name,
                skipped,
            )

        async with self.VECTOR_DB_LOCK:
            connection = await self.get_connection()
            await connection.drop_table(collection_name)
            await connection.create_table(
                name=collection_name,
                schema=MigrationLanceDataPoint,
            )
            collection = await connection.open_table(collection_name)

            if typed_old_rows:
                await collection.add(typed_old_rows)

            if new_lance_data_points:
                await (
                    collection.merge_insert("id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(new_lance_data_points)
                )

        logger.info(
            "Migrated collection '%s' schema (%d existing rows preserved)",
            collection_name,
            len(typed_old_rows),
        )

    @staticmethod
    def _all_datapoint_subclasses():
        seen = set()
        stack = list(DataPoint.__subclasses__())
        subclasses = []

        while stack:
            subclass = stack.pop()
            if subclass in seen:
                continue
            seen.add(subclass)
            subclasses.append(subclass)
            stack.extend(subclass.__subclasses__())

        return subclasses

    @staticmethod
    def _warm_up_common_datapoint_models():
        # Import commonly indexed data points so they are present in __subclasses__.
        # Imports are best-effort to avoid hard dependency on optional modules.

        # TODO: Hard-coded for now, might be cumbersome
        modules_to_import = [
            "cognee.modules.chunking.models",
            "cognee.tasks.summarization.models",
            "cognee.modules.engine.models",
            "cognee.modules.data.processing.document_types",
            "cognee.modules.graph.models.EdgeType",
        ]

        for module_name in modules_to_import:
            try:
                __import__(module_name)
            except Exception:
                logger.debug("Skipping optional datapoint import during migration: %s", module_name)

    @classmethod
    def _resolve_collection_payload_schema(cls, collection_name: str):
        if "_" not in collection_name:
            return None
        # Important note: Depends on naming convention (e.g. DocumentChunk_text)
        type_name, _, index_field = collection_name.rpartition("_")
        if not type_name or not index_field:
            return None

        cls._warm_up_common_datapoint_models()
        for data_point_cls in cls._all_datapoint_subclasses():
            if data_point_cls.__name__ != type_name:
                continue

            metadata_field = data_point_cls.model_fields.get("metadata")
            metadata_default = metadata_field.default if metadata_field else None
            if isinstance(metadata_default, dict):
                index_fields = metadata_default.get("index_fields", [])
                if index_fields and index_field not in index_fields:
                    continue

            return data_point_cls

        return None

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
        """Extract default values from the Pydantic payload model."""
        schema_model = self.get_data_point_schema(payload_schema)
        defaults = {}
        for name, field_info in schema_model.model_fields.items():
            if field_info.default is not None and not (
                hasattr(field_info, "is_required") and field_info.is_required()
            ):
                defaults[name] = field_info.default
        return defaults

    def _coerce_rows_to_typed_payload(self, rows: list, payload_schema: Optional[type]) -> list:
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
            except Exception as e:
                logger.debug(
                    "_coerce_rows_to_typed_payload: validation fell back for id=%s: %s",
                    row.get("id"),
                    e,
                )
                coerced.append(row)
                continue
            new_row = dict(row)
            new_row["payload"] = validated
            coerced.append(new_row)
        return coerced

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        try:
            collection = await self.get_collection(collection_name)
        except CollectionNotFoundError:
            # If collection doesn't exist, return empty list (no items to retrieve)
            return []

        if len(data_point_ids) == 1:
            query = collection.query().where(f"id = '{data_point_ids[0]}'")
        else:
            query = collection.query().where(f"id IN {tuple(data_point_ids)}")

        # Convert query results to list format
        results_list = await query.to_list()

        return [
            ScoredResult(
                id=parse_id(result["id"]),
                payload=result["payload"],
                score=0,
            )
            for result in results_list
        ]

    async def search(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: List[float] = None,
        limit: Optional[int] = 15,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
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
        query_texts: List[str],
        limit: Optional[int] = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
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

    async def delete_data_points(self, collection_name: str, data_point_ids: list[UUID]):
        # Skip deletion if collection doesn't exist
        if not await self.has_collection(collection_name):
            return

        collection = await self.get_collection(collection_name)

        # Delete one at a time to avoid commit conflicts
        for data_point_id in data_point_ids:
            await collection.delete(f"id = '{data_point_id}'")

    async def remove_belongs_to_set_tags(self, tags: List[str]) -> None:
        """
        Strip the given tag names from `belongs_to_set` arrays in every
        table and delete rows whose array becomes empty. Used to reconcile
        surviving shared rows after a dataset/NodeSet is deleted.

        LanceDB doesn't support in-place array mutation, so the update path
        reads rows that reference any target tag, rewrites the payload with
        the tag removed, and either re-inserts them (merge_insert) or
        deletes them when the array is empty.
        """
        if not tags:
            return None

        tag_set = set(tags)
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

            async with self.VECTOR_DB_LOCK:
                try:
                    rows = await collection.query().where(where_clause).to_list()
                except Exception as e:
                    logger.debug(
                        "remove_belongs_to_set_tags: row scan failed for '%s': %s",
                        collection_name,
                        e,
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

                for row_id in rows_to_delete:
                    await collection.delete(f"id = '{row_id}'")

                # LanceDB merge_insert silently no-ops when given dicts whose
                # nested payload shape doesn't match the struct schema, so
                # delete + re-add is the reliable path to persist the
                # rewritten belongs_to_set. If the re-add fails we've
                # already deleted the originals — escalate to WARNING with
                # the affected ids and re-raise so the caller sees it; a
                # silent debug log would leave the collection short of rows.
                for row in rows_to_update:
                    await collection.delete(f"id = '{row['id']}'")
                if rows_to_update:
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

        return None

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
                    belongs_to_set=(data_point.belongs_to_set or []),
                )
                for data_point in data_points
            ],
        )

    async def prune(self):
        connection = await self.get_connection()
        collection_names = await connection.table_names()

        for collection_name in collection_names:
            collection = await self.get_collection(collection_name)
            await collection.delete("id IS NOT NULL")
            await connection.drop_table(collection_name)

        if self.url.startswith("/"):
            db_dir_path = path.dirname(self.url)
            db_file_name = path.basename(self.url)
            await get_file_storage(db_dir_path).remove_all(db_file_name)

    def get_data_point_schema(self, model_type: BaseModel):
        related_models_fields = []

        for field_name, field_config in model_type.model_fields.items():
            if hasattr(field_config, "model_fields"):
                related_models_fields.append(field_name)

            elif hasattr(field_config.annotation, "model_fields"):
                related_models_fields.append(field_name)

            elif (
                get_origin(field_config.annotation) == Union
                or get_origin(field_config.annotation) is list
            ):
                models_list = get_args(field_config.annotation)
                if any(hasattr(model, "model_fields") for model in models_list):
                    related_models_fields.append(field_name)
                elif models_list and any(get_args(model) is DataPoint for model in models_list):
                    related_models_fields.append(field_name)
                elif models_list and any(
                    submodel is DataPoint for submodel in get_args(models_list[0])
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
                "belongs_to_set": (Optional[List[str]], None),
            },
            exclude_fields=["metadata"] + related_models_fields,
        )
