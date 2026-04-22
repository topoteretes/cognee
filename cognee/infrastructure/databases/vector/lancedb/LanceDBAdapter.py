import asyncio
import copy
from os import path
from uuid import UUID
from enum import Enum
import lancedb
from pydantic import BaseModel
from lancedb.pydantic import LanceModel, Vector
from typing import List, Optional, Union, get_args, get_origin, get_type_hints

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
_NO_DEFAULT = object()
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
    List: list,
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

    metadata: dict = {"index_fields": ["text"]}
    belongs_to_set: List[str] = []


class LanceDBAdapter(VectorDBInterface):
    name = "LanceDB"
    url: str
    api_key: str
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
    _payload_schema_cache: dict = {}
    _lance_datapoint_class_cache: dict = {}

    def __init__(
        self,
        url: Optional[str],
        api_key: Optional[str],
        embedding_engine: EmbeddingEngine,
        *,
        connection: Optional[object] = None,
        session: Optional[object] = None,
    ):
        """
        In subprocess-proxy mode, ``connection`` is a ``RemoteLanceDBConnection``
        and ``session`` is a ``LanceDBSubprocessSession``. In local mode both
        are ``None`` and the adapter lazily creates a ``lancedb.AsyncConnection``
        on first use.
        """
        self.url = url
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.VECTOR_DB_LOCK = asyncio.Lock()
        self.connection = connection
        self._session = session
        # Remember the adapter is bound to a specific subprocess session.
        # After close(), any operation that would re-open a real lancedb
        # connection locally is forbidden — otherwise we'd silently downgrade
        # subprocess-mode to local-mode and sidestep the whole point.
        self._subprocess_mode = session is not None
        self._permanently_closed = False

    async def get_connection(self):
        """
        Return the connection used by this adapter.

        - Local mode: lazily constructs a ``lancedb.AsyncConnection``.
        - Subprocess mode: returns the injected ``RemoteLanceDBConnection``
          (ensuring its underlying subprocess ``lancedb`` connection is opened).

        A subprocess-mode adapter that has been closed is an error state —
        we refuse to silently fall through to a local lancedb connection.
        """
        if self._permanently_closed:
            raise RuntimeError(
                "LanceDBAdapter is closed; a new adapter must be created "
                "(subprocess-mode adapters cannot be re-initialized)."
            )
        if self.connection is None:
            if self._subprocess_mode:
                raise RuntimeError(
                    "LanceDBAdapter subprocess session is gone; adapter cannot "
                    "be re-initialized in local mode."
                )
            self.connection = await lancedb.connect_async(self.url, api_key=self.api_key)
        else:
            # Remote connection lazily opens its own underlying lancedb handle.
            ensure = getattr(self.connection, "_ensure_connected", None)
            if ensure is not None:
                await ensure()
        return self.connection

    # ------------------------------------------------------------------
    # Subprocess-mode conversion helpers. In local mode these are no-ops.
    # ------------------------------------------------------------------
    def _schema_for_create_table(self, lance_model_cls):
        """Return the schema value to pass to ``connection.create_table``.

        Local mode: a ``LanceModel`` class (lancedb accepts it directly).
        Subprocess mode: a ``pa.Schema`` derived from the LanceModel so the
        worker doesn't need to see pydantic.
        """
        if self._session is None:
            return lance_model_cls
        return lance_model_cls.to_arrow_schema()

    def _records_for_write(self, records):
        """Convert LanceModel instances to a typed pyarrow RecordBatch in
        subprocess mode so the worker never needs to see pydantic. The
        RecordBatch carries both the data and the schema (derived from the
        LanceModel class via ``to_arrow_schema``) so LanceDB gets the exact
        types it expects.
        """
        if not records:
            return records
        if self._session is None:
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

    async def create_collection(self, collection_name: str, payload_schema: BaseModel):
        vector_size = self.embedding_engine.get_vector_size()

        payload_schema = self.get_data_point_schema(payload_schema)
        LanceDataPoint = self._make_lance_datapoint_cls(payload_schema, vector_size)

        if not await self.has_collection(collection_name):
            async with self.VECTOR_DB_LOCK:
                if not await self.has_collection(collection_name):
                    connection = await self.get_connection()
                    return await connection.create_table(
                        name=collection_name,
                        schema=self._schema_for_create_table(LanceDataPoint),
                        exist_ok=True,
                    )

    async def get_collection(self, collection_name: str):
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

        vector_size = self.embedding_engine.get_vector_size()

        # One LanceDataPoint class per (payload schema, vector size), cached
        # globally. Building a new class per call — let alone per record —
        # leaks pydantic SchemaValidator/Serializer state that never gets gc'd.
        def _lance_cls_for(data_point):
            schema = self.get_data_point_schema(type(data_point))
            return self._make_lance_datapoint_cls(schema, vector_size), schema

        def create_lance_data_point(data_point: DataPoint, vector: list[float]):
            lance_cls, payload_model = _lance_cls_for(data_point)
            properties = payload_model.model_validate(
                serialize_data(data_point.model_dump())
            ).model_dump()

            return lance_cls(
                id=str(data_point.id),
                vector=vector,
                payload=properties,
            )

        lance_data_points = [
            create_lance_data_point(data_point, data_vectors[data_point_index])
            for (data_point_index, data_point) in enumerate(data_points)
        ]

        lance_data_points = list({dp.id: dp for dp in lance_data_points}.values())

        try:
            async with self.VECTOR_DB_LOCK:
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
        MigrationLanceDataPoint = self._make_lance_datapoint_cls(
            schema_model, vector_size
        )

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

        async with self.VECTOR_DB_LOCK:
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

        if origin is Union:
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
        """Return the storable payload schema for ``model_type``. Memoized on
        the class — repeated calls with the same DataPoint subclass reuse the
        same synthesized Pydantic class instead of re-minting one every time
        (which pydantic's SchemaValidator / SchemaSerializer cache would
        otherwise accumulate indefinitely).
        """
        cached = self._payload_schema_cache.get(model_type)
        if cached is not None:
            return cached
        cached = self._build_data_point_schema(model_type)
        self._payload_schema_cache[model_type] = cached
        return cached

    @classmethod
    def _make_lance_datapoint_cls(cls, payload_schema, vector_size: int):
        """Return a concrete (non-generic) ``LanceDataPoint`` subclass for the
        given (payload_schema, vector_size) pair. Memoized globally so cognee
        workloads that keep hitting the same DataPoint types don't mint a new
        LanceModel subclass on every insert.
        """
        key = (payload_schema, int(vector_size))
        cached = cls._lance_datapoint_class_cache.get(key)
        if cached is not None:
            return cached

        class LanceDataPoint(LanceModel):
            id: str
            vector: Vector(vector_size)
            payload: payload_schema

        cls._lance_datapoint_class_cache[key] = LanceDataPoint
        return LanceDataPoint

    def _build_data_point_schema(self, model_type: BaseModel):
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

    async def close(self):
        """Release the underlying connection and, in subprocess mode, tear down
        the worker process. Once closed the adapter is not reusable.
        """
        self.connection = None
        if self._session is not None:
            try:
                self._session.shutdown()
            except Exception as e:
                logger.warning("Error shutting down LanceDB subprocess: %s", e)
            self._session = None
        self._permanently_closed = True
