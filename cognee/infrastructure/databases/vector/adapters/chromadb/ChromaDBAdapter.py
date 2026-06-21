import json
import asyncio
from uuid import UUID
from typing import Any, List, Literal, Optional

from pydantic import BaseModel

from cognee.shared.logging_utils import get_logger
from cognee.modules.storage.utils import get_own_properties
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.databases.exceptions import MissingQueryParameterError

from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface

try:
    from chromadb import AsyncHttpClient, Settings
except ImportError:  # pragma: no cover - optional extra
    AsyncHttpClient = Any  # type: ignore[misc, assignment]
    Settings = Any  # type: ignore[misc, assignment]

logger = get_logger("ChromaDBAdapter")

BELONGS_TO_SET_KEY = "belongs_to_set"
LEGACY_BELONGS_TO_SET_LIST_KEY = f"{BELONGS_TO_SET_KEY}__list"
LEGACY_BELONGS_TO_SET_MEMBER_PREFIX = f"{BELONGS_TO_SET_KEY}::"
MAX_NODE_NAME_LENGTH = 256


class IndexSchema(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"]}

    def model_dump(self):
        data = super().model_dump()
        return process_data_for_chroma(data)


def _normalize_belongs_to_set_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        tags = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
            tags = parsed if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            tags = [value]
    else:
        return []
    cleaned: list[str] = []
    for tag in tags:
        if isinstance(tag, str):
            stripped = tag.strip()
            if stripped and len(stripped) <= MAX_NODE_NAME_LENGTH:
                cleaned.append(stripped)
    return cleaned


def process_data_for_chroma(data: dict) -> dict:
    """Serialize metadata for ChromaDB, storing belongs_to_set as a native string array."""
    processed_data: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, UUID):
            processed_data[key] = str(value)
        elif isinstance(value, dict):
            processed_data[f"{key}__dict"] = json.dumps(value)
        elif isinstance(value, list):
            if key == BELONGS_TO_SET_KEY:
                processed_data[key] = _normalize_belongs_to_set_tags(value)
            else:
                processed_data[f"{key}__list"] = json.dumps(value)
        elif isinstance(value, (str, int, float, bool)):
            processed_data[key] = value
        else:
            processed_data[key] = str(value)
    return processed_data


def restore_data_from_chroma(data: dict) -> dict:
    """Restore metadata from ChromaDB storage, including legacy belongs_to_set formats."""
    restored_data: dict[str, Any] = {}
    dict_keys: list[str] = []
    list_keys: list[str] = []

    for key, value in data.items():
        if key.endswith("__dict"):
            dict_keys.append(key)
        elif key.endswith("__list"):
            list_keys.append(key)
        elif key.startswith(LEGACY_BELONGS_TO_SET_MEMBER_PREFIX):
            continue
        else:
            restored_data[key] = value

    for key in dict_keys:
        original_key = key[:-6]
        try:
            restored_data[original_key] = json.loads(data[key])
        except json.JSONDecodeError as exc:
            logger.debug("Error restoring dictionary from JSON: %s", exc)
            restored_data[key] = data[key]

    for key in list_keys:
        original_key = key[:-6]
        if original_key == BELONGS_TO_SET_KEY and isinstance(
            restored_data.get(BELONGS_TO_SET_KEY), list
        ):
            continue
        try:
            restored_data[original_key] = json.loads(data[key])
        except json.JSONDecodeError as exc:
            logger.debug("Error restoring list from JSON: %s", exc)
            restored_data[key] = data[key]

    if BELONGS_TO_SET_KEY not in restored_data and LEGACY_BELONGS_TO_SET_LIST_KEY in data:
        restored_data[BELONGS_TO_SET_KEY] = _normalize_belongs_to_set_tags(
            data[LEGACY_BELONGS_TO_SET_LIST_KEY]
        )

    if BELONGS_TO_SET_KEY in restored_data:
        restored_data[BELONGS_TO_SET_KEY] = _normalize_belongs_to_set_tags(
            restored_data[BELONGS_TO_SET_KEY]
        )

    return restored_data


def metadata_needs_belongs_to_set_migration(metadata: dict) -> bool:
    if not metadata:
        return False
    if LEGACY_BELONGS_TO_SET_LIST_KEY in metadata:
        return True
    if any(key.startswith(LEGACY_BELONGS_TO_SET_MEMBER_PREFIX) for key in metadata):
        return True
    current = metadata.get(BELONGS_TO_SET_KEY)
    return isinstance(current, str)


def migrate_belongs_to_set_metadata(metadata: dict) -> dict:
    """Convert legacy belongs_to_set storage to native array metadata."""
    migrated = dict(metadata)
    tags = _normalize_belongs_to_set_tags(migrated.get(BELONGS_TO_SET_KEY))

    if LEGACY_BELONGS_TO_SET_LIST_KEY in migrated:
        tags = _normalize_belongs_to_set_tags(migrated.pop(LEGACY_BELONGS_TO_SET_LIST_KEY)) or tags

    for key, value in list(migrated.items()):
        if key.startswith(LEGACY_BELONGS_TO_SET_MEMBER_PREFIX) and value is True:
            tag = key[len(LEGACY_BELONGS_TO_SET_MEMBER_PREFIX) :]
            if tag and tag not in tags:
                tags.append(tag)
            migrated.pop(key, None)

    migrated[BELONGS_TO_SET_KEY] = tags
    return migrated


def sanitize_node_names(node_name: Optional[List[str]]) -> Optional[List[str]]:
    if not node_name:
        return None
    cleaned: list[str] = []
    for raw in node_name:
        if not isinstance(raw, str):
            continue
        name = raw.strip()
        if not name or len(name) > MAX_NODE_NAME_LENGTH:
            continue
        cleaned.append(name)
    return cleaned or None


class ChromaDBAdapter(VectorDBInterface):
    """Community-registered ChromaDB vector adapter with native belongs_to_set arrays."""

    name = "ChromaDB"
    url: str
    api_key: str
    connection: AsyncHttpClient = None

    def __init__(
        self, url: Optional[str], api_key: Optional[str], embedding_engine: EmbeddingEngine
    ):
        self.embedding_engine = embedding_engine
        self.url = url
        self.api_key = api_key
        self.VECTOR_DB_LOCK = asyncio.Lock()

    async def get_connection(self) -> AsyncHttpClient:
        if self.connection is None:
            settings = Settings(
                chroma_client_auth_provider="token",
                chroma_client_auth_credentials=self.api_key or "",
            )
            self.connection = await AsyncHttpClient(host=self.url, settings=settings)
        return self.connection

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        collections = await self.get_collection_names()
        return collection_name in collections

    async def create_collection(self, collection_name: str, payload_schema=None):
        async with self.VECTOR_DB_LOCK:
            client = await self.get_connection()
            if not await self.has_collection(collection_name):
                await client.create_collection(
                    name=collection_name, metadata={"hnsw:space": "cosine"}
                )

    async def get_collection(self, collection_name: str):
        if not await self.has_collection(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' not found!")
        client = await self.get_connection()
        return await client.get_collection(collection_name)

    async def create_data_points(self, collection_name: str, data_points: list[DataPoint]):
        await self.create_collection(collection_name)
        collection = await self.get_collection(collection_name)

        texts = [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
        embeddings = await self.embed_data(texts)
        ids = [str(data_point.id) for data_point in data_points]
        metadatas = [
            process_data_for_chroma(get_own_properties(data_point)) for data_point in data_points
        ]

        await collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts
        )

    async def create_vector_index(self, index_name: str, index_property_name: str):
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=data_point.id,
                    text=getattr(data_point, data_point.metadata["index_fields"][0]),
                )
                for data_point in data_points
            ],
        )

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        try:
            collection = await self.get_collection(collection_name)
        except CollectionNotFoundError:
            return []

        results = await collection.get(ids=data_point_ids, include=["metadatas"])
        return [
            ScoredResult(
                id=parse_id(record_id),
                payload=restore_data_from_chroma(metadata),
                score=0,
            )
            for record_id, metadata in zip(results["ids"], results["metadatas"])
        ]

    @staticmethod
    def _build_where_filter(
        node_name: Optional[List[str]],
        node_name_filter_operator: Literal["OR", "AND"],
    ) -> Optional[dict]:
        sanitized = sanitize_node_names(node_name)
        if node_name_filter_operator not in ("OR", "AND"):
            raise ValueError(
                f"Unsupported node_name_filter_operator: {node_name_filter_operator!r}. "
                "Expected 'OR' or 'AND'."
            )
        if not sanitized:
            return None

        def _contains_clause(name: str) -> dict:
            return {BELONGS_TO_SET_KEY: {"$contains": name}}

        if len(sanitized) == 1:
            return _contains_clause(sanitized[0])
        operator_key = "$and" if node_name_filter_operator == "AND" else "$or"
        return {operator_key: [_contains_clause(name) for name in sanitized]}

    @staticmethod
    def _build_include_list(include_payload: bool, with_vector: bool) -> List[str]:
        include = ["distances"]
        if include_payload:
            include.append("metadatas")
        if with_vector:
            include.append("embeddings")
        return include

    async def search(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: List[float] = None,
        limit: Optional[int] = 15,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: Literal["OR", "AND"] = "OR",
    ):
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        try:
            collection = await self.get_collection(collection_name)
            if limit is None:
                limit = await collection.count()
            if limit <= 0:
                return []

            where_filter = self._build_where_filter(node_name, node_name_filter_operator)
            include_list = self._build_include_list(include_payload, with_vector)
            query_kwargs = {
                "query_embeddings": [query_vector],
                "include": include_list,
                "n_results": limit,
            }
            if where_filter is not None:
                query_kwargs["where"] = where_filter

            results = await collection.query(**query_kwargs)
            vector_list = []
            metadatas = results.get("metadatas")
            for index, (record_id, distance) in enumerate(
                zip(results["ids"][0], results["distances"][0])
            ):
                payload = None
                if include_payload and metadatas:
                    payload = restore_data_from_chroma(metadatas[0][index])
                vector_list.append(
                    {
                        "id": parse_id(record_id),
                        "payload": payload,
                        "score": float(distance),
                        "vector": results["embeddings"][0][index]
                        if with_vector and "embeddings" in results
                        else None,
                    }
                )

            return [
                ScoredResult(
                    id=row["id"],
                    payload=row["payload"],
                    score=row["score"],
                    vector=row["vector"],
                )
                for row in vector_list
            ]
        except Exception as exc:
            logger.warning("Error in ChromaDB search: %s", exc)
            return []

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = 5,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: Literal["OR", "AND"] = "OR",
    ):
        query_vectors = await self.embed_data(query_texts)
        collection = await self.get_collection(collection_name)
        where_filter = self._build_where_filter(node_name, node_name_filter_operator)
        include_list = self._build_include_list(include_payload, with_vectors)

        query_kwargs = {
            "query_embeddings": query_vectors,
            "include": include_list,
            "n_results": limit,
        }
        if where_filter is not None:
            query_kwargs["where"] = where_filter

        results = await collection.query(**query_kwargs)
        all_results = []
        metadatas = results.get("metadatas")
        for query_index in range(len(query_texts)):
            query_results = []
            for record_index, (record_id, distance) in enumerate(
                zip(results["ids"][query_index], results["distances"][query_index])
            ):
                payload = None
                if include_payload and metadatas:
                    payload = restore_data_from_chroma(metadatas[query_index][record_index])
                result = ScoredResult(
                    id=parse_id(record_id),
                    payload=payload,
                    score=float(distance),
                )
                if with_vectors and "embeddings" in results:
                    result.vector = results["embeddings"][query_index][record_index]
                query_results.append(result)
            all_results.append(query_results)
        return all_results

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        if not await self.has_collection(collection_name):
            return True
        collection = await self.get_collection(collection_name)
        await collection.delete(ids=data_point_ids)
        return True

    async def prune(self):
        client = await self.get_connection()
        collections = await client.list_collections()
        for collection_name in collections:
            await client.delete_collection(collection_name)
        return True

    async def get_collection_names(self):
        client = await self.get_connection()
        return await client.list_collections()

    async def run_migrations(self) -> dict[str, int]:
        """Migrate legacy belongs_to_set JSON/boolean metadata to native arrays."""
        summary = {"checked_collections": 0, "migrated_records": 0}
        try:
            collection_names = await self.get_collection_names()
        except Exception as exc:
            logger.warning("ChromaDB migration skipped: %s", exc)
            return summary

        for collection_name in collection_names:
            summary["checked_collections"] += 1
            try:
                collection = await self.get_collection(collection_name)
                records = await collection.get(include=["metadatas", "embeddings", "documents"])
            except Exception as exc:
                logger.warning(
                    "ChromaDB migration failed while reading '%s': %s", collection_name, exc
                )
                continue

            ids = records.get("ids") or []
            metadatas = records.get("metadatas") or []
            embeddings = records.get("embeddings") or []
            documents = records.get("documents") or []

            migrated_ids: list[str] = []
            migrated_metadatas: list[dict] = []
            migrated_embeddings: list[list[float]] = []
            migrated_documents: list[str] = []

            for index, record_id in enumerate(ids):
                metadata = metadatas[index] if index < len(metadatas) else {}
                if not metadata_needs_belongs_to_set_migration(metadata):
                    continue
                migrated_ids.append(record_id)
                migrated_metadatas.append(migrate_belongs_to_set_metadata(metadata))
                if embeddings:
                    migrated_embeddings.append(embeddings[index])
                if documents:
                    migrated_documents.append(documents[index])

            if not migrated_ids:
                continue

            upsert_kwargs = {
                "ids": migrated_ids,
                "metadatas": migrated_metadatas,
            }
            if migrated_embeddings:
                upsert_kwargs["embeddings"] = migrated_embeddings
            if migrated_documents:
                upsert_kwargs["documents"] = migrated_documents

            try:
                await collection.upsert(**upsert_kwargs)
                summary["migrated_records"] += len(migrated_ids)
            except Exception as exc:
                logger.warning(
                    "ChromaDB migration upsert failed for '%s': %s", collection_name, exc
                )

        if summary["migrated_records"]:
            logger.info(
                "ChromaDB belongs_to_set migration complete: %s records across %s collections",
                summary["migrated_records"],
                summary["checked_collections"],
            )
        return summary
