"""
Redis vector database adapter using RedisVL.

Uses cognee's EmbeddingEngine, VectorDBInterface, DataPoint, and ScoredResult.
Adapted from cognee-community Redis vector adapter.
"""

import asyncio
import json
from typing import Any, List, Optional
from uuid import UUID

from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema as RedisVLIndexSchema

from cognee.infrastructure.databases.exceptions import MissingQueryParameterError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.shared.logging_utils import get_logger

logger = get_logger("RedisVectorAdapter")


class VectorEngineInitializationError(Exception):
    """Raised when Redis vector engine initialization fails."""


def _serialize_for_json(obj: Any) -> Any:
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    return obj


class RedisVectorAdapter(VectorDBInterface):
    """Redis vector database adapter using RedisVL and cognee's embedding engine."""

    name = "Redis"
    url: str | None
    embedding_engine: EmbeddingEngine | None = None

    def __init__(
        self,
        url: str,
        embedding_engine: EmbeddingEngine | None = None,
    ) -> None:
        if not url:
            raise VectorEngineInitializationError("Redis connection URL is required")
        if not embedding_engine:
            raise VectorEngineInitializationError("Embedding engine is required")

        self.url = url
        self.embedding_engine = embedding_engine
        self._indices: dict[str, AsyncSearchIndex] = {}
        self.VECTOR_DB_LOCK = asyncio.Lock()

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        return await self.embedding_engine.embed_text(data)

    def _create_schema(self, collection_name: str) -> RedisVLIndexSchema:
        schema_dict = {
            "index": {
                "name": collection_name,
                "prefix": collection_name,
                "storage_type": "json",
            },
            "fields": [
                {"name": "id", "type": "tag", "attrs": {"sortable": True}},
                {"name": "text", "type": "text", "attrs": {"sortable": True}},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "m": 32,
                        "dims": self.embedding_engine.get_vector_size(),
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
                {"name": "payload_data", "type": "text", "attrs": {"sortable": True}},
            ],
        }
        return RedisVLIndexSchema.from_dict(schema_dict)

    def _get_index(self, collection_name: str) -> AsyncSearchIndex:
        if collection_name not in self._indices:
            schema = self._create_schema(collection_name)
            self._indices[collection_name] = AsyncSearchIndex(
                schema=schema, redis_url=self.url, validate_on_load=True
            )
        return self._indices[collection_name]

    async def has_collection(self, collection_name: str) -> bool:
        try:
            index = self._get_index(collection_name)
            return await index.exists()
        except Exception:
            return False

    async def create_collection(
        self,
        collection_name: str,
        payload_schema: Any | None = None,
    ) -> None:
        async with self.VECTOR_DB_LOCK:
            index = self._get_index(collection_name)
            if await self.has_collection(collection_name):
                logger.info("Collection %s already exists", collection_name)
                return
            await index.create(overwrite=False)
            logger.info("Created collection %s", collection_name)

    async def create_data_points(
        self, collection_name: str, data_points: list[DataPoint]
    ) -> None:
        index = self._get_index(collection_name)
        if not await self.has_collection(collection_name):
            raise CollectionNotFoundError(f"Collection {collection_name} not found")

        data_vectors = await self.embed_data(
            [DataPoint.get_embeddable_data(dp) for dp in data_points]
        )
        documents = []
        for data_point, embedding in zip(data_points, data_vectors, strict=False):
            payload = _serialize_for_json(data_point.model_dump())
            text_attr = data_point.metadata.get("index_fields", ["text"])[0]
            text_val = getattr(data_point, text_attr, "")
            documents.append({
                "id": str(data_point.id),
                "text": text_val,
                "vector": embedding,
                "payload_data": json.dumps(payload),
            })
        await index.load(documents, id_field="id")
        logger.info("Created %d data points in collection %s", len(data_points), collection_name)

    async def create_vector_index(self, index_name: str, index_property_name: str) -> None:
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ) -> None:
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            data_points,
        )

    async def retrieve(
        self, collection_name: str, data_point_ids: list[str]
    ) -> list[dict[str, Any]]:
        index = self._get_index(collection_name)
        results = []
        for data_id in data_point_ids:
            doc = await index.fetch(data_id)
            if doc:
                payload_str = doc.get("payload_data", "{}")
                try:
                    results.append(json.loads(payload_str))
                except json.JSONDecodeError:
                    results.append(doc)
        return results

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: Optional[int] = 15,
        with_vector: bool = False,
        include_payload: bool = False,
    ) -> list[ScoredResult]:
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()

        if not await self.has_collection(collection_name):
            logger.warning(
                "Collection %s not found in RedisVectorAdapter.search; returning []",
                collection_name,
            )
            return []

        index = self._get_index(collection_name)
        if limit is None:
            info = await index.info()
            limit = info.get("num_docs", 15)
        if limit <= 0:
            return []

        if query_vector is None:
            query_vector = (await self.embed_data([query_text]))[0]

        vector_query = VectorQuery(
            vector=query_vector,
            vector_field_name="vector",
            num_results=limit,
            return_score=True,
            normalize_vector_distance=False,
        )
        return_fields = ["id", "text", "payload_data"]
        if with_vector:
            return_fields.append("vector")
        vector_query = vector_query.return_fields(*return_fields)
        results = await index.query(vector_query)

        scored_results = []
        for doc in results:
            payload = None
            if include_payload:
                payload_str = doc.get("payload_data", "{}")
                try:
                    payload = json.loads(payload_str)
                except json.JSONDecodeError:
                    payload = doc
            raw_id = doc.get("id", "")
            id_val = parse_id(raw_id.split(":", 1)[1] if ":" in str(raw_id) else raw_id)
            scored_results.append(
                ScoredResult(
                    id=id_val,
                    payload=payload,
                    score=float(doc.get("vector_distance", 0.0)),
                )
            )
        return scored_results

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: Optional[int],
        with_vectors: bool = False,
        include_payload: bool = False,
    ) -> list[list[ScoredResult]]:
        vectors = await self.embed_data(query_texts)
        search_tasks = [
            self.search(
                collection_name=collection_name,
                query_vector=vec,
                limit=limit,
                with_vector=with_vectors,
                include_payload=include_payload,
            )
            for vec in vectors
        ]
        return list(await asyncio.gather(*search_tasks))

    async def delete_data_points(
        self, collection_name: str, data_point_ids: List[str]
    ) -> dict[str, int]:
        index = self._get_index(collection_name)
        deleted_count = await index.drop_documents(data_point_ids)
        logger.info("Deleted %d data points from collection %s", deleted_count, collection_name)
        return {"deleted": deleted_count}

    async def prune(self) -> None:
        for collection_name, index in list(self._indices.items()):
            try:
                if await index.exists():
                    await index.delete(drop=True)
                    logger.info("Dropped index %s", collection_name)
            except Exception as e:
                logger.warning("Failed to drop index %s: %s", collection_name, e)
        self._indices.clear()
        logger.info("Pruned all Redis vector collections")
