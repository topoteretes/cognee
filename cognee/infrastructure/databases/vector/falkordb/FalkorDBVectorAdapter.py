"""FalkorDB Vector Database Adapter for Cognee.

This adapter stores vector embeddings as node properties within FalkorDB,
enabling unified graph and vector storage in the same database.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface

logger = get_logger("FalkorDBVectorAdapter")

VECTOR_PROPERTY = "embedding"
TEXT_PROPERTY = "text"


class FalkorDBJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles UUID and datetime objects."""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def _sanitize_graph_name(raw: Optional[str]) -> Optional[str]:
    """Sanitize graph name for FalkorDB compatibility."""
    if not raw:
        return None
    return raw.replace(" ", "_").replace("'", "").replace("-", "_")


class FalkorDBVectorAdapter(VectorDBInterface):
    """
    FalkorDB vector database adapter implementing VectorDBInterface.

    Features:
    - Store vector embeddings in the same FalkorDB graph as graph data
    - Per-agent isolation via context variable
    - Cosine similarity search using FalkorDB vector indices
    """

    name = "FalkorDB"

    def __init__(
        self,
        url: str,
        port: int = 6379,
        api_key: Optional[str] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        graph_name: str = "CogneeGraph",
        **kwargs,
    ):
        """Initialize FalkorDB vector adapter.

        Args:
            url: FalkorDB host URL
            port: FalkorDB port (default: 6379)
            api_key: Optional password for authentication
            embedding_engine: Engine for generating embeddings
            graph_name: Default graph name (default: 'CogneeGraph')
        """
        self.host = url.replace("redis://", "").split(":")[0] if "redis://" in url else url
        self.port = int(port) if port else 6379
        self.password = api_key
        self._default_graph_name = graph_name
        self.embedding_engine = embedding_engine

        self.client = None
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._indices_created: set = set()

    def _get_graph_name_from_ctx(self) -> str:
        """Get graph name from context variable or fall back to default."""
        from cognee.context_global_variables import agent_graph_name_ctx

        ctx_name = _sanitize_graph_name(agent_graph_name_ctx.get())
        default_name = _sanitize_graph_name(self._default_graph_name) or "CogneeGraph"
        return ctx_name or default_name

    def _connect_sync(self) -> None:
        """Synchronous connection to FalkorDB."""
        if self.client:
            return

        try:
            from falkordb import FalkorDB
        except ImportError:
            raise ImportError(
                "FalkorDB is not installed. Please install with 'pip install cognee[falkordb]'"
            )

        kwargs: Dict[str, Any] = {"host": self.host, "port": self.port}
        if self.password:
            kwargs["password"] = self.password
        self.client = FalkorDB(**kwargs)

    async def _ensure_connected(self) -> None:
        """Ensure connection is established."""
        if self.client:
            return
        await asyncio.get_running_loop().run_in_executor(self._executor, self._connect_sync)

    async def _query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Execute a query against FalkorDB."""
        await self._ensure_connected()
        graph_name = self._get_graph_name_from_ctx()
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._query_sync, query, params or {}, graph_name
        )

    def _query_sync(
        self, query: str, params: Dict[str, Any], graph_name: str
    ) -> List[Any]:
        """Synchronous query execution."""
        if not self.client:
            self._connect_sync()
        assert self.client is not None

        graph = self.client.select_graph(graph_name)
        res = graph.query(query, params)

        data: List[Any] = []
        if not getattr(res, "result_set", None):
            return data

        header = getattr(res, "header", None)
        for record in res.result_set:
            if header:
                row: Dict[str, Any] = {}
                for i, col_def in enumerate(header):
                    if isinstance(col_def, (list, tuple)) and len(col_def) >= 2:
                        col_name = col_def[1] if isinstance(col_def[0], int) else col_def[0]
                    else:
                        col_name = str(col_def)
                    row[col_name] = record[i]
                data.append(row)
            else:
                data.append(record)
        return data

    async def has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists (always returns False to force creation)."""
        return False

    async def create_collection(
        self, collection_name: str, payload_schema: Optional[Any] = None
    ):
        """Create a vector index for the collection."""
        label = collection_name.replace("-", "_")
        vector_size = self.embedding_engine.get_vector_size() if self.embedding_engine else 384

        graph_name = self._get_graph_name_from_ctx()
        logger.debug(
            "FalkorDBVectorAdapter.create_collection graph=%s collection=%s",
            graph_name,
            label,
        )

        if (graph_name, label) in self._indices_created:
            return

        query = (
            f"CREATE VECTOR INDEX FOR (n:`{label}`) "
            f"ON (n.{VECTOR_PROPERTY}) "
            f"OPTIONS {{dimension: {vector_size}, similarityFunction: 'cosine'}}"
        )
        try:
            await self._query(query)
        except Exception as e:
            msg = str(e).lower()
            if "already exists" not in msg and "already indexed" not in msg:
                logger.warning(
                    "Vector index create failed for graph=%s label=%s: %s",
                    graph_name,
                    label,
                    e,
                )
        self._indices_created.add((graph_name, label))

    async def create_data_points(
        self, collection_name: str, data_points: List[DataPoint]
    ):
        """Insert data points with their embeddings."""
        label = collection_name.replace("-", "_")
        logger.debug(
            "FalkorDBVectorAdapter.create_data_points graph=%s collection=%s count=%s",
            self._get_graph_name_from_ctx(),
            label,
            len(data_points),
        )

        texts: List[str] = []
        for dp in data_points:
            try:
                embeddable = dp.get_embeddable_data(dp)
            except Exception:
                embeddable = None
            texts.append(str(embeddable) if embeddable is not None else str(dp))

        embeddings = (
            await self.embedding_engine.embed_text(texts) if self.embedding_engine else []
        )

        for dp, text, emb in zip(data_points, texts, embeddings):
            payload = dp.model_dump() if hasattr(dp, "model_dump") else {}
            if TEXT_PROPERTY not in payload:
                payload[TEXT_PROPERTY] = text
            emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)

            query = (
                f"MERGE (n:`{label}` {{id: $id}}) "
                f"SET n.{TEXT_PROPERTY} = $text, "
                f"n.{VECTOR_PROPERTY} = vecf32($emb), "
                f"n.payload = $payload"
            )
            await self._query(
                query,
                {
                    "id": str(dp.id),
                    "text": text,
                    "emb": emb_list,
                    "payload": json.dumps(payload, cls=FalkorDBJSONEncoder),
                },
            )

    async def retrieve(
        self, collection_name: str, data_point_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Retrieve data points by IDs."""
        label = collection_name.replace("-", "_")
        query = f"MATCH (n:`{label}`) WHERE n.id IN $ids RETURN n.id as id, n.payload as payload"
        results = await self._query(query, {"ids": data_point_ids})

        out: List[Dict[str, Any]] = []
        for r in results:
            payload = (
                json.loads(r.get("payload", "{}"))
                if isinstance(r.get("payload"), str)
                else {}
            )
            out.append({"id": r.get("id"), "payload": payload})
        return out

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: Optional[int] = 10,
        with_vector: bool = False,
        **kwargs,
    ) -> List[ScoredResult]:
        """Search for similar vectors."""
        label = collection_name.replace("-", "_")
        logger.debug(
            "FalkorDBVectorAdapter.search graph=%s collection=%s limit=%s",
            self._get_graph_name_from_ctx(),
            label,
            limit,
        )

        # Handle dimension mismatch
        if self.embedding_engine:
            expected_dim = self.embedding_engine.get_vector_size()
            if query_vector is not None and len(query_vector) != expected_dim:
                if query_text:
                    query_vector = (await self.embedding_engine.embed_text([query_text]))[0]
                else:
                    logger.warning(
                        "Vector dim mismatch (got=%s expected=%s); returning empty.",
                        len(query_vector),
                        expected_dim,
                    )
                    return []

        if query_vector is None and query_text and self.embedding_engine:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]
        if query_vector is None:
            return []

        emb_list = (
            query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        )
        query = (
            f"CALL db.idx.vector.queryNodes('{label}', '{VECTOR_PROPERTY}', {int(limit or 10)}, "
            f"vecf32($emb)) "
            f"YIELD node, score "
            f"RETURN node.id as id, node.payload as payload, score"
        )

        try:
            results = await self._query(query, {"emb": emb_list})
        except Exception as e:
            msg = str(e)
            if "Invalid arguments for procedure 'db.idx.vector.queryNodes'" in msg:
                logger.warning(
                    "Vector queryNodes failed for graph=%s collection=%s; returning empty.",
                    self._get_graph_name_from_ctx(),
                    label,
                )
                return []
            if "Attempted to access undefined attribute" in msg:
                return []
            raise

        scored: List[ScoredResult] = []
        for r in results:
            if not r.get("id"):
                continue
            payload = (
                json.loads(r.get("payload", "{}"))
                if isinstance(r.get("payload"), str)
                else {}
            )
            scored.append(
                ScoredResult(id=r["id"], payload=payload, score=float(r.get("score") or 0.0))
            )
        return scored

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: Optional[int],
        with_vectors: bool = False,
    ) -> List[List[ScoredResult]]:
        """Perform batch search across multiple queries."""
        return [
            await self.search(collection_name, query_text=t, limit=limit)
            for t in query_texts
        ]

    async def delete_data_points(
        self, collection_name: str, data_point_ids: List[str]
    ):
        """Delete data points by IDs."""
        label = collection_name.replace("-", "_")
        query = f"MATCH (n:`{label}`) WHERE n.id IN $ids DETACH DELETE n"
        await self._query(query, {"ids": data_point_ids})

    async def prune(self):
        """Delete all data in the graph."""
        await self._query("MATCH (n) DETACH DELETE n")

    async def embed_data(self, data: List[str]) -> List[List[float]]:
        """Embed text data into vectors."""
        if not self.embedding_engine:
            return []
        embeddings = await self.embedding_engine.embed_text(data)
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """Create a vector index for the given index name and property."""
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: List[DataPoint]
    ):
        """Index data points under the given index."""
        await self.create_data_points(f"{index_name}_{index_property_name}", data_points)
