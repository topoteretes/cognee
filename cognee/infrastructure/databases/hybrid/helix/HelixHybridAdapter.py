"""HelixDB unified graph + vector adapter.

Subclasses :class:`HelixGraphDB` and implements :class:`VectorDBInterface`, so a
single HelixDB instance backs both Cognee's graph and vector stores — mirroring
``NeptuneAnalyticsAdapter(NeptuneGraphDB, VectorDBInterface)``.

Unlike Neptune (which has one graph-global vector index and tags nodes with a
collection property), HelixDB has native per-``(label, property)`` vector indexes.
Each Cognee vector "collection" ``"{Type}_{field}"`` maps to a ``NodeVector`` index
on the shared ``COGNEE_NODE`` label keyed by a property named after the collection,
so the embedding lives on the *same* node that carries the graph edges — the
unification win. Embeddings are computed client-side by Cognee's embedding engine
(HelixDB v2 does not embed server-side).
"""

import asyncio
import json
from collections import Counter
from typing import Any, Dict, List, Optional
from uuid import UUID

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.exceptions import (
    MissingQueryParameterError,
    MutuallyExclusiveQueryParametersError,
)
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.modules.engine.utils.generate_edge_id import generate_edge_id
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text

from cognee.infrastructure.databases.graph.helix_driver.adapter import (
    HelixGraphDB,
    NODE_LABEL,
    ID_PROP,
    TENANT_PROP,
    DEFAULT_TENANT,
    VECTOR_PROP_PREFIX,
    _encode_scalar_props,
)
from cognee.infrastructure.databases.graph.helix_driver import ast
from cognee.infrastructure.databases.graph.helix_driver.client import HelixQueryError

logger = get_logger("HelixHybridAdapter")

DEFAULT_SEARCH_LIMIT = 15
# Over-fetch factor when a belongs_to_set filter is applied client-side.
_FILTER_OVERFETCH = 4


class IndexSchema(DataPoint):
    """Minimal data point used to index a single embeddable text field."""

    id: str
    text: str
    belongs_to_set: List[str] = []
    metadata: dict = {"index_fields": ["text"]}


class HelixHybridAdapter(HelixGraphDB, VectorDBInterface):
    """Unified HelixDB adapter implementing both graph and vector interfaces."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        tenant_id: Optional[str] = None,
        tenant_partitioned_vectors: bool = False,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, tenant_id=tenant_id)
        self.embedding_engine = embedding_engine
        self.tenant_partitioned_vectors = tenant_partitioned_vectors
        self._ensured_vector_indexes: set = set()
        self._vector_index_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _vec_prop(collection: str) -> str:
        """Node property name holding the vector for a collection."""
        return f"{VECTOR_PROP_PREFIX}{collection}"

    def _validate_embedding_engine(self) -> None:
        if self.embedding_engine is None:
            raise ValueError("HelixDB requires an embedding engine for vector operations")

    async def embed_data(self, data: List[str]) -> List[List[float]]:
        self._validate_embedding_engine()
        return await self.embedding_engine.embed_text(data)

    async def _ensure_vector_indexes(self, collections: List[str]) -> None:
        pending = [c for c in collections if c not in self._ensured_vector_indexes]
        if not pending:
            return
        async with self._vector_index_lock:
            pending = [c for c in pending if c not in self._ensured_vector_indexes]
            if not pending:
                return
            tenant_property = TENANT_PROP if self.tenant_partitioned_vectors else None
            queries = [
                ast.query_entry(
                    f"idx{i}",
                    [
                        ast.create_node_vector_index(
                            NODE_LABEL, self._vec_prop(collection), tenant_property
                        )
                    ],
                )
                for i, collection in enumerate(pending)
            ]
            await self._write(queries, [])
            self._ensured_vector_indexes.update(pending)

    def _vector_point_props(
        self, dp: DataPoint, vectors: Dict[str, List[float]]
    ) -> List[List[Any]]:
        """Encode a data point's scalar props + belongs_to_set + its vectors."""
        props = self._serialize_properties(dp.model_dump())
        props[ID_PROP] = str(dp.id)
        props[TENANT_PROP] = self.tenant_id
        props.pop("belongs_to_set", None)
        encoded = _encode_scalar_props(props)
        encoded.append(["belongs_to_set", ast.input_string_array(_belongs_to_set_names(dp))])
        for collection, vector in vectors.items():
            encoded.append([collection, ast.input_vector(vector)])
        return encoded

    # ------------------------------------------------------------------ #
    # VectorDBInterface: collections
    # ------------------------------------------------------------------ #

    async def has_collection(self, collection_name: str) -> bool:
        # Collections are implicit (a vector property + index on COGNEE_NODE).
        return collection_name in self._ensured_vector_indexes

    async def create_collection(self, collection_name: str, payload_schema: Optional[Any] = None):
        await self._ensure_vector_indexes([collection_name])

    async def create_vector_index(self, index_name: str, index_property_name: str):
        await self._ensure_vector_indexes([f"{index_name}_{index_property_name}"])

    async def get_collection(self, collection_name: str):
        return None

    async def get_connection(self):
        return None

    def get_data_point_schema(self, model_type: Any) -> Any:
        return model_type

    async def run_migrations(self):
        return None

    # ------------------------------------------------------------------ #
    # VectorDBInterface: data points
    # ------------------------------------------------------------------ #

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        self._validate_embedding_engine()
        valid = []
        for dp in data_points:
            text = DataPoint.get_embeddable_data(dp)
            if text is not None and text != "":
                valid.append((dp, text))
        if not valid:
            return
        await self._ensure_vector_indexes([collection_name])
        vectors = await self.embedding_engine.embed_text([text for _, text in valid])

        queries: List[Dict[str, Any]] = []
        for idx, ((dp, _), vector) in enumerate(zip(valid, vectors)):
            encoded = self._vector_point_props(dp, {self._vec_prop(collection_name): vector})
            queries.extend(self._upsert_node_queries(idx, str(dp.id), encoded))
        await self._write(queries, [])

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: List[DataPoint]
    ):
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=str(dp.id),
                    text=getattr(dp, dp.metadata["index_fields"][0]),
                    belongs_to_set=_belongs_to_set_names(dp),
                )
                for dp in data_points
            ],
        )

    async def retrieve(self, collection_name: str, data_point_ids: List[str]):
        nodes = await self.get_nodes([str(i) for i in data_point_ids])
        return [
            ScoredResult(id=_to_uuid(node.get(ID_PROP)), score=0, payload=node) for node in nodes
        ]

    async def delete_data_points(self, collection_name: str, data_point_ids: List[UUID]):
        await self.delete_nodes([str(i) for i in data_point_ids])

    async def prune(self):
        await self.delete_graph()

    # ------------------------------------------------------------------ #
    # VectorDBInterface: search
    # ------------------------------------------------------------------ #

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: Optional[int] = DEFAULT_SEARCH_LIMIT,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
    ) -> List[ScoredResult]:
        if query_text is not None and query_vector is not None:
            raise MutuallyExclusiveQueryParametersError()
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()
        if query_vector is None:
            query_vector = (await self.embed_data([query_text]))[0]

        k = limit if (limit and limit > 0) else DEFAULT_SEARCH_LIMIT
        # When the index is not tenant-partitioned, tenants share it; filtering by
        # tenant_id happens in Python because a Where step drops $distance. Project
        # the real `id` property ($id is the internal u64) plus distance + tenant.
        filter_tenant = not self.tenant_partitioned_vectors
        needs_overfetch = bool(node_name) or (filter_tenant and self.tenant_id != DEFAULT_TENANT)
        over_fetch = max(k * _FILTER_OVERFETCH, 50) if needs_overfetch else k
        tenant_value = self.tenant_id if self.tenant_partitioned_vectors else None

        projections = [ast.projection(ID_PROP, "id"), ast.projection("$distance", "distance")]
        if filter_tenant:
            projections.append(ast.projection(TENANT_PROP, TENANT_PROP))

        try:
            rows = await self._read(
                "hits",
                [
                    ast.vector_search_nodes(
                        NODE_LABEL,
                        self._vec_prop(collection_name),
                        query_vector,
                        over_fetch,
                        tenant_value,
                    ),
                    ast.project(projections),
                ],
            )
        except HelixQueryError as error:
            # A collection that was never written to has no vector index yet —
            # treat it as empty rather than surfacing a 500 (matches how other
            # adapters return [] for an unknown/empty collection).
            if "index not found" in str(error).lower():
                return []
            raise
        hits = []
        for row in rows:
            if not isinstance(row, dict) or row.get("id") is None:
                continue
            if filter_tenant and row.get(TENANT_PROP) != self.tenant_id:
                continue
            hits.append((row["id"], row.get("distance")))

        payloads: Dict[Any, dict] = {}
        if (include_payload or node_name) and hits:
            nodes = await self.get_nodes([h[0] for h in hits])
            payloads = {node.get(ID_PROP): node for node in nodes}

        results: List[ScoredResult] = []
        for node_id, distance in hits:
            payload = payloads.get(node_id)
            if node_name and not _matches_node_set(payload, node_name, node_name_filter_operator):
                continue
            results.append(
                ScoredResult(
                    id=_to_uuid(node_id),
                    score=float(distance or 0.0),
                    payload=payload if include_payload else None,
                )
            )
            if len(results) >= k:
                break
        return results

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: Optional[int] = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
    ):
        vectors = await self.embed_data(query_texts)
        return await asyncio.gather(
            *[
                self.search(
                    collection_name,
                    query_text=None,
                    query_vector=vector,
                    limit=limit,
                    with_vector=with_vectors,
                    include_payload=include_payload,
                    node_name=node_name,
                    node_name_filter_operator=node_name_filter_operator,
                )
                for vector in vectors
            ]
        )

    # ------------------------------------------------------------------ #
    # belongs_to_set cleanup (both GraphDBInterface + VectorDBInterface)
    # ------------------------------------------------------------------ #

    async def remove_belongs_to_set_tags(
        self, tags: List[str], node_ids: Optional[List[str]] = None
    ) -> None:
        tag_set = set(tags)
        scope_ids = {str(n) for n in node_ids} if node_ids else None
        nodes, _ = await self.get_graph_data()
        queries: List[Dict[str, Any]] = []
        for node_id, props in nodes:
            if scope_ids is not None and node_id not in scope_ids:
                continue
            current = _as_string_list(props.get("belongs_to_set"))
            if not tag_set.intersection(current):
                continue
            remaining = [t for t in current if t not in tag_set]
            if remaining:
                queries.append(
                    ast.query_entry(
                        None,
                        [
                            ast.n_where(self._tenant_scope(ast.eq(ID_PROP, node_id))),
                            ast.set_property("belongs_to_set", ast.input_string_array(remaining)),
                        ],
                    )
                )
            else:
                queries.append(
                    ast.query_entry(
                        None,
                        [ast.n_where(self._tenant_scope(ast.eq(ID_PROP, node_id))), ast.DROP],
                    )
                )
        if queries:
            await self._write(queries, [])

    # ------------------------------------------------------------------ #
    # Hybrid extensions (single-round-trip graph + vector writes)
    # ------------------------------------------------------------------ #

    async def add_nodes_with_vectors(self, data_points: List[DataPoint]) -> None:
        """Insert nodes and their embeddings in one write batch — each node
        carries its graph properties *and* every index-field vector at once.
        """
        if not data_points:
            return

        # Group embeddable fields into collections, embed each collection's texts.
        groups: Dict[str, List[DataPoint]] = {}
        for dp in data_points:
            if not getattr(dp, "metadata", None):
                continue
            for field_name in dp.metadata.get("index_fields", []):
                if getattr(dp, field_name, None) is None:
                    continue
                groups.setdefault(f"{type(dp).__name__}_{field_name}", []).append(dp)

        await self._ensure_vector_indexes(list(groups.keys()))

        vectors_by_collection: Dict[str, Dict[str, List[float]]] = {}
        for collection, points in groups.items():
            field = collection.rsplit("_", 1)[1]
            texts = [getattr(dp, field) for dp in points]
            texts = [t.strip() if isinstance(t, str) else t for t in texts]
            embedded = await self.embedding_engine.embed_text(texts)
            vectors_by_collection[collection] = {
                str(dp.id): vec for dp, vec in zip(points, embedded)
            }

        queries: List[Dict[str, Any]] = []
        for idx, dp in enumerate(data_points):
            node_vectors = {
                self._vec_prop(collection): vmap[str(dp.id)]
                for collection, vmap in vectors_by_collection.items()
                if str(dp.id) in vmap
            }
            encoded = self._vector_point_props(dp, node_vectors)
            queries.extend(self._upsert_node_queries(idx, str(dp.id), encoded))
        await self._write(queries, [])

    async def add_edges_with_vectors(self, edges: List) -> None:
        """Insert edges and index their relationship-type embeddings."""
        if not edges:
            return
        await self.add_edges(edges)

        edge_texts = []
        for edge in edges:
            props = edge[3] if len(edge) > 3 and edge[3] else {}
            edge_text = get_edge_retrieval_text(props.get("edge_text"), edge[2])
            if edge_text:
                edge_texts.append(edge_text)
        edge_type_counts = Counter(edge_texts)
        if not edge_type_counts:
            return

        await self.create_data_points(
            "EdgeType_relationship_name",
            [
                IndexSchema(id=str(generate_edge_id(edge_id=text)), text=text, belongs_to_set=[])
                for text in edge_type_counts
            ],
        )

    async def prune_all(self) -> None:
        await self.delete_graph()

    async def search_graph_with_distances(
        self,
        query_text: str,
        collection_name: str = "Entity_name",
        node_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Vector hits enriched with node info + distance (single search)."""
        results = await self.search(
            collection_name, query_text=query_text, limit=limit, include_payload=True
        )
        enriched = []
        for result in results:
            payload = result.payload or {}
            enriched.append(
                {
                    "node_id": str(result.id),
                    "node_name": payload.get("name"),
                    "node_type": payload.get("type"),
                    "relationship_name": None,
                    "distance": result.score,
                }
            )
        return enriched


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #


def _to_uuid(value: Any) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _belongs_to_set_names(dp: DataPoint) -> List[str]:
    items = getattr(dp, "belongs_to_set", None)
    if not items:
        return []
    names = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("id")
            if name is not None:
                names.append(str(name))
        elif hasattr(item, "name"):
            names.append(str(item.name))
    return names


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (ValueError, TypeError):
            pass
        return [value]
    return []


def _matches_node_set(payload: Optional[dict], node_name: List[str], operator: str) -> bool:
    belongs = set(_as_string_list((payload or {}).get("belongs_to_set")))
    if operator == "AND":
        return all(name in belongs for name in node_name)
    return any(name in belongs for name in node_name)
