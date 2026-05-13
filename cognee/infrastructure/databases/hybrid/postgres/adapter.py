"""Postgres hybrid adapter combining graph and vector functionality.

Composes a PostgresAdapter (graph) and PGVectorAdapter (vector) that
share the same Postgres database. Implements both GraphDBInterface and
VectorDBInterface by delegating to the underlying adapters.

Because both adapters hit the same database, this adapter can issue
combined SQL queries that JOIN graph and vector tables in a single
round-trip, and can perform graph+vector writes in a single transaction.
"""

import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Any, List, Union, Optional, Tuple, Type, TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.databases.vector.pgvector.serialize_data import serialize_data
from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import IndexSchema

if TYPE_CHECKING:
    from cognee.infrastructure.databases.graph.postgres.adapter import PostgresAdapter
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter
from cognee.modules.storage.utils import JSONEncoder
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.engine.utils.generate_edge_id import generate_edge_id

logger = get_logger()

# Regex for validating collection table names (alphanumeric + underscore only)
_SAFE_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_table_name(name: str) -> str:
    """Validate and quote a collection name for use as a SQL table identifier."""
    if not _SAFE_TABLE_NAME.match(name):
        raise ValueError(f"Invalid collection table name: {name!r}")
    return f'"{name}"'


class PostgresHybridAdapter(GraphDBInterface, VectorDBInterface):
    """Hybrid adapter backed by a single Postgres database.

    Holds a PostgresAdapter for graph operations and a PGVectorAdapter
    for vector operations. Both share the same underlying database, so
    combined queries can JOIN across graph_node/graph_edge and vector
    collection tables.
    """

    def __init__(
        self,
        graph_adapter: "PostgresAdapter",
        vector_adapter: "PGVectorAdapter",
    ) -> None:
        self._graph = graph_adapter
        self._vector = vector_adapter
        # Expose embedding_engine for callers that access it directly
        # (e.g. index_data_points checks vector_engine.embedding_engine)
        self.embedding_engine = vector_adapter.embedding_engine

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the graph adapter (vector adapter initializes lazily)."""
        await self._graph.initialize()

    # ------------------------------------------------------------------
    # GraphDBInterface: delegate to PostgresAdapter
    # ------------------------------------------------------------------

    async def query(self, query_str: str, params: Optional[dict] = None) -> List[Any]:
        return await self._graph.query(query_str, params)

    async def is_empty(self) -> bool:
        return await self._graph.is_empty()

    async def add_node(
        self, node: Union[DataPoint, str], properties: Optional[Dict[str, Any]] = None
    ) -> None:
        return await self._graph.add_node(node, properties)

    async def add_nodes(self, nodes: Union[List[Tuple[str, Dict]], List[DataPoint]]) -> None:
        return await self._graph.add_nodes(nodes)

    async def delete_node(self, node_id: str) -> None:
        return await self._graph.delete_node(node_id)

    async def delete_nodes(self, node_ids: List[str]) -> None:
        return await self._graph.delete_nodes(node_ids)

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return await self._graph.get_node(node_id)

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        return await self._graph.get_nodes(node_ids)

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        return await self._graph.add_edge(source_id, target_id, relationship_name, properties)

    async def add_edges(
        self, edges: Union[List[Tuple[str, str, str, Optional[Dict[str, Any]]]], List]
    ) -> None:
        return await self._graph.add_edges(edges)

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        return await self._graph.has_edge(source_id, target_id, relationship_name)

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        return await self._graph.has_edges(edges)

    async def get_edges(self, node_id: str) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        return await self._graph.get_edges(node_id)

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        return await self._graph.get_neighbors(node_id)

    async def get_connections(
        self, node_id: Union[str, UUID]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        return await self._graph.get_connections(node_id)

    async def get_graph_data(
        self,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        return await self._graph.get_graph_data()

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, str, str, Dict]]]:
        return await self._graph.get_filtered_graph_data(attribute_filters)

    async def get_nodeset_subgraph(
        self,
        node_type: Type[Any],
        node_name: List[str],
        node_name_filter_operator: str = "OR",
    ) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, str, str, dict]]]:
        return await self._graph.get_nodeset_subgraph(
            node_type, node_name, node_name_filter_operator
        )

    async def get_neighborhood(
        self,
        node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        return await self._graph.get_neighborhood(node_ids, depth, edge_types)

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        return await self._graph.get_graph_metrics(include_optional)

    async def delete_graph(self) -> None:
        return await self._graph.delete_graph()

    async def get_triplets_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        return await self._graph.get_triplets_batch(offset, limit)

    # ------------------------------------------------------------------
    # VectorDBInterface: delegate to PGVectorAdapter
    # ------------------------------------------------------------------

    async def embed_data(self, data: List[str]) -> List[List[float]]:
        return await self._vector.embed_data(data)

    async def has_collection(self, collection_name: str) -> bool:
        return await self._vector.has_collection(collection_name)

    async def create_collection(self, collection_name: str, payload_schema=None):
        return await self._vector.create_collection(collection_name, payload_schema)

    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        return await self._vector.create_data_points(collection_name, data_points)

    async def create_vector_index(self, index_name: str, index_property_name: str):
        return await self._vector.create_vector_index(index_name, index_property_name)

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: List[DataPoint]
    ):
        return await self._vector.index_data_points(index_name, index_property_name, data_points)

    async def retrieve(self, collection_name: str, data_point_ids: List[str]):
        return await self._vector.retrieve(collection_name, data_point_ids)

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
        return await self._vector.search(
            collection_name,
            query_text,
            query_vector,
            limit,
            with_vector,
            include_payload,
            node_name,
            node_name_filter_operator,
        )

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: Optional[int] = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
    ):
        return await self._vector.batch_search(
            collection_name,
            query_texts,
            limit,
            with_vectors,
            include_payload,
            node_name,
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: List[UUID]):
        return await self._vector.delete_data_points(collection_name, data_point_ids)

    async def prune(self):
        return await self._vector.prune()

    # ------------------------------------------------------------------
    # Hybrid: combined graph+vector writes
    # ------------------------------------------------------------------

    async def add_nodes_with_vectors(self, data_points: List[DataPoint]) -> None:
        """Insert nodes into graph and their embeddings into vector tables
        in a single database transaction.

        All graph node rows are inserted in one batched statement, and each
        vector collection's rows are inserted in one batched statement per
        collection. This keeps the transaction short (one statement per table)
        and avoids deadlocks when multiple concurrent callers write to the
        same tables.
        """
        if not data_points:
            return

        await self._graph.initialize()
        now = datetime.now(timezone.utc)

        # Group data points by (type_name, field_name) for vector indexing
        vector_groups: Dict[str, List[Tuple[DataPoint, str]]] = {}
        for dp in data_points:
            for field_name in dp.metadata.get("index_fields", []):
                if getattr(dp, field_name, None) is None:
                    continue
                collection = f"{type(dp).__name__}_{field_name}"
                vector_groups.setdefault(collection, [])
                vector_groups[collection].append((dp, field_name))

        # Embed all texts grouped by collection
        embeddings_by_collection: Dict[str, List[Tuple[DataPoint, List[float], str]]] = {}
        for collection, items in vector_groups.items():
            valid_items = [(dp, getattr(dp, field_name, None)) for dp, field_name in items]
            valid_items = [(dp, t.strip() if isinstance(t, str) else t) for dp, t in valid_items]
            valid_items = [(dp, t) for dp, t in valid_items if t is not None]
            if not valid_items:
                continue
            texts = [t for _, t in valid_items]
            vectors = await self._vector.embed_data(texts)
            embeddings_by_collection[collection] = [
                (dp, vec, t) for (dp, t), vec in zip(valid_items, vectors)
            ]

        # Ensure vector collection tables exist
        for collection in vector_groups:
            await self._vector.create_vector_index(
                collection.rsplit("_", 1)[0], collection.rsplit("_", 1)[1]
            )

        # Build all rows in Python, then one INSERT per table inside the transaction.
        core_keys = {"id", "name", "type"}
        node_rows = []
        for dp in data_points:
            props = dp.model_dump() if hasattr(dp, "model_dump") else vars(dp)
            extra = {k: v for k, v in props.items() if k not in core_keys}
            node_rows.append(
                {
                    "id": str(props.get("id", "")),
                    "name": str(props.get("name", "")),
                    "type": str(props.get("type", "")),
                    "properties": json.dumps(extra, cls=JSONEncoder),
                    "now": now,
                }
            )

        vector_rows_by_table: Dict[str, List[Dict]] = {}
        for collection, items in embeddings_by_collection.items():
            table = _validate_table_name(collection)
            rows = []
            for dp, vector, embed_text in items:
                index_point = IndexSchema(
                    id=dp.id,
                    text=embed_text,
                    belongs_to_set=(dp.belongs_to_set or []),
                )
                payload = serialize_data(index_point.model_dump())
                rows.append(
                    {
                        "id": str(dp.id),
                        "payload": json.dumps(payload),
                        "vector": str(vector),
                    }
                )
            vector_rows_by_table[table] = rows

        # Single transaction: one batched INSERT per table
        async with self._graph._session() as session:
            if node_rows:
                await session.execute(
                    text("""
                        INSERT INTO graph_node (id, name, type, properties, created_at, updated_at)
                        VALUES (:id, :name, :type, CAST(:properties AS jsonb), :now, :now)
                        ON CONFLICT (id) DO UPDATE SET
                            name = EXCLUDED.name,
                            type = EXCLUDED.type,
                            properties = EXCLUDED.properties,
                            updated_at = EXCLUDED.updated_at
                    """),
                    node_rows,
                )

            for table, rows in vector_rows_by_table.items():
                if rows:
                    await session.execute(
                        text(f"""
                            INSERT INTO {table} (id, payload, vector)
                            VALUES (:id, CAST(:payload AS json), :vector)
                            ON CONFLICT (id) DO UPDATE SET
                                payload = EXCLUDED.payload,
                                vector = EXCLUDED.vector
                        """),
                        rows,
                    )

            await session.commit()

    async def add_edges_with_vectors(
        self, edges: List[Tuple[str, str, str, Dict[str, Any]]]
    ) -> None:
        """Insert edges into graph and their type embeddings into vector
        tables in a single database transaction.

        All graph edge rows are inserted in one batched statement, and
        edge type vector rows in one batched statement. This keeps the
        transaction short and avoids deadlocks.
        """
        if not edges:
            return

        await self._graph.initialize()
        now = datetime.now(timezone.utc)

        # Collect edge type counts for EdgeType vector indexing
        edge_texts = []
        for edge in edges:
            props = edge[3] if len(edge) > 3 and edge[3] else {}
            edge_text = props.get("edge_text", edge[2])
            edge_texts.append(edge_text)

        edge_type_counts = Counter(edge_texts)

        # Embed unique edge types
        unique_texts = list(edge_type_counts.keys())
        if unique_texts:
            unique_vectors = await self._vector.embed_data(unique_texts)
            text_to_vector = dict(zip(unique_texts, unique_vectors))
        else:
            text_to_vector = {}

        # Ensure edge type collection exists
        collection = "EdgeType_relationship_name"
        await self._vector.create_vector_index("EdgeType", "relationship_name")

        # Build all rows in Python, then one INSERT per table.
        edge_rows = []
        for edge in edges:
            source_id, target_id, rel_name = str(edge[0]), str(edge[1]), edge[2]
            props = edge[3] if len(edge) > 3 and edge[3] else {}
            props_json = json.dumps(props, cls=JSONEncoder)
            edge_rows.append(
                {
                    "src": source_id,
                    "tgt": target_id,
                    "rel": rel_name,
                    "props": props_json,
                    "now": now,
                }
            )

        vector_rows = []
        table = _validate_table_name(collection)
        for edge_text, count in edge_type_counts.items():
            edge_id = generate_edge_id(edge_id=edge_text)
            vector = text_to_vector.get(edge_text)
            if vector is None:
                continue
            edge_type_dp = EdgeType(
                id=edge_id,
                relationship_name=edge_text,
                number_of_edges=count,
            )
            index_point = IndexSchema(
                id=edge_id,
                text=edge_text,
                belongs_to_set=(edge_type_dp.belongs_to_set or []),
            )
            payload = json.dumps(serialize_data(index_point.model_dump()))
            vector_rows.append(
                {
                    "id": str(edge_id),
                    "payload": payload,
                    "vector": str(vector),
                }
            )

        # Single transaction: one batched INSERT per table
        async with self._graph._session() as session:
            if edge_rows:
                await session.execute(
                    text("""
                        INSERT INTO graph_edge
                            (source_id, target_id, relationship_name, properties,
                             created_at, updated_at)
                        VALUES (:src, :tgt, :rel, CAST(:props AS jsonb), :now, :now)
                        ON CONFLICT (source_id, target_id, relationship_name) DO UPDATE SET
                            properties = EXCLUDED.properties,
                            updated_at = EXCLUDED.updated_at
                    """),
                    edge_rows,
                )

            if vector_rows:
                await session.execute(
                    text(f"""
                        INSERT INTO {table} (id, payload, vector)
                        VALUES (:id, CAST(:payload AS json), :vector)
                        ON CONFLICT (id) DO UPDATE SET
                            payload = EXCLUDED.payload,
                            vector = EXCLUDED.vector
                    """),
                    vector_rows,
                )

            await session.commit()

    async def delete_nodes_with_vectors(
        self,
        node_ids: List[str],
        collections: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Delete nodes from graph and optionally their embeddings from
        vector tables in a single database transaction.

        Args:
            node_ids: IDs of nodes to delete from graph_node.
            collections: Mapping of collection_name -> list of IDs to delete
                from that collection. If None, only graph rows are deleted;
                callers are responsible for supplying collection info.
        """
        if not node_ids:
            return

        await self._graph.initialize()

        async with self._graph._session() as session:
            # Delete from graph (CASCADE removes edges)
            await session.execute(
                text("DELETE FROM graph_node WHERE id = ANY(:ids)"),
                {"ids": node_ids},
            )

            # Delete from vector collections (caller must specify which)
            if collections:
                for collection_name, ids in collections.items():
                    if not ids:
                        continue
                    table = _validate_table_name(collection_name)
                    await session.execute(
                        text(f"DELETE FROM {table} WHERE CAST(id AS text) = ANY(:ids)"),
                        {"ids": ids},
                    )

            await session.commit()

    async def delete_edges_with_vectors(
        self,
        edge_type_ids: Optional[List[str]] = None,
        triplet_ids: Optional[List[str]] = None,
    ) -> None:
        """Delete edge type and triplet embeddings from vector tables.

        Graph edges are deleted via CASCADE when their nodes are deleted,
        so this only handles the vector side.

        Args:
            edge_type_ids: IDs to delete from EdgeType_relationship_name.
            triplet_ids: IDs to delete from Triplet_text.
        """
        async with self._graph._session() as session:
            if edge_type_ids:
                table = _validate_table_name("EdgeType_relationship_name")
                await session.execute(
                    text(f"DELETE FROM {table} WHERE CAST(id AS text) = ANY(:ids)"),
                    {"ids": edge_type_ids},
                )

            if triplet_ids:
                try:
                    table = _validate_table_name("Triplet_text")
                    await session.execute(
                        text(f"DELETE FROM {table} WHERE CAST(id AS text) = ANY(:ids)"),
                        {"ids": triplet_ids},
                    )
                except Exception as e:
                    error_msg = str(e).lower()
                    if "does not exist" in error_msg or "relation" in error_msg:
                        logger.debug("Triplet_text table not found, skipping: %s", e)
                    else:
                        logger.warning("Unexpected error deleting from Triplet_text: %s", e)

            await session.commit()

    async def prune_all(self) -> None:
        """Truncate all graph tables and drop all vector collections."""
        await self._graph.delete_graph()
        await self._vector.prune()

    # ------------------------------------------------------------------
    # Hybrid: combined graph+vector search
    # ------------------------------------------------------------------

    async def search_graph_with_distances(
        self,
        query_text: str,
        collection_name: str = "Entity_name",
        node_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Combined graph traversal + vector similarity in a single SQL query.

        Finds neighbors of a node (or all nodes if node_id is None) and
        ranks them by embedding similarity to the query text. Collapses
        the separate vector-search + graph-traversal pattern into one
        database round-trip.

        Returns a list of dicts with keys:
            node_id, node_name, node_type, relationship_name, distance
        """
        query_vector = (await self._vector.embed_data([query_text]))[0]
        table = _validate_table_name(collection_name)

        if node_id is not None:
            # Neighbors of a specific node, ranked by vector distance
            sql = text(f"""
                SELECT
                    gn.id AS node_id,
                    gn.name AS node_name,
                    gn.type AS node_type,
                    ge.relationship_name,
                    vc.vector <=> CAST(:query_vector AS vector) AS distance
                FROM graph_edge ge
                JOIN graph_node gn ON gn.id = CASE
                    WHEN ge.source_id = :node_id THEN ge.target_id
                    ELSE ge.source_id
                END
                JOIN {table} vc ON CAST(vc.id AS text) = gn.id
                WHERE ge.source_id = :node_id OR ge.target_id = :node_id
                ORDER BY distance
                LIMIT :limit
            """)
            params = {
                "query_vector": str(query_vector),
                "node_id": node_id,
                "limit": limit,
            }
        else:
            # Top-k distinct nodes by vector distance, then join edge context
            sql = text(f"""
                WITH top_nodes AS (
                    SELECT
                        gn.id AS node_id,
                        gn.name AS node_name,
                        gn.type AS node_type,
                        vc.vector <=> CAST(:query_vector AS vector) AS distance
                    FROM graph_node gn
                    JOIN {table} vc ON CAST(vc.id AS text) = gn.id
                    ORDER BY distance
                    LIMIT :limit
                )
                SELECT
                    tn.node_id,
                    tn.node_name,
                    tn.node_type,
                    ge.relationship_name,
                    ge.source_id,
                    ge.target_id,
                    tn.distance
                FROM top_nodes tn
                LEFT JOIN graph_edge ge ON ge.source_id = tn.node_id
                    OR ge.target_id = tn.node_id
                ORDER BY tn.distance
            """)
            params = {
                "query_vector": str(query_vector),
                "limit": limit,
            }

        async with self._graph._session() as session:
            result = await session.execute(sql, params)
            rows = result.mappings().fetchall()

        return [
            {
                "node_id": row["node_id"],
                "node_name": row["node_name"],
                "node_type": row["node_type"],
                "relationship_name": row["relationship_name"],
                "distance": row["distance"],
            }
            for row in rows
        ]
