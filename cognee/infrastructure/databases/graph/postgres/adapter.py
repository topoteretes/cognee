"""Postgres graph adapter using two tables (graph_node, graph_edge) over SQLAlchemy + asyncpg."""

import json
import asyncio
from uuid import UUID
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Union, Optional, Tuple, Type

from sqlalchemy import text

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.storage.utils import JSONEncoder

logger = get_logger()

# Schema DDL executed once on initialise()
_SCHEMA_DDL = [
    """
    CREATE TABLE IF NOT EXISTS graph_node (
        id          TEXT PRIMARY KEY,
        name        TEXT,
        type        TEXT,
        properties  JSONB,
        created_at  TIMESTAMPTZ DEFAULT now(),
        updated_at  TIMESTAMPTZ DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_edge (
        source_id         TEXT NOT NULL REFERENCES graph_node(id) ON DELETE CASCADE,
        target_id         TEXT NOT NULL REFERENCES graph_node(id) ON DELETE CASCADE,
        relationship_name TEXT NOT NULL,
        properties        JSONB,
        created_at        TIMESTAMPTZ DEFAULT now(),
        updated_at        TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (source_id, target_id, relationship_name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edge_source ON graph_edge(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_edge_target ON graph_edge(target_id)",
    "CREATE INDEX IF NOT EXISTS idx_node_type   ON graph_node(type)",
    # Covering index: neighbor lookups without heap reads
    """
    CREATE INDEX IF NOT EXISTS idx_edge_source_cover
    ON graph_edge(source_id) INCLUDE (target_id, relationship_name)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_edge_target_cover
    ON graph_edge(target_id) INCLUDE (source_id, relationship_name)
    """,
]


class PostgresAdapter(GraphDBInterface):
    """Graph-as-tables adapter backed by Postgres, accessed via SQLAlchemy async sessions."""

    def __init__(self, relational_engine):
        """Accept an existing SQLAlchemyAdapter (shared with the relational layer)."""
        self.engine = relational_engine

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        async with self._session() as session:
            for ddl in _SCHEMA_DDL:
                await session.execute(text(ddl))
            await session.commit()

    async def _ensure_initialized(self) -> None:
        """Re-run initialize() if tables were dropped (e.g. by prune_system).

        Uses CREATE IF NOT EXISTS, so this is cheap and idempotent.
        """
        await self.initialize()

    @asynccontextmanager
    async def _session(self):
        """Yield an async session from the underlying engine."""
        async with self.engine.get_async_session() as session:
            yield session

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _serialize_properties(self, props: Dict[str, Any]) -> str:
        """Serialize a dict to a JSON string, handling datetimes and UUIDs."""
        return json.dumps(props, cls=JSONEncoder)

    def _split_core_and_extra(self, properties: Dict[str, Any]) -> Tuple[dict, str]:
        """Separate core columns (id, name, type) from the rest, JSON-encode the rest."""
        core = {
            "id": str(properties.get("id", "")),
            "name": str(properties.get("name", "")),
            "type": str(properties.get("type", "")),
        }
        extra = {k: v for k, v in properties.items() if k not in core}
        return core, self._serialize_properties(extra)

    def _parse_node_row(self, row) -> Dict[str, Any]:
        """Convert a (id, name, type, properties) row to a merged dict."""
        data = {"id": row.id, "name": row.name, "type": row.type}
        if row.properties:
            props = row.properties if isinstance(row.properties, dict) else json.loads(row.properties)
            data.update(props)
        return data

    # ------------------------------------------------------------------
    # GraphDBInterface: query
    # ------------------------------------------------------------------

    async def query(self, query_str: str, params: Optional[dict] = None) -> List[Any]:
        """Raw Cypher queries are not supported on the Postgres graph backend.

        All graph operations should use the typed adapter methods (add_nodes,
        get_neighbors, get_connections, etc.). If you need raw Cypher support,
        use a graph-native backend (Neo4j, Kuzu).
        """
        raise NotImplementedError(
            "The Postgres graph backend does not support raw Cypher queries. "
            "Use a graph-native backend (Neo4j, Kuzu) for raw query support, "
            "or use the typed adapter methods (add_nodes, get_neighbors, etc.)."
        )

    # ------------------------------------------------------------------
    # GraphDBInterface: node operations
    # ------------------------------------------------------------------

    async def is_empty(self) -> bool:
        await self._ensure_initialized()
        async with self._session() as session:
            result = await session.execute(
                text("SELECT EXISTS(SELECT 1 FROM graph_node LIMIT 1)")
            )
            return not result.scalar()

    async def add_node(self, node: Union[DataPoint, str], properties: Optional[Dict[str, Any]] = None) -> None:
        """Wrapper: add a single node via add_nodes."""
        if isinstance(node, str):
            props = properties or {}
            props.setdefault("id", node)
            # Pack as a DataPoint-like object would be, using a tuple
            await self.add_nodes([(node, props)])
        else:
            await self.add_nodes([node])

    async def add_nodes(self, nodes: Union[List[Tuple[str, Dict]], List[DataPoint]]) -> None:
        if not nodes:
            return

        now = datetime.now(timezone.utc)

        # Normalize all inputs to (core_dict, extra_json) pairs
        rows = []
        for node in nodes:
            if isinstance(node, tuple):
                props = {"id": node[0], **(node[1] or {})}
            elif hasattr(node, "model_dump"):
                props = node.model_dump()
            else:
                props = vars(node)
            core, extra_json = self._split_core_and_extra(props)
            rows.append({**core, "properties": extra_json, "now": now})

        # Single session, single commit for the whole batch
        async with self._session() as session:
            for row in rows:
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
                    row,
                )
            await session.commit()

    async def delete_node(self, node_id: str) -> None:
        """Wrapper: delete a single node via delete_nodes."""
        await self.delete_nodes([node_id])

    async def delete_nodes(self, node_ids: List[str]) -> None:
        if not node_ids:
            return
        async with self._session() as session:
            await session.execute(
                text("DELETE FROM graph_node WHERE id = ANY(:ids)"), {"ids": node_ids}
            )
            await session.commit()

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Wrapper: get a single node via get_nodes."""
        results = await self.get_nodes([node_id])
        return results[0] if results else None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        if not node_ids:
            return []
        async with self._session() as session:
            result = await session.execute(
                text("SELECT id, name, type, properties FROM graph_node WHERE id = ANY(:ids)"),
                {"ids": node_ids},
            )
            return [self._parse_node_row(row) for row in result.fetchall()]

    # ------------------------------------------------------------------
    # GraphDBInterface: edge operations
    # ------------------------------------------------------------------

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Wrapper: add a single edge via add_edges."""
        await self.add_edges([(str(source_id), str(target_id), relationship_name, properties or {})])

    async def add_edges(
        self, edges: Union[List[Tuple[str, str, str, Optional[Dict[str, Any]]]], List]
    ) -> None:
        if not edges:
            return

        now = datetime.now(timezone.utc)

        async with self._session() as session:
            for edge in edges:
                source_id, target_id, rel_name = str(edge[0]), str(edge[1]), edge[2]
                props = edge[3] if len(edge) > 3 and edge[3] else {}
                props_json = self._serialize_properties(props)

                await session.execute(
                    text("""
                        INSERT INTO graph_edge
                            (source_id, target_id, relationship_name, properties, created_at, updated_at)
                        VALUES (:src, :tgt, :rel, CAST(:props AS jsonb), :now, :now)
                        ON CONFLICT (source_id, target_id, relationship_name) DO UPDATE SET
                            properties = EXCLUDED.properties,
                            updated_at = EXCLUDED.updated_at
                    """),
                    {"src": source_id, "tgt": target_id, "rel": rel_name,
                     "props": props_json, "now": now},
                )
            await session.commit()

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """Wrapper: check a single edge via has_edges."""
        result = await self.has_edges([(str(source_id), str(target_id), relationship_name)])
        return len(result) > 0

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        if not edges:
            return []

        existing = []
        async with self._session() as session:
            for src, tgt, rel in edges:
                result = await session.execute(
                    text("""
                        SELECT EXISTS(
                            SELECT 1 FROM graph_edge
                            WHERE source_id = :src AND target_id = :tgt
                              AND relationship_name = :rel
                        )
                    """),
                    {"src": str(src), "tgt": str(tgt), "rel": str(rel)},
                )
                if result.scalar():
                    existing.append((str(src), str(tgt), str(rel)))
        return existing

    async def get_edges(self, node_id: str) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        async with self._session() as session:
            result = await session.execute(
                text("""
                    SELECT
                        n.id, n.name, n.type, n.properties,
                        e.relationship_name,
                        m.id, m.name, m.type, m.properties
                    FROM graph_edge e
                    JOIN graph_node n ON n.id = e.source_id
                    JOIN graph_node m ON m.id = e.target_id
                    WHERE e.source_id = :nid OR e.target_id = :nid
                """),
                {"nid": node_id},
            )
            edges = []
            for row in result.fetchall():
                src = {"id": row[0], "name": row[1], "type": row[2]}
                if row[3]:
                    src.update(row[3] if isinstance(row[3], dict) else json.loads(row[3]))
                tgt = {"id": row[5], "name": row[6], "type": row[7]}
                if row[8]:
                    tgt.update(row[8] if isinstance(row[8], dict) else json.loads(row[8]))
                edges.append((src, row[4], tgt))
            return edges

    # ------------------------------------------------------------------
    # GraphDBInterface: neighbor and connection queries
    # ------------------------------------------------------------------

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        async with self._session() as session:
            result = await session.execute(
                text("""
                    SELECT DISTINCT m.id, m.name, m.type, m.properties
                    FROM graph_edge e
                    JOIN graph_node m ON m.id = CASE
                        WHEN e.source_id = :nid THEN e.target_id
                        ELSE e.source_id
                    END
                    WHERE e.source_id = :nid OR e.target_id = :nid
                """),
                {"nid": node_id},
            )
            return [self._parse_node_row(row) for row in result.fetchall()]

    async def get_connections(
        self, node_id: Union[str, UUID]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        nid = str(node_id)

        async with self._session() as session:
            result = await session.execute(
                text("""
                    SELECT
                        n.id, n.name, n.type, n.properties,
                        e.relationship_name, e.properties AS edge_props,
                        m.id, m.name, m.type, m.properties
                    FROM graph_edge e
                    JOIN graph_node n ON n.id = e.source_id
                    JOIN graph_node m ON m.id = e.target_id
                    WHERE e.source_id = :nid OR e.target_id = :nid
                """),
                {"nid": nid},
            )

            connections = []
            for row in result.fetchall():
                src = {"id": row[0], "name": row[1], "type": row[2]}
                if row[3]:
                    src.update(row[3] if isinstance(row[3], dict) else json.loads(row[3]))

                edge = {"relationship_name": row[4]}
                if row[5]:
                    edge_props = row[5] if isinstance(row[5], dict) else json.loads(row[5])
                    edge.update(edge_props)

                tgt = {"id": row[6], "name": row[7], "type": row[8]}
                if row[9]:
                    tgt.update(row[9] if isinstance(row[9], dict) else json.loads(row[9]))

                connections.append((src, edge, tgt))
            return connections

    # ------------------------------------------------------------------
    # GraphDBInterface: graph-wide reads
    # ------------------------------------------------------------------

    async def get_graph_data(
        self,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        async with self._session() as session:
            # Fetch nodes
            node_result = await session.execute(
                text("SELECT id, name, type, properties FROM graph_node")
            )
            nodes = []
            for row in node_result.fetchall():
                data = {"name": row[1], "type": row[2]}
                if row[3]:
                    data.update(row[3] if isinstance(row[3], dict) else json.loads(row[3]))
                nodes.append((row[0], data))

            if not nodes:
                return [], []

            # Fetch edges
            edge_result = await session.execute(
                text("""
                    SELECT source_id, target_id, relationship_name, properties
                    FROM graph_edge
                """)
            )
            edges = []
            for row in edge_result.fetchall():
                props = {}
                if row[3]:
                    props = row[3] if isinstance(row[3], dict) else json.loads(row[3])
                edges.append((row[0], row[1], row[2], props))

            return nodes, edges

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, str, str, Dict]]]:
        # Build WHERE clause from attribute filters
        where_parts = []
        params = {}
        for i, filter_dict in enumerate(attribute_filters):
            for attr, values in filter_dict.items():
                param = f"filt_{i}_{attr}"
                where_parts.append(f"n.{attr} = ANY(:{param})")
                params[param] = values

        where_clause = " AND ".join(where_parts)

        async with self._session() as session:
            # Fetch filtered nodes
            node_result = await session.execute(
                text(f"SELECT id, name, type, properties FROM graph_node n WHERE {where_clause}"),
                params,
            )
            nodes = []
            node_ids = []
            for row in node_result.fetchall():
                data = {"name": row[1], "type": row[2]}
                if row[3]:
                    data.update(row[3] if isinstance(row[3], dict) else json.loads(row[3]))
                nodes.append((row[0], data))
                node_ids.append(row[0])

            if not nodes:
                return [], []

            # Fetch edges between filtered nodes
            edge_result = await session.execute(
                text("""
                    SELECT source_id, target_id, relationship_name, properties
                    FROM graph_edge
                    WHERE source_id = ANY(:ids) AND target_id = ANY(:ids)
                """),
                {"ids": node_ids},
            )
            edges = []
            for row in edge_result.fetchall():
                props = {}
                if row[3]:
                    props = row[3] if isinstance(row[3], dict) else json.loads(row[3])
                edges.append((row[0], row[1], row[2], props))

            return nodes, edges

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str]
    ) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, str, str, dict]]]:
        label = node_type.__name__

        async with self._session() as session:
            # Find primary nodes
            primary_result = await session.execute(
                text("""
                    SELECT DISTINCT id FROM graph_node
                    WHERE type = :label AND name = ANY(:names)
                """),
                {"label": label, "names": node_name},
            )
            primary_ids = [row[0] for row in primary_result.fetchall()]
            if not primary_ids:
                return [], []

            # Find neighbors
            nbr_result = await session.execute(
                text("""
                    SELECT DISTINCT CASE
                        WHEN source_id = ANY(:ids) THEN target_id
                        ELSE source_id
                    END AS neighbor_id
                    FROM graph_edge
                    WHERE source_id = ANY(:ids) OR target_id = ANY(:ids)
                """),
                {"ids": primary_ids},
            )
            neighbor_ids = [row[0] for row in nbr_result.fetchall()]
            all_ids = list(set(primary_ids + neighbor_ids))

            # Fetch all relevant nodes
            node_result = await session.execute(
                text("SELECT id, name, type, properties FROM graph_node WHERE id = ANY(:ids)"),
                {"ids": all_ids},
            )
            nodes = []
            for row in node_result.fetchall():
                data = {"id": row[0], "name": row[1], "type": row[2]}
                if row[3]:
                    data.update(row[3] if isinstance(row[3], dict) else json.loads(row[3]))
                nodes.append((row[0], data))

            # Fetch edges between these nodes
            edge_result = await session.execute(
                text("""
                    SELECT source_id, target_id, relationship_name, properties
                    FROM graph_edge
                    WHERE source_id = ANY(:ids) AND target_id = ANY(:ids)
                """),
                {"ids": all_ids},
            )
            edges = []
            for row in edge_result.fetchall():
                props = {}
                if row[3]:
                    props = row[3] if isinstance(row[3], dict) else json.loads(row[3])
                edges.append((row[0], row[1], row[2], props))

            return nodes, edges

    # ------------------------------------------------------------------
    # GraphDBInterface: metrics
    # ------------------------------------------------------------------

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        async with self._session() as session:
            # Basic counts
            n_result = await session.execute(text("SELECT count(*) FROM graph_node"))
            num_nodes = n_result.scalar()
            e_result = await session.execute(text("SELECT count(*) FROM graph_edge"))
            num_edges = e_result.scalar()

            mean_degree = (2 * num_edges) / num_nodes if num_nodes else None
            edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

            # Connected components via recursive CTE
            comp_result = await session.execute(text("""
                WITH RECURSIVE component AS (
                    SELECT id AS node_id, id AS comp_root
                    FROM graph_node
                    UNION
                    SELECT
                        CASE WHEN e.source_id = c.node_id THEN e.target_id ELSE e.source_id END,
                        c.comp_root
                    FROM component c
                    JOIN graph_edge e ON e.source_id = c.node_id OR e.target_id = c.node_id
                ),
                node_comp AS (
                    SELECT node_id, MIN(comp_root) AS comp_id
                    FROM component
                    GROUP BY node_id
                )
                SELECT comp_id, count(*) AS sz
                FROM node_comp
                GROUP BY comp_id
                ORDER BY sz DESC
            """))
            comp_rows = comp_result.fetchall()
            num_components = len(comp_rows)
            component_sizes = [row[1] for row in comp_rows]

            metrics = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "mean_degree": mean_degree,
                "edge_density": edge_density,
                "num_connected_components": num_components,
                "sizes_of_connected_components": component_sizes,
            }

            if include_optional:
                # Self-loops
                sl_result = await session.execute(
                    text("SELECT count(*) FROM graph_edge WHERE source_id = target_id")
                )
                metrics["num_selfloops"] = sl_result.scalar()

                # Diameter and avg shortest path via BFS would be expensive;
                # return -1 as placeholder (same as Kuzu when not computed)
                metrics["diameter"] = -1
                metrics["avg_shortest_path_length"] = -1
                metrics["avg_clustering"] = -1
            else:
                metrics["num_selfloops"] = -1
                metrics["diameter"] = -1
                metrics["avg_shortest_path_length"] = -1
                metrics["avg_clustering"] = -1

            return metrics

    # ------------------------------------------------------------------
    # GraphDBInterface: deletion
    # ------------------------------------------------------------------

    async def delete_graph(self) -> None:
        await self._ensure_initialized()
        async with self._session() as session:
            await session.execute(text("TRUNCATE graph_edge, graph_node CASCADE"))
            await session.commit()

    # ------------------------------------------------------------------
    # GraphDBInterface: triplets
    # ------------------------------------------------------------------

    async def get_triplets_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        if offset < 0:
            raise ValueError(f"Offset must be non-negative, got {offset}")
        if limit < 0:
            raise ValueError(f"Limit must be non-negative, got {limit}")

        async with self._session() as session:
            result = await session.execute(
                text("""
                    SELECT
                        s.id, s.name, s.type, s.properties,
                        e.relationship_name, e.properties AS edge_props,
                        t.id, t.name, t.type, t.properties
                    FROM graph_edge e
                    JOIN graph_node s ON s.id = e.source_id
                    JOIN graph_node t ON t.id = e.target_id
                    ORDER BY e.source_id, e.target_id
                    OFFSET :off LIMIT :lim
                """),
                {"off": offset, "lim": limit},
            )

            triplets = []
            for row in result.fetchall():
                # Parse start node
                start_node = {"id": row[0], "name": row[1], "type": row[2]}
                if row[3]:
                    start_node.update(row[3] if isinstance(row[3], dict) else json.loads(row[3]))

                # Parse relationship
                rel = {"relationship_name": row[4]}
                if row[5]:
                    rel_props = row[5] if isinstance(row[5], dict) else json.loads(row[5])
                    rel.update(rel_props)

                # Parse end node
                end_node = {"id": row[6], "name": row[7], "type": row[8]}
                if row[9]:
                    end_node.update(row[9] if isinstance(row[9], dict) else json.loads(row[9]))

                triplets.append({
                    "start_node": start_node,
                    "relationship_properties": rel,
                    "end_node": end_node,
                })
            return triplets
