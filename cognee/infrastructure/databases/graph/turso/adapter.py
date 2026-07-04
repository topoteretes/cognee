"""Turso/libSQL graph adapter using two tables (graph_node, graph_edge) over SQLAlchemy + aiosqlite."""

import asyncio
import json
import time
from uuid import UUID
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Union, Optional, Tuple, Type

from sqlalchemy import text, insert
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.storage.utils import JSONEncoder

from .tables import _meta, _node_table, _edge_table

logger = get_logger()

_WRITE_CHUNK_SIZE = 500


def _in_params(prefix: str, values: List[str]) -> Tuple[str, Dict[str, str]]:
    """Build named params for SQLite IN clause (SQLite has no ANY(:list) support)."""
    params = {f"{prefix}_{i}": v for i, v in enumerate(values)}
    placeholders = ", ".join(f":{prefix}_{i}" for i in range(len(values)))
    return placeholders, params


class TursoAdapter(GraphDBInterface):
    """Graph-as-tables adapter backed by Turso/libSQL, accessed via SQLAlchemy async sessions."""

    _ALLOWED_FILTER_ATTRS = {"id", "name", "type"}

    def __init__(
        self,
        connection_string: str,
        remote_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        client_name: str = "cognee",
        sync_interval_seconds: float = 5.0,
    ) -> None:
        """Create engine and sessionmaker from a Turso connection string.

        If ``remote_url`` is set, the local file at ``connection_string`` is
        treated as an embedded replica kept in sync with the remote libSQL/Turso
        database via ``turso.aio.sync.connect()``.
        """
        # pyturso 0.6.1 omits has_stop on AsyncAdapt_turso_dbapi.
        # SQLAlchemy 2.0.51+ reads this flag in SQLiteDialect_aiosqlite.__init__
        # to decide whether force-close via stop() is available.
        # Setting False tells SQLAlchemy to skip force-close and use graceful
        # close only — correct since turso.aio has no stop() method.
        try:
            import turso.sqlalchemy.dialect as _turso_dialect

            if not hasattr(_turso_dialect.AsyncAdapt_turso_dbapi, "has_stop"):
                _turso_dialect.AsyncAdapt_turso_dbapi.has_stop = False
        except ImportError:
            pass

        self.db_uri = connection_string
        self.remote_url = remote_url or None
        self.auth_token = auth_token or None
        self.client_name = client_name
        self.sync_interval_seconds = sync_interval_seconds
        self._last_pull_monotonic: Optional[float] = None

        if self.remote_url:
            if "aioturso" not in connection_string:
                raise ValueError(
                    "Turso remote sync mode requires the aioturso driver "
                    "(sqlite+aioturso:///<local replica path>)."
                )

            def _async_creator_fn(database, **kw):
                import turso.aio.sync as turso_sync

                return turso_sync.connect(
                    database,
                    remote_url=self.remote_url,
                    auth_token=self.auth_token,
                    client_name=self.client_name,
                    bootstrap_if_empty=True,
                )

            self.engine = create_async_engine(
                self.db_uri,
                json_serializer=lambda obj: json.dumps(obj, cls=JSONEncoder),
                poolclass=StaticPool,
                connect_args={"async_creator_fn": _async_creator_fn},
            )
        else:
            self.engine = create_async_engine(
                self.db_uri,
                json_serializer=lambda obj: json.dumps(obj, cls=JSONEncoder),
            )
        self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)
        self._write_lock = asyncio.Lock()

    async def _get_sync_connection(self):
        """Reach the live turso.aio.sync ConnectionSync backing this engine."""
        async with self.engine.connect() as conn:
            raw = await conn.get_raw_connection()
            return raw.driver_connection

    async def _sync_pull(self) -> None:
        """Pull remote changes into the local replica."""
        if not self.remote_url:
            return
        conn = await self._get_sync_connection()
        await conn.pull()
        self._last_pull_monotonic = time.monotonic()

    async def _sync_push(self) -> None:
        """Push local writes to the remote database."""
        if not self.remote_url:
            return
        conn = await self._get_sync_connection()
        await conn.push()

    async def _maybe_pull(self) -> None:
        """Pull remote changes, throttled to at most once per sync_interval_seconds."""
        if not self.remote_url:
            return
        now = time.monotonic()
        if (
            self._last_pull_monotonic is not None
            and (now - self._last_pull_monotonic) < self.sync_interval_seconds
        ):
            return
        await self._sync_pull()

    async def close(self) -> None:
        """Dispose connection pool. Called by closing_lru_cache on eviction."""
        if self.remote_url:
            try:
                await self._sync_push()
            except Exception:
                logger.warning("Best-effort push before Turso adapter close failed", exc_info=True)
        await self.engine.dispose(close=True)

    async def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        if self.remote_url:
            await self._maybe_pull()
        async with self.engine.begin() as conn:
            # SQLite disables FK enforcement by default; cascade deletes need this
            await conn.execute(text("PRAGMA foreign_keys = ON"))
            await conn.run_sync(_meta.create_all, checkfirst=True)

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Any]:
        """Yield an async session from the underlying engine."""
        if self.remote_url:
            await self._maybe_pull()
        async with self.sessionmaker() as session:
            await session.execute(text("PRAGMA foreign_keys = ON"))
            yield session

    def _serialize_properties(self, props: Dict[str, Any]) -> str:
        """Serialize a dict to a JSON string, handling datetimes and UUIDs."""
        return json.dumps(props, cls=JSONEncoder)

    def _parse_node_row(self, row) -> Dict[str, Any]:
        """Convert a (id, name, type, properties) row to a merged dict."""
        data = {"id": row.id, "name": row.name, "type": row.type}
        if row.properties is not None:
            props = (
                row.properties if isinstance(row.properties, dict) else json.loads(row.properties)
            )
            data.update(props)
        return data

    async def query(self, query_str: str, params: Optional[dict] = None) -> List[Any]:
        """Not supported. Use typed adapter methods or a graph-native backend.

        Raises:
        -------
            NotImplementedError
        """
        raise NotImplementedError(
            "The Turso graph backend does not support raw Cypher queries. "
            "Use a graph-native backend (Neo4j, Ladybug) for raw query support, "
            "or use the typed adapter methods (add_nodes, get_neighbors, etc.)."
        )

    async def is_empty(self) -> bool:
        """Return True if the graph has no nodes."""
        await self.initialize()
        async with self._session() as session:
            result = await session.execute(text("SELECT EXISTS(SELECT 1 FROM graph_node LIMIT 1)"))
            return not result.scalar()

    async def add_node(
        self, node: Union[DataPoint, str], properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a single node. Delegates to add_nodes."""
        if isinstance(node, str):
            props = properties or {}
            props.setdefault("id", node)
            await self.add_nodes([(node, props)])
        else:
            await self.add_nodes([node])

    async def add_nodes(self, nodes: Union[List[Tuple[str, Dict]], List[DataPoint]]) -> None:
        """Add multiple nodes via batch upsert."""
        if not nodes:
            return

        now = datetime.now(timezone.utc)
        core_keys = {"id", "name", "type"}

        rows = []
        for node in nodes:
            if isinstance(node, tuple):
                props = {**(node[1] or {}), "id": node[0]}
            elif hasattr(node, "model_dump"):
                props = node.model_dump()
            else:
                props = vars(node)

            extra = {k: v for k, v in props.items() if k not in core_keys}
            rows.append(
                {
                    "id": str(props.get("id", "")),
                    "name": str(props.get("name", "")),
                    "type": str(props.get("type", "")),
                    "properties": self._serialize_properties(extra),
                    "created_at": now,
                    "updated_at": now,
                }
            )

        # Deduplicate by id (last wins)
        rows = list({r["id"]: r for r in rows}.values())

        async with self._write_lock:
            async with self._session() as session:
                for i in range(0, len(rows), _WRITE_CHUNK_SIZE):
                    chunk = rows[i : i + _WRITE_CHUNK_SIZE]
                    for row in chunk:
                        # prefix_with("OR REPLACE") → INSERT OR REPLACE INTO ...
                        stmt = insert(_node_table).prefix_with("OR REPLACE").values(row)
                        await session.execute(stmt)
                await session.commit()
            if self.remote_url:
                await self._sync_push()

    async def delete_node(self, node_id: str) -> None:
        """Delete a single node. Delegates to delete_nodes."""
        await self.delete_nodes([node_id])

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """Delete multiple nodes by ID. Cascade-deletes connected edges."""
        if not node_ids:
            return
        placeholders, params = _in_params("did", node_ids)
        async with self._write_lock:
            async with self._session() as session:
                await session.execute(
                    text(f"DELETE FROM graph_node WHERE id IN ({placeholders})"), params
                )
                await session.commit()
            if self.remote_url:
                await self._sync_push()

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single node by ID."""
        results = await self.get_nodes([node_id])
        return results[0] if results else None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple nodes by ID."""
        if not node_ids:
            return []
        placeholders, params = _in_params("gid", node_ids)
        async with self._session() as session:
            result = await session.execute(
                text(
                    f"SELECT id, name, type, properties FROM graph_node WHERE id IN ({placeholders})"
                ),
                params,
            )
            return [self._parse_node_row(row) for row in result.fetchall()]

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single edge. Delegates to add_edges."""
        await self.add_edges(
            [(str(source_id), str(target_id), relationship_name, properties or {})]
        )

    async def add_edges(
        self, edges: Union[List[Tuple[str, str, str, Optional[Dict[str, Any]]]], List]
    ) -> None:
        """Add multiple edges via batch upsert."""
        if not edges:
            return

        now = datetime.now(timezone.utc)

        rows = []
        for edge in edges:
            raw_props = edge[3] if len(edge) > 3 and edge[3] else {}
            rows.append(
                {
                    "source_id": str(edge[0]),
                    "target_id": str(edge[1]),
                    "relationship_name": edge[2],
                    "properties": self._serialize_properties(raw_props),
                    "created_at": now,
                    "updated_at": now,
                }
            )

        # Deduplicate by composite key (last wins)
        rows = list(
            {(r["source_id"], r["target_id"], r["relationship_name"]): r for r in rows}.values()
        )

        async with self._write_lock:
            async with self._session() as session:
                for i in range(0, len(rows), _WRITE_CHUNK_SIZE):
                    chunk = rows[i : i + _WRITE_CHUNK_SIZE]
                    for row in chunk:
                        stmt = insert(_edge_table).prefix_with("OR REPLACE").values(row)
                        await session.execute(stmt)
                await session.commit()
            if self.remote_url:
                await self._sync_push()

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """Check whether a single edge exists."""
        result = await self.has_edges([(str(source_id), str(target_id), relationship_name)])
        return len(result) > 0

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Return subset of input edge tuples that exist in the database."""
        if not edges:
            return []

        found: List[Tuple[str, str, str]] = []
        async with self._session() as session:
            for source_id, target_id, relationship_name in edges:
                result = await session.execute(
                    text("""
                        SELECT source_id, target_id, relationship_name
                        FROM graph_edge
                        WHERE source_id = :src
                          AND target_id = :tgt
                          AND relationship_name = :rel
                    """),
                    {"src": str(source_id), "tgt": str(target_id), "rel": relationship_name},
                )
                row = result.fetchone()
                if row:
                    found.append((row[0], row[1], row[2]))
        return found

    async def get_edges(self, node_id: str) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        """Retrieve all edges connected to a node as (source_dict, rel_name, target_dict)."""
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

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Retrieve all nodes directly connected to a given node."""
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
        """Retrieve all connections (source, edge, target) for a node."""
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

    async def get_graph_data(
        self,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Retrieve all nodes as (id, props) and edges as (src, tgt, rel, props)."""
        async with self._session() as session:
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

            edge_result = await session.execute(
                text("SELECT source_id, target_id, relationship_name, properties FROM graph_edge")
            )
            edges = []
            for row in edge_result.fetchall():
                props = {}
                if row[3]:
                    props = row[3] if isinstance(row[3], dict) else json.loads(row[3])
                edges.append((row[0], row[1], row[2], props))

            return nodes, edges

    async def get_id_filtered_graph_data(
        self, target_ids: List[str]
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Retrieve subgraph for edges touching target_ids, plus their endpoint nodes."""
        if not target_ids:
            return [], []
        ids = [str(i) for i in target_ids]
        placeholders, params = _in_params("tid", ids)

        async with self._session() as session:
            edge_result = await session.execute(
                text(f"""
                    SELECT source_id, target_id, relationship_name, properties
                    FROM graph_edge
                    WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                """),
                params,
            )
            edges = []
            endpoint_ids: set = set()
            for row in edge_result.fetchall():
                props = {}
                if row[3]:
                    props = row[3] if isinstance(row[3], dict) else json.loads(row[3])
                endpoint_ids.update((row[0], row[1]))
                edges.append((row[0], row[1], row[2], props))

            if not endpoint_ids:
                return [], []

            ep_ph, ep_params = _in_params("ep", list(endpoint_ids))
            node_result = await session.execute(
                text(f"SELECT id, name, type, properties FROM graph_node WHERE id IN ({ep_ph})"),
                ep_params,
            )
            nodes = []
            for row in node_result.fetchall():
                data = {"name": row[1], "type": row[2]}
                if row[3]:
                    data.update(row[3] if isinstance(row[3], dict) else json.loads(row[3]))
                nodes.append((row[0], data))

            return nodes, edges

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, str, str, Dict]]]:
        """Retrieve nodes matching attribute filters, plus edges between them."""
        if not attribute_filters:
            return await self.get_graph_data()

        where_parts = []
        params: Dict[str, Any] = {}
        for i, filter_dict in enumerate(attribute_filters):
            for attr, filter_values in filter_dict.items():
                if attr not in self._ALLOWED_FILTER_ATTRS:
                    raise ValueError(f"Invalid filter attribute: {attr!r}")
                ph, fp = _in_params(f"filt_{i}_{attr}", [str(v) for v in filter_values])
                where_parts.append(f"n.{attr} IN ({ph})")
                params.update(fp)

        if not where_parts:
            return await self.get_graph_data()

        where_clause = " AND ".join(where_parts)

        async with self._session() as session:
            node_result = await session.execute(
                text(f"""
                    SELECT id, name, type, properties
                    FROM graph_node n
                    WHERE {where_clause}
                """),
                params,
            )
            node_rows = node_result.fetchall()
            if not node_rows:
                return [], []

            node_ids = [row[0] for row in node_rows]
            nodes = []
            for row in node_rows:
                data = {"name": row[1], "type": row[2]}
                if row[3]:
                    data.update(row[3] if isinstance(row[3], dict) else json.loads(row[3]))
                nodes.append((row[0], data))

            fn_ph, fn_params = _in_params("fn", node_ids)
            edge_result = await session.execute(
                text(f"""
                    SELECT source_id, target_id, relationship_name, properties
                    FROM graph_edge
                    WHERE source_id IN ({fn_ph}) AND target_id IN ({fn_ph})
                """),
                fn_params,
            )
            edges = []
            for row in edge_result.fetchall():
                props = {}
                if row[3]:
                    props = row[3] if isinstance(row[3], dict) else json.loads(row[3])
                edges.append((row[0], row[1], row[2], props))

            return nodes, edges

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str], node_name_filter_operator: str = "OR"
    ) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, str, str, dict]]]:
        """Retrieve subgraph of matching nodes, their neighbors, and interconnecting edges."""
        label = node_type.__name__

        name_ph, name_params = _in_params("nm", node_name)
        name_params["label"] = label

        if node_name_filter_operator == "OR":
            neighbor_cte = """
                    neighbor_ids AS (
                        SELECT DISTINCT CASE
                            WHEN e.source_id IN (SELECT id FROM primary_nodes)
                            THEN e.target_id ELSE e.source_id
                        END AS id
                        FROM graph_edge e
                        WHERE e.source_id IN (SELECT id FROM primary_nodes)
                           OR e.target_id IN (SELECT id FROM primary_nodes)
                    )"""
        else:
            neighbor_cte = """
                    neighbor_ids AS (
                        SELECT nbr_id AS id FROM (
                            SELECT CASE
                                WHEN e.source_id IN (SELECT id FROM primary_nodes)
                                THEN e.target_id ELSE e.source_id
                            END AS nbr_id,
                            CASE
                                WHEN e.source_id IN (SELECT id FROM primary_nodes)
                                THEN e.source_id ELSE e.target_id
                            END AS primary_id
                            FROM graph_edge e
                            WHERE e.source_id IN (SELECT id FROM primary_nodes)
                               OR e.target_id IN (SELECT id FROM primary_nodes)
                        ) sub
                        GROUP BY nbr_id
                        HAVING COUNT(DISTINCT primary_id) = :primary_count
                    )"""

        query_str = f"""
                    WITH primary_nodes AS (
                        SELECT DISTINCT id
                        FROM graph_node
                        WHERE type = :label AND name IN ({name_ph})
                    ),
                    {neighbor_cte},
                    all_ids AS (
                        SELECT id FROM primary_nodes
                        UNION
                        SELECT id FROM neighbor_ids
                    )
                    SELECT 'node' AS kind,
                           n.id, n.name, n.type, n.properties,
                           NULL AS source_id, NULL AS target_id,
                           NULL AS relationship_name, NULL AS edge_props
                    FROM graph_node n
                    WHERE n.id IN (SELECT id FROM all_ids)
                    UNION ALL
                    SELECT 'edge', NULL, NULL, NULL, NULL,
                           e.source_id, e.target_id,
                           e.relationship_name, e.properties
                    FROM graph_edge e
                    WHERE e.source_id IN (SELECT id FROM all_ids)
                      AND e.target_id IN (SELECT id FROM all_ids)
                """

        if node_name_filter_operator != "OR":
            name_params["primary_count"] = len(node_name)

        async with self._session() as session:
            result = await session.execute(text(query_str), name_params)

            nodes = []
            edges = []
            for row in result.fetchall():
                if row[0] == "node":
                    data = {"name": row[2], "type": row[3]}
                    if row[4]:
                        data.update(row[4] if isinstance(row[4], dict) else json.loads(row[4]))
                    nodes.append((row[1], data))
                else:
                    props = {}
                    if row[8]:
                        props = row[8] if isinstance(row[8], dict) else json.loads(row[8])
                    edges.append((row[5], row[6], row[7], props))

            return nodes, edges

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        """Compute graph metrics matching the PostgresAdapter output schema."""
        async with self._session() as session:
            n_result = await session.execute(text("SELECT count(*) FROM graph_node"))
            num_nodes = n_result.scalar()
            e_result = await session.execute(text("SELECT count(*) FROM graph_edge"))
            num_edges = e_result.scalar()

            mean_degree = (2 * num_edges) / num_nodes if num_nodes else None
            edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

            # SQLite supports recursive CTEs for connected components
            comp_result = await session.execute(
                text("""
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
            """)
            )
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
                sl_result = await session.execute(
                    text("SELECT count(*) FROM graph_edge WHERE source_id = target_id")
                )
                metrics["num_selfloops"] = sl_result.scalar()
                metrics["diameter"] = -1
                metrics["avg_shortest_path_length"] = -1
                metrics["avg_clustering"] = -1
            else:
                metrics["num_selfloops"] = -1
                metrics["diameter"] = -1
                metrics["avg_shortest_path_length"] = -1
                metrics["avg_clustering"] = -1

            return metrics

    async def get_neighborhood(
        self,
        node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Get the k-hop neighborhood subgraph around seed nodes."""
        if not node_ids:
            return [], []

        edge_filter = ""
        et_params: Dict[str, Any] = {}
        if edge_types:
            et_ph, et_params = _in_params("et", edge_types)
            edge_filter = f"AND e.relationship_name IN ({et_ph})"

        # SQLite has no unnest(); build seed rows via UNION ALL with named params
        seed_selects = " UNION ALL ".join(
            [f"SELECT :seed_{i} AS id, 0 AS hops" for i in range(len(node_ids))]
        )
        seed_params = {f"seed_{i}": nid for i, nid in enumerate(node_ids)}

        query_str = f"""
            WITH RECURSIVE neighborhood(id, hops) AS (
                {seed_selects}
              UNION
                SELECT CASE WHEN e.source_id = n.id THEN e.target_id
                            ELSE e.source_id END,
                       n.hops + 1
                FROM neighborhood n
                JOIN graph_edge e ON (e.source_id = n.id OR e.target_id = n.id)
                    {edge_filter}
                WHERE n.hops < :depth
            ),
            ids AS (SELECT DISTINCT id FROM neighborhood)

            SELECT 'node' AS kind,
                   gn.id, gn.name, gn.type, gn.properties,
                   NULL AS source_id, NULL AS target_id,
                   NULL AS relationship_name, NULL AS edge_properties
            FROM graph_node gn
            JOIN ids ON gn.id = ids.id

            UNION ALL

            SELECT 'edge' AS kind,
                   NULL, NULL, NULL, NULL,
                   ge.source_id, ge.target_id,
                   ge.relationship_name, ge.properties
            FROM graph_edge ge
            WHERE ge.source_id IN (SELECT id FROM ids)
              AND ge.target_id IN (SELECT id FROM ids)
        """

        params: Dict[str, Any] = {"depth": depth, **seed_params, **et_params}

        async with self._session() as session:
            result = await session.execute(text(query_str), params)

            nodes = []
            edges = []
            for row in result.fetchall():
                if row.kind == "node":
                    data = self._parse_node_row(row)
                    data.pop("id", None)
                    nodes.append((row.id, data))
                else:
                    props = {}
                    if row.edge_properties is not None:
                        props = (
                            row.edge_properties
                            if isinstance(row.edge_properties, dict)
                            else json.loads(row.edge_properties)
                        )
                    edges.append((row.source_id, row.target_id, row.relationship_name, props))

            return nodes, edges

    async def delete_graph(self) -> None:
        """Delete all nodes and edges from the graph."""
        await self.initialize()
        async with self._write_lock:
            async with self._session() as session:
                await session.execute(text("DELETE FROM graph_edge"))
                await session.execute(text("DELETE FROM graph_node"))
                await session.commit()
            if self.remote_url:
                await self._sync_push()

    async def get_triplets_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        """Retrieve a batch of (source, relationship, target) triplets."""
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
                    ORDER BY e.source_id, e.target_id, e.relationship_name
                    LIMIT :lim OFFSET :off
                """),
                {"off": offset, "lim": limit},
            )

            triplets = []
            for row in result.fetchall():
                start_node = {"id": row[0], "name": row[1], "type": row[2]}
                if row[3]:
                    start_node.update(row[3] if isinstance(row[3], dict) else json.loads(row[3]))

                rel = {"relationship_name": row[4]}
                if row[5]:
                    rel_props = row[5] if isinstance(row[5], dict) else json.loads(row[5])
                    rel.update(rel_props)

                end_node = {"id": row[6], "name": row[7], "type": row[8]}
                if row[9]:
                    end_node.update(row[9] if isinstance(row[9], dict) else json.loads(row[9]))

                triplets.append(
                    {
                        "start_node": start_node,
                        "relationship_properties": rel,
                        "end_node": end_node,
                    }
                )
            return triplets
