"""Turso/libSQL graph adapter using two tables (graph_node, graph_edge) over SQLAlchemy + aiosqlite."""

import asyncio
import json
from uuid import UUID
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Union, Optional, Tuple, Type

from sqlalchemy import text, event
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.storage.utils import JSONEncoder

from .tables import _meta, _node_table, _edge_table

logger = get_logger()

_WRITE_CHUNK_SIZE = 500


def _in_params(prefix: str, values: List[str]) -> Tuple[str, Dict[str, str]]:
    """Build named params for a SQLite IN clause of small, bounded lists
    (edge types, node names, filter values). SQLite has no ANY(:list) support.
    """
    params = {f"{prefix}_{i}": v for i, v in enumerate(values)}
    placeholders = ", ".join(f":{prefix}_{i}" for i in range(len(values)))
    return placeholders, params


def _id_subquery(prefix: str, ids: List[str]) -> Tuple[str, Dict[str, str]]:
    """Build a ``(SELECT value FROM json_each(:prefix))`` subquery and a single
    JSON-array param for an id list.

    One bound parameter regardless of list size — mirrors Postgres's
    ``= ANY(:ids)`` and stays under SQLite's per-statement variable cap, so bulk
    deletes/reads of large id sets do not raise "too many SQL variables".
    """
    return (
        f"(SELECT value FROM json_each(:{prefix}))",
        {prefix: json.dumps([str(i) for i in ids])},
    )


class TursoAdapter(GraphDBInterface):
    """Graph-as-tables adapter backed by Turso/libSQL, accessed via SQLAlchemy async sessions."""

    _ALLOWED_FILTER_ATTRS = {"id", "name", "type"}

    def __init__(self, connection_string: str) -> None:
        """Create engine and sessionmaker for a local libSQL file.

        A libSQL file is a SQLite file, so cognee talks to it through the same
        aiosqlite driver it already uses for SQLite (``sqlite+aiosqlite:///``).
        """
        self.db_uri = connection_string
        # Properties are serialized to TEXT columns by _serialize_properties, so
        # there is no JSON-typed column for SQLAlchemy to encode — hence no
        # json_serializer, unlike the Postgres adapter with its JSONB columns.
        self.engine = create_async_engine(self.db_uri)

        # These PRAGMAs are connection-scoped and a no-op inside a transaction, so
        # apply them on every new connection via the connect event, mirroring the
        # relational SqlAlchemyAdapter (issue #2717):
        #   - foreign_keys: SQLite disables FK enforcement by default; graph_edge's
        #     ON DELETE CASCADE relies on it.
        #   - journal_mode=WAL + busy_timeout: concurrent writers wait instead of
        #     failing with "database is locked".
        @event.listens_for(self.engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=120000")
            finally:
                cursor.close()

        self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)
        self._write_lock = asyncio.Lock()

    async def close(self) -> None:
        """Dispose connection pool. Called by closing_lru_cache on eviction."""
        await self.engine.dispose(close=True)

    async def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(_meta.create_all, checkfirst=True)

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Any]:
        """Yield an async session from the underlying engine."""
        async with self.sessionmaker() as session:
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

    async def add_nodes(
        self,
        nodes: Union[List[Tuple[str, Dict]], List[DataPoint]],
        source_ref_key: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> None:
        """Add multiple nodes via batch upsert.

        ``source_ref_key`` / ``pipeline_run_id`` are the graph-provenance stamp the
        storage path always passes; Turso does not fold provenance into the graph
        (it uses the relational-ledger delete path), so they are accepted and
        ignored — the same contract as any non-graph-provenance backend.
        """
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

        # Upsert with ON CONFLICT DO UPDATE (not INSERT OR REPLACE). REPLACE deletes
        # the conflicting row first, which fires graph_edge's ON DELETE CASCADE and
        # would wipe a node's edges every time it is re-added; DO UPDATE edits in
        # place, preserving edges and created_at. Mirrors the Postgres adapter.
        async with self._write_lock:
            async with self._session() as session:
                for i in range(0, len(rows), _WRITE_CHUNK_SIZE):
                    chunk = rows[i : i + _WRITE_CHUNK_SIZE]
                    stmt = sqlite_insert(_node_table).values(chunk)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["id"],
                        set_={
                            "name": stmt.excluded.name,
                            "type": stmt.excluded.type,
                            "properties": stmt.excluded.properties,
                            "updated_at": stmt.excluded.updated_at,
                        },
                    )
                    await session.execute(stmt)
                await session.commit()

    async def delete_node(self, node_id: str) -> None:
        """Delete a single node. Delegates to delete_nodes."""
        await self.delete_nodes([node_id])

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """Delete multiple nodes by ID. Cascade-deletes connected edges."""
        if not node_ids:
            return
        subquery, params = _id_subquery("did", node_ids)
        async with self._write_lock:
            async with self._session() as session:
                await session.execute(
                    text(f"DELETE FROM graph_node WHERE id IN {subquery}"), params
                )
                await session.commit()

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single node by ID."""
        results = await self.get_nodes([node_id])
        return results[0] if results else None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple nodes by ID."""
        if not node_ids:
            return []
        subquery, params = _id_subquery("gid", node_ids)
        async with self._session() as session:
            result = await session.execute(
                text(f"SELECT id, name, type, properties FROM graph_node WHERE id IN {subquery}"),
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
        self,
        edges: Union[List[Tuple[str, str, str, Optional[Dict[str, Any]]]], List],
        source_ref_key: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> None:
        """Add multiple edges via batch upsert.

        ``source_ref_key`` / ``pipeline_run_id`` are accepted and ignored (see
        ``add_nodes``).
        """
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

        # ON CONFLICT DO UPDATE, not INSERT OR REPLACE (see add_nodes for why).
        async with self._write_lock:
            async with self._session() as session:
                for i in range(0, len(rows), _WRITE_CHUNK_SIZE):
                    chunk = rows[i : i + _WRITE_CHUNK_SIZE]
                    stmt = sqlite_insert(_edge_table).values(chunk)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["source_id", "target_id", "relationship_name"],
                        set_={
                            "properties": stmt.excluded.properties,
                            "updated_at": stmt.excluded.updated_at,
                        },
                    )
                    await session.execute(stmt)
                await session.commit()

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """Check whether a single edge exists."""
        result = await self.has_edges([(str(source_id), str(target_id), relationship_name)])
        return len(result) > 0

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Return subset of input edge tuples that exist in the database.

        Resolved with a single set-based query (candidates joined against
        graph_edge via json_each) rather than one SELECT per edge — this runs on
        the cognify dedup hot path with thousands of candidate edges per batch.
        """
        if not edges:
            return []

        candidates = json.dumps([[str(s), str(t), str(r)] for s, t, r in edges])
        async with self._session() as session:
            result = await session.execute(
                text("""
                    SELECT j.value ->> 0, j.value ->> 1, j.value ->> 2
                    FROM json_each(:candidates) j
                    WHERE EXISTS (
                        SELECT 1 FROM graph_edge e
                        WHERE e.source_id = j.value ->> 0
                          AND e.target_id = j.value ->> 1
                          AND e.relationship_name = j.value ->> 2
                    )
                """),
                {"candidates": candidates},
            )
            return [(row[0], row[1], row[2]) for row in result.fetchall()]

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
        subquery, params = _id_subquery("tid", target_ids)

        async with self._session() as session:
            edge_result = await session.execute(
                text(f"""
                    SELECT source_id, target_id, relationship_name, properties
                    FROM graph_edge
                    WHERE source_id IN {subquery} OR target_id IN {subquery}
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

            ep_subquery, ep_params = _id_subquery("ep", list(endpoint_ids))
            node_result = await session.execute(
                text(
                    f"SELECT id, name, type, properties FROM graph_node WHERE id IN {ep_subquery}"
                ),
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
                if not filter_values:
                    # Empty value list matches nothing (SQLite has no "IN ()");
                    # matches Postgres's "= ANY('{}')" semantics.
                    where_parts.append("0 = 1")
                    continue
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

            fn_subquery, fn_params = _id_subquery("fn", node_ids)
            edge_result = await session.execute(
                text(f"""
                    SELECT source_id, target_id, relationship_name, properties
                    FROM graph_edge
                    WHERE source_id IN {fn_subquery} AND target_id IN {fn_subquery}
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
        if not node_name:
            return [], []
        label = node_type.__name__

        name_ph, name_params = _in_params("nm", node_name)
        params: Dict[str, Any] = {**name_params, "label": label}

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
            params["primary_count"] = len(node_name)

        async with self._session() as session:
            result = await session.execute(text(query_str), params)

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

        # SQLite has no unnest(); seed the recursion from a JSON array (one bound
        # param, no per-seed variable cap).
        seed_subquery, seed_params = _id_subquery("seeds", node_ids)

        query_str = f"""
            WITH RECURSIVE neighborhood(id, hops) AS (
                SELECT value AS id, 0 AS hops FROM json_each(:seeds)
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

    async def remove_belongs_to_set_tags(
        self,
        tags: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> None:
        """Strip ``tags`` from each node's ``belongs_to_set`` property array.

        Keeps the denormalized membership list consistent with the additive
        belongs_to_set edges after a NodeSet (or its dataset) is deleted.
        ``belongs_to_set`` lives inside the JSON ``properties`` TEXT blob, so this
        is a read-filter-write over that array. Mirrors the Postgres adapter.
        """
        if not tags:
            return None
        if node_ids is not None and not node_ids:
            return None

        tag_set = set(tags)
        async with self._session() as session:
            if node_ids is not None:
                subquery, params = _id_subquery("bts", node_ids)
                result = await session.execute(
                    text(f"SELECT id, properties FROM graph_node WHERE id IN {subquery}"), params
                )
            else:
                result = await session.execute(text("SELECT id, properties FROM graph_node"))
            rows = result.fetchall()

        updates = []
        for row in rows:
            properties = json.loads(row[1]) if row[1] else {}
            current = properties.get("belongs_to_set")
            if not isinstance(current, list) or not any(tag in tag_set for tag in current):
                continue
            properties["belongs_to_set"] = [tag for tag in current if tag not in tag_set]
            updates.append({"id": row[0], "properties": self._serialize_properties(properties)})

        if updates:
            now = datetime.now(timezone.utc)
            async with self._write_lock:
                async with self._session() as session:
                    for update in updates:
                        await session.execute(
                            text(
                                "UPDATE graph_node SET properties = :p, updated_at = :now "
                                "WHERE id = :id"
                            ),
                            {"id": update["id"], "p": update["properties"], "now": now},
                        )
                    await session.commit()
        return None
