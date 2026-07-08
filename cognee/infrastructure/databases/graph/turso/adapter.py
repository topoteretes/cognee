"""Turso graph adapter using two tables (graph_node, graph_edge) over libsql API."""

import asyncio
import json
from uuid import UUID
from datetime import datetime, timezone
from typing import Dict, Any, List, Union, Optional, Tuple, Type
import logging

import libsql
from sqlalchemy import text, values, select, exists, func, String
from sqlalchemy import column as sa_column
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.storage.utils import JSONEncoder

from .tables import _meta, _node_table, _edge_table

logger = get_logger()

_WRITE_CHUNK_SIZE = 1000

class TursoAdapter(GraphDBInterface):
    """Graph-as-tables adapter backed by Turso (libSQL)."""

    _ALLOWED_FILTER_ATTRS = {"id", "name", "type"}

    def __init__(self, connection_string: str) -> None:
        self.db_uri = connection_string
        # If the user passes a file:// URL or just a path, libsql requires valid format.
        # Ensure it works correctly with libsql.
        self._write_lock = asyncio.Lock()

    async def _execute_sync(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    def _connect(self):
        # By default libsql.connect creates a new connection each time.
        # In a real app we might want to pool this, but for now we mirror typical sqlite usage
        # or rely on factory caching.
        return libsql.connect(self.db_uri)

    async def close(self) -> None:
        """Close connection (if pooled). Factory handles lifecycle."""
        pass

    async def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        def _init():
            conn = self._connect()
            try:
                # Use sqlite dialect to generate create table statements
                from sqlalchemy.schema import CreateTable, CreateIndex
                from sqlalchemy.dialects import sqlite
                dialect = sqlite.dialect()
                
                for table in _meta.sorted_tables:
                    stmt = str(CreateTable(table).compile(dialect=dialect)).strip()
                    # SQLite CreateTable does not include IF NOT EXISTS by default when compiled this way
                    stmt = stmt.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
                    conn.execute(stmt)
                    
                    for index in table.indexes:
                        idx_stmt = str(CreateIndex(index).compile(dialect=dialect)).strip()
                        idx_stmt = idx_stmt.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS")
                        conn.execute(idx_stmt)
                conn.commit()
            finally:
                conn.close()

        await self._execute_sync(_init)

    def _serialize_properties(self, props: Dict[str, Any]) -> str:
        return json.dumps(props, cls=JSONEncoder)

    def _parse_node_row(self, row) -> Dict[str, Any]:
        """Convert a (id, name, type, properties) row to a merged dict."""
        data = {"id": row[0], "name": row[1], "type": row[2]}
        props = row[3]
        if props:
            data.update(json.loads(props))
        return data

    async def query(self, query_str: str, params: Optional[dict] = None) -> List[Any]:
        raise NotImplementedError(
            "The Turso graph backend does not support raw Cypher queries."
        )

    async def is_empty(self) -> bool:
        await self.initialize()
        
        def _check():
            conn = self._connect()
            try:
                cursor = conn.execute("SELECT 1 FROM graph_node LIMIT 1")
                return cursor.fetchone() is None
            finally:
                conn.close()
                
        return await self._execute_sync(_check)

    async def add_node(
        self, node: Union[DataPoint, str], properties: Optional[Dict[str, Any]] = None
    ) -> None:
        if isinstance(node, str):
            props = properties or {}
            props.setdefault("id", node)
            await self.add_nodes([(node, props)])
        else:
            await self.add_nodes([node])

    async def add_nodes(self, nodes: Union[List[Tuple[str, Dict]], List[DataPoint]]) -> None:
        if not nodes:
            return

        now = datetime.now(timezone.utc).isoformat()
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
            rows.append({
                "id": str(props.get("id", "")),
                "name": str(props.get("name", "")),
                "type": str(props.get("type", "")),
                "properties": self._serialize_properties(extra),
                "created_at": now,
                "updated_at": now,
            })

        # Deduplicate
        rows = list({r["id"]: r for r in rows}.values())
        
        sql = """
            INSERT INTO graph_node (id, name, type, properties, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                properties = excluded.properties,
                updated_at = excluded.updated_at
        """

        def _insert():
            conn = self._connect()
            try:
                for i in range(0, len(rows), _WRITE_CHUNK_SIZE):
                    chunk = rows[i:i+_WRITE_CHUNK_SIZE]
                    params = [(r["id"], r["name"], r["type"], r["properties"], r["created_at"], r["updated_at"]) for r in chunk]
                    conn.executemany(sql, params)
                conn.commit()
            finally:
                conn.close()

        async with self._write_lock:
            await self._execute_sync(_insert)

    async def delete_graph(self) -> None:
        def _delete_graph():
            conn = self._connect()
            try:
                conn.execute("DELETE FROM graph_edge")
                conn.execute("DELETE FROM graph_node")
                conn.commit()
            finally:
                conn.close()
                
        async with self._write_lock:
            await self._execute_sync(_delete_graph)

    async def delete_node(self, node_id: str) -> None:
        await self.delete_nodes([node_id])

    async def delete_nodes(self, node_ids: List[str]) -> None:
        if not node_ids:
            return
            
        def _delete():
            conn = self._connect()
            try:
                placeholders = ",".join("?" for _ in node_ids)
                conn.execute(f"DELETE FROM graph_node WHERE id IN ({placeholders})", tuple(node_ids))
                conn.commit()
            finally:
                conn.close()

        async with self._write_lock:
            await self._execute_sync(_delete)

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        results = await self.get_nodes([node_id])
        return results[0] if results else None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        if not node_ids:
            return []
            
        def _get():
            conn = self._connect()
            try:
                placeholders = ",".join("?" for _ in node_ids)
                cursor = conn.execute(
                    f"SELECT id, name, type, properties FROM graph_node WHERE id IN ({placeholders})", 
                    tuple(node_ids)
                )
                return [self._parse_node_row(row) for row in cursor.fetchall()]
            finally:
                conn.close()
                
        return await self._execute_sync(_get)

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self.add_edges(
            [(str(source_id), str(target_id), relationship_name, properties or {})]
        )

    async def add_edges(
        self, edges: Union[List[Tuple[str, str, str, Optional[Dict[str, Any]]]], List]
    ) -> None:
        if not edges:
            return

        now = datetime.now(timezone.utc).isoformat()

        rows = []
        for edge in edges:
            raw_props = edge[3] if len(edge) > 3 and edge[3] else {}
            rows.append({
                "source_id": str(edge[0]),
                "target_id": str(edge[1]),
                "relationship_name": edge[2],
                "properties": self._serialize_properties(raw_props),
                "created_at": now,
                "updated_at": now,
            })

        # Deduplicate
        rows = list({(r["source_id"], r["target_id"], r["relationship_name"]): r for r in rows}.values())

        sql = """
            INSERT INTO graph_edge (source_id, target_id, relationship_name, properties, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relationship_name) DO UPDATE SET
                properties = excluded.properties,
                updated_at = excluded.updated_at
        """

        def _insert():
            conn = self._connect()
            try:
                for i in range(0, len(rows), _WRITE_CHUNK_SIZE):
                    chunk = rows[i:i+_WRITE_CHUNK_SIZE]
                    params = [(r["source_id"], r["target_id"], r["relationship_name"], r["properties"], r["created_at"], r["updated_at"]) for r in chunk]
                    conn.executemany(sql, params)
                conn.commit()
            finally:
                conn.close()

        async with self._write_lock:
            await self._execute_sync(_insert)

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        result = await self.has_edges([(str(source_id), str(target_id), relationship_name)])
        return len(result) > 0

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        if not edges:
            return []

        CHUNK_SIZE = 10_000
        found: List[Tuple[str, str, str]] = []

        def _check():
            conn = self._connect()
            try:
                for i in range(0, len(edges), CHUNK_SIZE):
                    chunk = edges[i : i + CHUNK_SIZE]
                    # SQLite supports IN with tuples: WHERE (source, target, rel) IN ((?, ?, ?), ...)
                    # Build dynamic query
                    placeholders = ",".join("(?, ?, ?)" for _ in chunk)
                    flat_params = []
                    for s, t, r in chunk:
                        flat_params.extend([str(s), str(t), str(r)])
                        
                    cursor = conn.execute(
                        f"SELECT source_id, target_id, relationship_name FROM graph_edge WHERE (source_id, target_id, relationship_name) IN ({placeholders})",
                        tuple(flat_params)
                    )
                    found.extend((row[0], row[1], row[2]) for row in cursor.fetchall())
            finally:
                conn.close()

        await self._execute_sync(_check)
        return found

    async def get_edges(self, node_id: str) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        def _get():
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    SELECT
                        n.id, n.name, n.type, n.properties,
                        e.relationship_name,
                        m.id, m.name, m.type, m.properties
                    FROM graph_edge e
                    JOIN graph_node n ON n.id = e.source_id
                    JOIN graph_node m ON m.id = e.target_id
                    WHERE e.source_id = ? OR e.target_id = ?
                    """,
                    (node_id, node_id)
                )
                edges = []
                for row in cursor.fetchall():
                    src = {"id": row[0], "name": row[1], "type": row[2]}
                    if row[3]:
                        src.update(json.loads(row[3]))
                    tgt = {"id": row[5], "name": row[6], "type": row[7]}
                    if row[8]:
                        tgt.update(json.loads(row[8]))
                    edges.append((src, row[4], tgt))
                return edges
            finally:
                conn.close()
                
        return await self._execute_sync(_get)

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        def _get():
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT m.id, m.name, m.type, m.properties
                    FROM graph_edge e
                    JOIN graph_node m ON m.id = CASE
                        WHEN e.source_id = ? THEN e.target_id
                        ELSE e.source_id
                    END
                    WHERE e.source_id = ? OR e.target_id = ?
                    """,
                    (node_id, node_id, node_id)
                )
                return [self._parse_node_row(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        return await self._execute_sync(_get)

    async def get_connections(
        self, node_id: Union[str, UUID]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        nid = str(node_id)
        
        def _get():
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    SELECT
                        n.id, n.name, n.type, n.properties,
                        e.relationship_name, e.properties AS edge_props,
                        m.id, m.name, m.type, m.properties
                    FROM graph_edge e
                    JOIN graph_node n ON n.id = e.source_id
                    JOIN graph_node m ON m.id = e.target_id
                    WHERE e.source_id = ? OR e.target_id = ?
                    """,
                    (nid, nid)
                )
                connections = []
                for row in cursor.fetchall():
                    src = {"id": row[0], "name": row[1], "type": row[2]}
                    if row[3]:
                        src.update(json.loads(row[3]))

                    edge = {"relationship_name": row[4]}
                    if row[5]:
                        edge.update(json.loads(row[5]))

                    tgt = {"id": row[6], "name": row[7], "type": row[8]}
                    if row[9]:
                        tgt.update(json.loads(row[9]))

                    connections.append((src, edge, tgt))
                return connections
            finally:
                conn.close()
                
        return await self._execute_sync(_get)

    async def get_graph_data(
        self,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        def _get():
            conn = self._connect()
            try:
                n_cursor = conn.execute("SELECT id, name, type, properties FROM graph_node")
                nodes = []
                for row in n_cursor.fetchall():
                    data = {"name": row[1], "type": row[2]}
                    if row[3]:
                        data.update(json.loads(row[3]))
                    nodes.append((row[0], data))

                if not nodes:
                    return [], []

                e_cursor = conn.execute("SELECT source_id, target_id, relationship_name, properties FROM graph_edge")
                edges = []
                for row in e_cursor.fetchall():
                    props = json.loads(row[3]) if row[3] else {}
                    edges.append((row[0], row[1], row[2], props))

                return nodes, edges
            finally:
                conn.close()
                
        return await self._execute_sync(_get)

    async def get_id_filtered_graph_data(
        self, target_ids: List[str]
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        if not target_ids:
            return [], []
        ids = [str(i) for i in target_ids]

        def _get():
            conn = self._connect()
            try:
                placeholders = ",".join("?" for _ in ids)
                e_cursor = conn.execute(
                    f"""
                    SELECT source_id, target_id, relationship_name, properties
                    FROM graph_edge
                    WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
                    """,
                    tuple(ids + ids)
                )
                edges = []
                endpoint_ids = set()
                for row in e_cursor.fetchall():
                    props = json.loads(row[3]) if row[3] else {}
                    endpoint_ids.update((row[0], row[1]))
                    edges.append((row[0], row[1], row[2], props))

                if not endpoint_ids:
                    return [], []

                endpoint_ids_list = list(endpoint_ids)
                n_placeholders = ",".join("?" for _ in endpoint_ids_list)
                n_cursor = conn.execute(
                    f"SELECT id, name, type, properties FROM graph_node WHERE id IN ({n_placeholders})",
                    tuple(endpoint_ids_list)
                )
                nodes = []
                for row in n_cursor.fetchall():
                    data = {"name": row[1], "type": row[2]}
                    if row[3]:
                        data.update(json.loads(row[3]))
                    nodes.append((row[0], data))

                return nodes, edges
            finally:
                conn.close()
                
        return await self._execute_sync(_get)

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, str, str, Dict]]]:
        if not attribute_filters:
            return await self.get_graph_data()

        where_parts = []
        params = []
        for filter_dict in attribute_filters:
            for attr, filter_values in filter_dict.items():
                if attr not in self._ALLOWED_FILTER_ATTRS:
                    raise ValueError(f"Invalid filter attribute: {attr!r}")
                placeholders = ",".join("?" for _ in filter_values)
                where_parts.append(f"n.{attr} IN ({placeholders})")
                params.extend(filter_values)

        if not where_parts:
            return await self.get_graph_data()

        where_clause = " AND ".join(where_parts)

        def _get():
            conn = self._connect()
            try:
                query = f"""
                    WITH filtered_nodes AS (
                        SELECT id, name, type, properties
                        FROM graph_node n
                        WHERE {where_clause}
                    )
                    SELECT 'node' AS kind, fn.id, fn.name, fn.type, fn.properties,
                           NULL AS source_id, NULL AS target_id,
                           NULL AS relationship_name, NULL AS edge_props
                    FROM filtered_nodes fn
                    UNION ALL
                    SELECT 'edge', NULL, NULL, NULL, NULL,
                           e.source_id, e.target_id,
                           e.relationship_name, e.properties
                    FROM graph_edge e
                    WHERE e.source_id IN (SELECT id FROM filtered_nodes)
                      AND e.target_id IN (SELECT id FROM filtered_nodes)
                """
                cursor = conn.execute(query, tuple(params))
                
                nodes = []
                edges = []
                for row in cursor.fetchall():
                    if row[0] == "node":
                        data = {"name": row[2], "type": row[3]}
                        if row[4]:
                            data.update(json.loads(row[4]))
                        nodes.append((row[1], data))
                    else:
                        props = json.loads(row[8]) if row[8] else {}
                        edges.append((row[5], row[6], row[7], props))
                return nodes, edges
            finally:
                conn.close()

        return await self._execute_sync(_get)

    async def get_neighborhood(
        self,
        node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Get the k-hop neighborhood subgraph around seed nodes."""
        if not node_ids:
            return [], []

        def _get():
            conn = self._connect()
            try:
                edge_filter = ""
                query_params = [json.dumps(node_ids)]
                
                if edge_types:
                    placeholders = ", ".join("?" for _ in range(len(edge_types)))
                    edge_filter = f"AND e.relationship_name IN ({placeholders})"
                    query_params.extend(edge_types)
                    
                query_params.append(depth)

                # Turso/SQLite doesn't directly support unnesting text arrays.
                # However, we can use the json_each table-valued function on a JSON array of seeds.
                query_str = f"""
                    WITH RECURSIVE neighborhood(id, hops) AS (
                        SELECT value, 0 FROM json_each(?)
                      UNION
                        SELECT CASE WHEN e.source_id = n.id THEN e.target_id
                                    ELSE e.source_id END,
                               n.hops + 1
                        FROM neighborhood n
                        JOIN graph_edge e ON (e.source_id = n.id OR e.target_id = n.id)
                            {edge_filter}
                        WHERE n.hops < ?
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
                
                cursor = conn.execute(query_str, tuple(query_params))
                
                nodes = []
                edges = []
                for row in cursor.fetchall():
                    kind = row[0]
                    if kind == "node":
                        data = {"name": row[2], "type": row[3]}
                        if row[4]:
                            data.update(json.loads(row[4]))
                        nodes.append((row[1], data))
                    else:
                        source_id = row[5]
                        target_id = row[6]
                        relationship_name = row[7]
                        edge_properties_raw = row[8]
                        props = {}
                        if edge_properties_raw:
                            props = json.loads(edge_properties_raw)
                        edges.append((source_id, target_id, relationship_name, props))
                        
                return nodes, edges
            finally:
                conn.close()

        return await self._execute_sync(_get)

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str], node_name_filter_operator: str = "OR"
    ) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, str, str, dict]]]:
        label = node_type.__name__
        
        def _get():
            conn = self._connect()
            try:
                names_placeholders = ",".join("?" for _ in node_name)
                
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
                                HAVING COUNT(DISTINCT primary_id) = ?
                            )"""

                query_str = f"""
                            WITH primary_nodes AS (
                                SELECT DISTINCT id
                                FROM graph_node
                                WHERE type = ? AND name IN ({names_placeholders})
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

                params = [label] + node_name
                if node_name_filter_operator != "OR":
                    params.append(len(node_name))
                    
                cursor = conn.execute(query_str, tuple(params))
                
                nodes = []
                edges = []
                for row in cursor.fetchall():
                    if row[0] == "node":
                        data = {"name": row[2], "type": row[3]}
                        if row[4]:
                            data.update(json.loads(row[4]))
                        nodes.append((row[1], data))
                    else:
                        props = json.loads(row[8]) if row[8] else {}
                        edges.append((row[5], row[6], row[7], props))

                return nodes, edges
            finally:
                conn.close()

        return await self._execute_sync(_get)

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        def _get():
            conn = self._connect()
            try:
                num_nodes = conn.execute("SELECT count(*) FROM graph_node").fetchone()[0]
                num_edges = conn.execute("SELECT count(*) FROM graph_edge").fetchone()[0]

                mean_degree = (2 * num_edges) / num_nodes if num_nodes else None
                edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

                comp_result = conn.execute(
                    """
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
                    """
                )
                comp_rows = comp_result.fetchall()
                num_components = len(comp_rows)
                component_sizes = [row[1] for row in comp_rows]

                metrics = {
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                    "mean_degree": mean_degree,
                    "edge_density": edge_density,
                    "num_components": num_components,
                    "component_sizes": component_sizes,
                    "diameter": -1,
                    "avg_shortest_path": -1,
                    "clustering_coefficient": -1,
                    "self_loops": -1,
                }
                
                if include_optional:
                    self_loops = conn.execute("SELECT count(*) FROM graph_edge WHERE source_id = target_id").fetchone()[0]
                    metrics["self_loops"] = self_loops
                    
                return metrics
            finally:
                conn.close()

        return await self._execute_sync(_get)

    async def get_triplets_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        """Retrieve a batch of (source, relationship, target) triplets.

        Parameters:
        -----------
            offset: Number of triplets to skip.
            limit: Maximum number of triplets to return.

        Returns:
        --------
            A list of dicts with 'start_node', 'relationship_properties',
            and 'end_node' keys.
        """
        if offset < 0:
            raise ValueError(f"Offset must be non-negative, got {offset}")
        if limit < 0:
            raise ValueError(f"Limit must be non-negative, got {limit}")

        def _get():
            conn = self._connect()
            try:
                query_str = """
                    SELECT
                        s.id, s.name, s.type, s.properties,
                        e.relationship_name, e.properties AS edge_props,
                        t.id, t.name, t.type, t.properties
                    FROM graph_edge e
                    JOIN graph_node s ON s.id = e.source_id
                    JOIN graph_node t ON t.id = e.target_id
                    ORDER BY e.source_id, e.target_id, e.relationship_name
                    LIMIT ? OFFSET ?
                """
                cursor = conn.execute(query_str, (limit, offset))
                
                triplets = []
                for row in cursor.fetchall():
                    start_node = {"id": row[0], "name": row[1], "type": row[2]}
                    if row[3]:
                        start_node.update(json.loads(row[3]))
                        
                    rel = {"relationship_name": row[4]}
                    if row[5]:
                        rel.update(json.loads(row[5]))
                        
                    end_node = {"id": row[6], "name": row[7], "type": row[8]}
                    if row[9]:
                        end_node.update(json.loads(row[9]))
                        
                    triplets.append({
                        "start_node": start_node,
                        "relationship_properties": rel,
                        "end_node": end_node,
                    })
                return triplets
            finally:
                conn.close()
                
        return await self._execute_sync(_get)
