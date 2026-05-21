"""Postgres graph adapter using two tables (graph_node, graph_edge) over SQLAlchemy + asyncpg."""

import json
from uuid import UUID
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, Union, Optional, Tuple, Type

from sqlalchemy import text, values, select, exists, func, String
from sqlalchemy import column as sa_column
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.storage.utils import JSONEncoder

from .tables import _meta, _node_table, _edge_table

logger = get_logger()


class PostgresAdapter(GraphDBInterface):
    """Graph-as-tables adapter backed by Postgres, accessed via SQLAlchemy async sessions."""

    _ALLOWED_FILTER_ATTRS = {"id", "name", "type"}

    def __init__(self, connection_string: str) -> None:
        """Create engine and sessionmaker from a Postgres connection string."""
        self.db_uri = connection_string
        self.engine = create_async_engine(self.db_uri)
        self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)

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
            "The Postgres graph backend does not support raw Cypher queries. "
            "Use a graph-native backend (Neo4j, Ladybug) for raw query support, "
            "or use the typed adapter methods (add_nodes, get_neighbors, etc.)."
        )

    async def is_empty(self) -> bool:
        """Check whether the graph contains any nodes.

        Returns:
        --------
            bool: True if the graph has no nodes.
        """
        await self.initialize()
        async with self._session() as session:
            result = await session.execute(text("SELECT EXISTS(SELECT 1 FROM graph_node LIMIT 1)"))
            return not result.scalar()

    async def add_node(
        self, node: Union[DataPoint, str], properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a single node. Delegates to add_nodes.

        Parameters:
        -----------
            node: A DataPoint instance or a string node ID.
            properties: Optional property dict when node is a string ID.
        """
        if isinstance(node, str):
            props = properties or {}
            props.setdefault("id", node)
            await self.add_nodes([(node, props)])
        else:
            await self.add_nodes([node])

    async def add_nodes(self, nodes: Union[List[Tuple[str, Dict]], List[DataPoint]]) -> None:
        """Add multiple nodes via batch upsert.

        Parameters:
        -----------
            nodes: A list of (id, properties) tuples or DataPoint instances.
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
                    "properties": json.loads(json.dumps(extra, cls=JSONEncoder)),
                    "created_at": now,
                    "updated_at": now,
                }
            )

        # Deduplicate by id (last wins) to avoid ON CONFLICT errors within one batch
        rows = list({r["id"]: r for r in rows}.values())

        stmt = pg_insert(_node_table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "name": stmt.excluded.name,
                "type": stmt.excluded.type,
                "properties": stmt.excluded.properties,
                "updated_at": func.now(),
            },
        )

        async with self._session() as session:
            await session.execute(stmt)
            await session.commit()

    async def delete_node(self, node_id: str) -> None:
        """Delete a single node. Delegates to delete_nodes.

        Parameters:
        -----------
            node_id: The ID of the node to delete.
        """
        await self.delete_nodes([node_id])

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """Delete multiple nodes by ID. Cascade-deletes connected edges.

        Parameters:
        -----------
            node_ids: List of node IDs to delete.
        """
        if not node_ids:
            return
        async with self._session() as session:
            await session.execute(
                text("DELETE FROM graph_node WHERE id = ANY(:ids)"), {"ids": node_ids}
            )
            await session.commit()

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single node by ID.

        Parameters:
        -----------
            node_id: The ID of the node to retrieve.

        Returns:
        --------
            A property dict for the node, or None if not found.
        """
        results = await self.get_nodes([node_id])
        return results[0] if results else None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple nodes by ID.

        Parameters:
        -----------
            node_ids: List of node IDs to retrieve.

        Returns:
        --------
            A list of property dicts, one per found node.
        """
        if not node_ids:
            return []
        async with self._session() as session:
            result = await session.execute(
                text("SELECT id, name, type, properties FROM graph_node WHERE id = ANY(:ids)"),
                {"ids": node_ids},
            )
            return [self._parse_node_row(row) for row in result.fetchall()]

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single edge. Delegates to add_edges.

        Parameters:
        -----------
            source_id: Source node ID.
            target_id: Target node ID.
            relationship_name: The edge label.
            properties: Optional property dict for the edge.
        """
        await self.add_edges(
            [(str(source_id), str(target_id), relationship_name, properties or {})]
        )

    async def add_edges(
        self, edges: Union[List[Tuple[str, str, str, Optional[Dict[str, Any]]]], List]
    ) -> None:
        """Add multiple edges via batch upsert.

        Parameters:
        -----------
            edges: A list of (source_id, target_id, relationship_name, properties) tuples.
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
                    "properties": json.loads(self._serialize_properties(raw_props)),
                    "created_at": now,
                    "updated_at": now,
                }
            )

        # Deduplicate by composite key (last wins) to avoid ON CONFLICT errors within one batch
        rows = list(
            {(r["source_id"], r["target_id"], r["relationship_name"]): r for r in rows}.values()
        )

        stmt = pg_insert(_edge_table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["source_id", "target_id", "relationship_name"],
            set_={
                "properties": stmt.excluded.properties,
                "updated_at": func.now(),
            },
        )

        async with self._session() as session:
            await session.execute(stmt)
            await session.commit()

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """Check whether a single edge exists.

        Parameters:
        -----------
            source_id: Source node ID.
            target_id: Target node ID.
            relationship_name: The edge label.

        Returns:
        --------
            True if the edge exists.
        """
        result = await self.has_edges([(str(source_id), str(target_id), relationship_name)])
        return len(result) > 0

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Check which of the given edges exist.

        Parameters:
        -----------
            edges: A list of (source_id, target_id, relationship_name) tuples to check.

        Returns:
        --------
            The subset of input tuples that exist in the database.
        """
        if not edges:
            return []

        candidates = values(
            sa_column("src", String),
            sa_column("tgt", String),
            sa_column("rel", String),
            name="q",
        ).data([(str(s), str(t), str(r)) for s, t, r in edges])

        stmt = select(candidates.c.src, candidates.c.tgt, candidates.c.rel).where(
            exists(
                select(text("1"))
                .select_from(_edge_table)
                .where(_edge_table.c.source_id == candidates.c.src)
                .where(_edge_table.c.target_id == candidates.c.tgt)
                .where(_edge_table.c.relationship_name == candidates.c.rel)
            )
        )

        async with self._session() as session:
            result = await session.execute(stmt)
            return [(row[0], row[1], row[2]) for row in result.fetchall()]

    async def get_edges(self, node_id: str) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        """Retrieve all edges connected to a node.

        Parameters:
        -----------
            node_id: The ID of the node.

        Returns:
        --------
            A list of (source_dict, relationship_name, target_dict) tuples.
        """
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
        """Retrieve all nodes directly connected to a given node.

        Parameters:
        -----------
            node_id: The ID of the node.

        Returns:
        --------
            A list of property dicts for neighboring nodes.
        """
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
        """Retrieve all connections (source, edge, target) for a node.

        Parameters:
        -----------
            node_id: The ID of the node.

        Returns:
        --------
            A list of (source_dict, edge_dict, target_dict) tuples.
        """
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
        """Retrieve all nodes and edges in the graph.

        Returns:
        --------
            A tuple of (nodes, edges) where nodes are (id, props) and
            edges are (source_id, target_id, relationship_name, props).
        """
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
        """Retrieve nodes matching attribute filters, plus edges between them.

        Parameters:
        -----------
            attribute_filters: A list of {attr: [values]} dicts. Only 'id',
                'name', and 'type' are valid filter attributes.

        Returns:
        --------
            A tuple of (nodes, edges) matching the filters.
        """
        if not attribute_filters:
            return await self.get_graph_data()

        # Validate attribute names against whitelist to prevent SQL injection
        where_parts = []
        params = {}
        for i, filter_dict in enumerate(attribute_filters):
            for attr, filter_values in filter_dict.items():
                if attr not in self._ALLOWED_FILTER_ATTRS:
                    raise ValueError(f"Invalid filter attribute: {attr!r}")
                param = f"filt_{i}_{attr}"
                where_parts.append(f"n.{attr} = ANY(:{param})")
                params[param] = filter_values

        if not where_parts:
            return await self.get_graph_data()

        where_clause = " AND ".join(where_parts)

        async with self._session() as session:
            result = await session.execute(
                text(f"""
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
                """),
                params,
            )

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

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str], node_name_filter_operator: str = "OR"
    ) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, str, str, dict]]]:
        """Retrieve a subgraph containing matching nodes, their neighbors, and interconnecting edges.

        Parameters:
        -----------
            node_type: The DataPoint subclass whose __name__ is the type label.
            node_name: List of node names to match.

        Returns:
        --------
            A tuple of (nodes, edges) for the subgraph.
        """
        label = node_type.__name__

        # OR: neighbor of any primary node qualifies
        # AND: neighbor must be connected to every primary node
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
                        WHERE type = :label AND name = ANY(:names)
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

        params = {"label": label, "names": node_name}
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
        """Compute graph metrics (node/edge counts, degree, density, components).

        Parameters:
        -----------
            include_optional: If True, also compute self-loop count.

        Returns:
        --------
            A dict of metric names to values. Diameter, avg shortest path,
            and clustering return -1 (not computed).
        """
        async with self._session() as session:
            n_result = await session.execute(text("SELECT count(*) FROM graph_node"))
            num_nodes = n_result.scalar()
            e_result = await session.execute(text("SELECT count(*) FROM graph_edge"))
            num_edges = e_result.scalar()

            mean_degree = (2 * num_edges) / num_nodes if num_nodes else None
            edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

            # Connected components via recursive CTE
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
        """Get the k-hop neighborhood subgraph around seed nodes.

        Uses a single recursive CTE query to collect all node IDs within
        `depth` hops, then returns nodes and edges for that subgraph.
        """
        if not node_ids:
            return [], []

        # Optional edge type filter for the CTE traversal
        edge_filter = ""
        if edge_types:
            placeholders = ", ".join(f":et_{i}" for i in range(len(edge_types)))
            edge_filter = f"AND e.relationship_name IN ({placeholders})"

        # Single query: recursive CTE finds reachable IDs, then joins
        # nodes and edges in two unioned result sets distinguished by 'kind'
        query_str = f"""
            WITH RECURSIVE neighborhood(id, hops) AS (
                SELECT unnest(:seeds), 0
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

        params: Dict[str, Any] = {"seeds": list(node_ids), "depth": depth}
        if edge_types:
            for i, et in enumerate(edge_types):
                params[f"et_{i}"] = et

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
        async with self._session() as session:
            await session.execute(text("TRUNCATE graph_edge, graph_node CASCADE"))
            await session.commit()

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
                    OFFSET :off LIMIT :lim
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
