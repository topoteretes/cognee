"""Turso graph adapter: two tables (graph_node, graph_edge) on the Turso rewrite engine.

Runs on ``pyturso`` through cognee's ``sqlite+cognee_turso://`` SQLAlchemy dialect
(:mod:`cognee.infrastructure.databases.turso`). The engine rejects a few SQLite
constructs, which shapes the code below:

* no recursive CTEs — k-hop neighborhoods are expanded one hop per query and
  connected components are computed in Python (union-find over the edge list);
* bind parameters must be None, numbers, strings or bytes — raw ``text()``
  statements bind datetimes through a typed ``bindparam``;
* under ``TURSO_JOURNAL_MODE=mvcc`` writes run as ``BEGIN CONCURRENT`` and are
  retried on ``Write-write conflict``; ``initialize()`` runs its DDL exclusively.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, bindparam, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.turso import (
    configure_engine,
    connect_args_for_mode,
    exclusive_transaction,
    get_turso_config,
    retry_on_conflict,
    turso_url,
)
from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.Timestamp import Timestamp
from cognee.modules.engine.utils.generate_timestamp_datapoint import date_to_int
from cognee.modules.storage.utils import JSONEncoder
from cognee.shared.logging_utils import get_logger

from .tables import _edge_table, _meta, _node_table

logger = get_logger()

_WRITE_CHUNK_SIZE = 500


def _in_params(prefix: str, values: list[str]) -> tuple[str, dict[str, str]]:
    """Build named params for a SQLite IN clause of small, bounded lists
    (edge types, node names, filter values). SQLite has no ANY(:list) support.
    """
    params = {f"{prefix}_{i}": v for i, v in enumerate(values)}
    placeholders = ", ".join(f":{prefix}_{i}" for i in range(len(values)))
    return placeholders, params


def _id_subquery(prefix: str, ids: list[str]) -> tuple[str, dict[str, str]]:
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


def _component_sizes(node_ids: list[str], edges: list[tuple[str, str]]) -> list[int]:
    """Sizes of the connected components (undirected), largest first.

    Union-find over the edge list; every node id is its own set until joined,
    so isolated nodes are components of size one.
    """
    parent: dict[str, str] = {node_id: node_id for node_id in node_ids}

    def find(item: str) -> str:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:  # path compression
            parent[item], item = root, parent[item]
        return root

    for source_id, target_id in edges:
        parent.setdefault(source_id, source_id)
        parent.setdefault(target_id, target_id)
        root_a, root_b = find(source_id), find(target_id)
        if root_a != root_b:
            parent[root_b] = root_a

    sizes: dict[str, int] = {}
    for node_id in parent:
        root = find(node_id)
        sizes[root] = sizes.get(root, 0) + 1
    return sorted(sizes.values(), reverse=True)


class TursoAdapter(GraphDBInterface):
    """Graph-as-tables adapter on the Turso rewrite engine, accessed via SQLAlchemy async sessions."""

    # ``query()`` executes SQL against the graph tables, not Cypher.
    supports_cypher_queries = False

    _ALLOWED_FILTER_ATTRS = {"id", "name", "type"}

    def __init__(self, database_path: str) -> None:
        """Create engine and sessionmaker for a local Turso database file (or ``:memory:``)."""
        self.database_path = database_path
        self.turso_config = get_turso_config()
        self.db_uri = turso_url(database_path)
        # Properties are serialized to TEXT columns by _serialize_properties, so
        # there is no JSON-typed column for SQLAlchemy to encode — hence no
        # json_serializer, unlike the Postgres adapter with its JSONB columns.
        #
        # NullPool for on-disk files (matching the relational SqlAlchemyAdapter):
        # per-dataset engines are cached and evicted, and pooling would keep
        # connections/file handles open to a file that eviction may delete. An
        # in-memory database must instead keep its single connection alive, or the
        # schema vanishes between operations, so leave those on the default pool.
        engine_kwargs = {} if ":memory:" in database_path else {"poolclass": NullPool}
        self.engine = create_async_engine(
            self.db_uri, connect_args=connect_args_for_mode(self.turso_config), **engine_kwargs
        )
        # Connection PRAGMAs (journal mode, busy_timeout, foreign_keys=ON so
        # graph_edge's ON DELETE CASCADE fires) and, in mvcc mode, the
        # BEGIN CONCURRENT hook. Connection scoped, so applied on every connect.
        configure_engine(self.engine, foreign_keys=True, config=self.turso_config)

        self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)
        self._write_lock = asyncio.Lock()

    async def close(self) -> None:
        """Dispose connection pool. Called by closing_lru_cache on eviction."""
        await self.engine.dispose(close=True)

    async def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        async with exclusive_transaction(), self.engine.begin() as conn:
            await conn.run_sync(_meta.create_all, checkfirst=True)

    async def _upsert_rows(self, table, index_elements: list[str], rows: list[dict], set_columns):
        """One committed transaction of chunked ``INSERT ... ON CONFLICT DO UPDATE``."""
        async with self._session() as session:
            for i in range(0, len(rows), _WRITE_CHUNK_SIZE):
                chunk = rows[i : i + _WRITE_CHUNK_SIZE]
                stmt = sqlite_insert(table).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=index_elements,
                    set_={column: getattr(stmt.excluded, column) for column in set_columns},
                )
                await session.execute(stmt)
            await session.commit()

    async def _write(self, operation) -> None:
        """Serialize this adapter's writes and retry the whole transaction on a conflict."""
        async with self._write_lock:
            await retry_on_conflict(operation)

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Any]:
        """Yield an async session from the underlying engine."""
        async with self.sessionmaker() as session:
            yield session

    def _serialize_properties(self, props: dict[str, Any]) -> str:
        """Serialize a dict to a JSON string, handling datetimes and UUIDs."""
        return json.dumps(props, cls=JSONEncoder)

    def _parse_node_row(self, row) -> dict[str, Any]:
        """Convert a (id, name, type, properties) row to a merged dict."""
        data = {"id": row.id, "name": row.name, "type": row.type}
        if row.properties is not None:
            props = (
                row.properties if isinstance(row.properties, dict) else json.loads(row.properties)
            )
            data.update(props)
        return data

    async def query(self, query_str: str, params: dict | None = None) -> list[Any]:
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
        self, node: DataPoint | str, properties: dict[str, Any] | None = None
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
        nodes: list[tuple[str, dict]] | list[DataPoint],
        source_ref_key: str | None = None,
        pipeline_run_id: str | None = None,
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
        await self._write(
            lambda: self._upsert_rows(
                _node_table, ["id"], rows, ("name", "type", "properties", "updated_at")
            )
        )

    async def delete_node(self, node_id: str) -> None:
        """Delete a single node. Delegates to delete_nodes."""
        await self.delete_nodes([node_id])

    async def delete_nodes(self, node_ids: list[str]) -> None:
        """Delete multiple nodes by ID. Cascade-deletes connected edges."""
        if not node_ids:
            return
        subquery, params = _id_subquery("did", node_ids)

        async def delete() -> None:
            async with self._session() as session:
                await session.execute(
                    text(f"DELETE FROM graph_node WHERE id IN {subquery}"), params
                )
                await session.commit()

        await self._write(delete)

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Retrieve a single node by ID."""
        results = await self.get_nodes([node_id])
        return results[0] if results else None

    async def get_nodes(self, node_ids: list[str]) -> list[dict[str, Any]]:
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
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Add a single edge. Delegates to add_edges."""
        await self.add_edges(
            [(str(source_id), str(target_id), relationship_name, properties or {})]
        )

    async def add_edges(
        self,
        edges: list[tuple[str, str, str, dict[str, Any] | None]] | list,
        source_ref_key: str | None = None,
        pipeline_run_id: str | None = None,
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
        await self._write(
            lambda: self._upsert_rows(
                _edge_table,
                ["source_id", "target_id", "relationship_name"],
                rows,
                ("properties", "updated_at"),
            )
        )

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """Check whether a single edge exists."""
        result = await self.has_edges([(str(source_id), str(target_id), relationship_name)])
        return len(result) > 0

    async def has_edges(self, edges: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
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

    async def get_edges(self, node_id: str) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
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

    async def get_neighbors(self, node_id: str) -> list[dict[str, Any]]:
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
        self, node_id: str | UUID
    ) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
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

    async def get_top_degree_node_ids(self, top_k: int) -> list[str]:
        """Rank a bounded physical edge prefix, with the same recency bias as Postgres."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        from cognee.infrastructure.databases.graph.degree_seeds import EDGE_SAMPLE_ROWS

        await self.initialize()
        async with self.sessionmaker() as session:
            result = await session.execute(
                text("""
                    WITH sampled_edges AS MATERIALIZED (
                        SELECT source_id, target_id FROM graph_edge LIMIT :sample
                    )
                    SELECT node_id FROM (
                        SELECT source_id AS node_id FROM sampled_edges
                        UNION ALL
                        SELECT target_id AS node_id FROM sampled_edges
                    ) endpoints
                    GROUP BY node_id ORDER BY count(*) DESC, node_id LIMIT :top_k
                """),
                {"sample": EDGE_SAMPLE_ROWS, "top_k": top_k},
            )
            seeds = [str(row[0]) for row in result.all()]
            if len(seeds) < top_k:
                # Fetch at most top_k ids; at most len(seeds) can overlap.
                result = await session.execute(
                    text("SELECT id FROM graph_node LIMIT :top_k"), {"top_k": top_k}
                )
                seeds.extend(str(row[0]) for row in result.all() if str(row[0]) not in seeds)
            return seeds[:top_k]

    async def get_graph_data(
        self,
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, str, str, dict[str, Any]]]]:
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
        self, target_ids: list[str]
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, str, str, dict[str, Any]]]]:
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
        self, attribute_filters: list[dict[str, list[str | int]]]
    ) -> tuple[list[tuple[str, dict]], list[tuple[str, str, str, dict]]]:
        """Retrieve nodes matching attribute filters, plus edges between them."""
        if not attribute_filters:
            return await self.get_graph_data()

        where_parts = []
        params: dict[str, Any] = {}
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
        self, node_type: type[Any], node_name: list[str], node_name_filter_operator: str = "OR"
    ) -> tuple[list[tuple[str, dict]], list[tuple[str, str, str, dict]]]:
        """Retrieve subgraph of matching nodes, their neighbors, and interconnecting edges."""
        if not node_name:
            return [], []
        label = node_type.__name__

        name_ph, name_params = _in_params("nm", node_name)
        params: dict[str, Any] = {**name_params, "label": label}

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

    async def get_graph_metrics(self, include_optional: bool = False) -> dict[str, Any]:
        """Compute graph metrics matching the PostgresDemoAdapter output schema."""
        async with self._session() as session:
            n_result = await session.execute(text("SELECT count(*) FROM graph_node"))
            num_nodes = n_result.scalar()
            e_result = await session.execute(text("SELECT count(*) FROM graph_edge"))
            num_edges = e_result.scalar()

            mean_degree = (2 * num_edges) / num_nodes if num_nodes else None
            edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

            # The Turso engine has no recursive CTEs (0.7.x), so connected
            # components are computed here with union-find over the edge list.
            # Isolated nodes count as components of size one, as before.
            node_ids = [row[0] for row in await session.execute(text("SELECT id FROM graph_node"))]
            edge_rows = (
                await session.execute(text("SELECT source_id, target_id FROM graph_edge"))
            ).fetchall()
            component_sizes = _component_sizes(node_ids, edge_rows)
            num_components = len(component_sizes)

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
        node_ids: list[str],
        depth: int = 1,
        edge_types: list[str] | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, str, str, dict[str, Any]]]]:
        """Get the k-hop neighborhood subgraph around seed nodes."""
        if not node_ids:
            return [], []

        edge_filter = ""
        et_params: dict[str, Any] = {}
        if edge_types:
            et_ph, et_params = _in_params("et", edge_types)
            edge_filter = f"AND e.relationship_name IN ({et_ph})"

        async with self._session() as session:
            # Breadth-first expansion, one query per hop (the engine has no
            # recursive CTEs). Each frontier is bound as a single JSON array, so
            # neither seed count nor hop width hits the variable cap. Edge types
            # only constrain the traversal, as in the recursive version; the
            # returned edge set is every edge between the collected nodes.
            visited = {str(node_id) for node_id in node_ids}
            frontier = set(visited)
            for _hop in range(max(depth, 0)):
                if not frontier:
                    break
                frontier_subquery, frontier_params = _id_subquery("frontier", sorted(frontier))
                result = await session.execute(
                    text(
                        f"""
                        SELECT e.source_id, e.target_id
                        FROM graph_edge e
                        WHERE (e.source_id IN {frontier_subquery}
                               OR e.target_id IN {frontier_subquery})
                          {edge_filter}
                        """
                    ),
                    {**frontier_params, **et_params},
                )
                next_frontier = set()
                for source_id, target_id in result.fetchall():
                    for endpoint in (source_id, target_id):
                        if endpoint not in visited:
                            visited.add(endpoint)
                            next_frontier.add(endpoint)
                frontier = next_frontier

            ids_subquery, ids_params = _id_subquery("ids", sorted(visited))
            node_result = await session.execute(
                text(
                    f"SELECT id, name, type, properties FROM graph_node WHERE id IN {ids_subquery}"
                ),
                ids_params,
            )
            nodes = []
            for row in node_result.fetchall():
                data = self._parse_node_row(row)
                data.pop("id", None)
                nodes.append((row.id, data))

            edge_result = await session.execute(
                text(
                    f"""
                    SELECT source_id, target_id, relationship_name, properties
                    FROM graph_edge
                    WHERE source_id IN {ids_subquery} AND target_id IN {ids_subquery}
                    """
                ),
                ids_params,
            )
            edges = []
            for row in edge_result.fetchall():
                props = {}
                if row.properties is not None:
                    props = (
                        row.properties
                        if isinstance(row.properties, dict)
                        else json.loads(row.properties)
                    )
                edges.append((row.source_id, row.target_id, row.relationship_name, props))

            return nodes, edges

    # ------------------------------------------------------------------ #
    # Temporal retrieval (SearchType.TEMPORAL), mirroring the Ladybug/Neo4j
    # adapters: Timestamp nodes carry ``time_at`` (ms since the epoch) in their
    # properties; Event nodes sit within two hops of their timestamps.
    # ------------------------------------------------------------------ #
    async def collect_time_ids(
        self,
        time_from: Timestamp | None = None,
        time_to: Timestamp | None = None,
    ) -> list[str]:
        """Return ids of ``Timestamp`` nodes whose ``time_at`` lies in the inclusive range.

        Either bound may be omitted; with neither, nothing is selected (as in the
        Ladybug adapter). A Timestamp without a numeric ``time_at`` is skipped.
        """
        if not time_from and not time_to:
            return []

        conditions = ["type = 'Timestamp'", "json_extract(properties, '$.time_at') IS NOT NULL"]
        params: dict[str, Any] = {}
        if time_from:
            conditions.append("CAST(json_extract(properties, '$.time_at') AS INTEGER) >= :lower")
            params["lower"] = date_to_int(time_from)
        if time_to:
            conditions.append("CAST(json_extract(properties, '$.time_at') AS INTEGER) <= :upper")
            params["upper"] = date_to_int(time_to)

        async with self._session() as session:
            result = await session.execute(
                text(f"SELECT id FROM graph_node WHERE {' AND '.join(conditions)}"), params
            )
            return [row[0] for row in result.fetchall()]

    async def collect_events(self, ids: list[str] | str) -> list[dict[str, Any]]:
        """Collect the ``Event`` nodes within one or two hops of ``ids``.

        Same contract as the Ladybug adapter: ``[{"events": [...]}]`` where each
        event has ``id``, ``name``, ``description`` and, when set, ``location``.
        ``ids`` may also be the comma-joined string form the Neo4j path produces.
        """
        if isinstance(ids, str):
            ids = [uid.strip().strip("'\"") for uid in ids.split(",") if uid.strip()]
        seeds = {str(uid) for uid in ids}
        if not seeds:
            return [{"events": []}]

        nodes, _ = await self.get_neighborhood(sorted(seeds), depth=2)
        events = []
        for node_id, data in nodes:
            if node_id in seeds or data.get("type") != "Event":
                continue
            event: dict[str, Any] = {
                "id": node_id,
                "name": data.get("name"),
                "description": data.get("description"),
            }
            if data.get("location"):
                event["location"] = data["location"]
            events.append(event)
        return [{"events": events}]

    async def delete_graph(self) -> None:
        """Delete all nodes and edges from the graph."""
        await self.initialize()

        async def wipe() -> None:
            async with self._session() as session:
                await session.execute(text("DELETE FROM graph_edge"))
                await session.execute(text("DELETE FROM graph_node"))
                await session.commit()

        await self._write(wipe)

    async def get_triplets_batch(self, offset: int, limit: int) -> list[dict[str, Any]]:
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
        tags: list[str],
        node_ids: list[str] | None = None,
    ) -> None:
        """Strip ``tags`` from each node's ``belongs_to_set`` property array.

        Keeps the denormalized membership list consistent with the additive
        belongs_to_set edges after a NodeSet (or its dataset) is deleted.
        ``belongs_to_set`` lives inside the JSON ``properties`` TEXT blob, so this
        is a read-filter-write over that array. Mirrors the Postgres adapter.
        """
        if not tags:
            return
        if node_ids is not None and not node_ids:
            return

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
            # The engine binds only None/numbers/str/bytes; the typed bindparam
            # renders the datetime the way the DateTime column stores it.
            update_stmt = text(
                "UPDATE graph_node SET properties = :p, updated_at = :now WHERE id = :id"
            ).bindparams(bindparam("now", type_=DateTime(timezone=True)))

            async def apply_updates() -> None:
                async with self._session() as session:
                    for update in updates:
                        await session.execute(
                            update_stmt,
                            {"id": update["id"], "p": update["properties"], "now": now},
                        )
                    await session.commit()

            await self._write(apply_updates)
        return
