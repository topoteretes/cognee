"""Postgres graph adapter using two tables (graph_node, graph_edge) over SQLAlchemy + asyncpg.

DEMO: Using Postgres as a graph store is currently a demo feature and is not
production-ready. Use it to demo keeping relational metadata, PGVector, and graph
state in a single Postgres service, but rely on a graph-native backend such as Kuzu or Neo4j
for production workloads.

Interested in further development or production use of Postgres as a graph database? Write to
us at social@cognee.ai to explore the options.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from uuid import UUID
from typing import Callable, Dict, Any, List, Union, Optional, Tuple, Type

from sqlalchemy import NullPool, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.relational import get_relational_config
from cognee.modules.storage.utils import JSONEncoder
from cognee.modules.graph.methods.sanitize_relational_payload import sanitize_relational_payload
from cognee.infrastructure.databases.provenance import (
    EdgeDeleteData,
    EdgeIdentity,
    NodeDeleteData,
)
from cognee.infrastructure.databases.provenance.source_refs import (
    get_dataset_id_from_source_ref_key,
    get_pipeline_run_id_from_source_run_ref,
    get_source_ref_key_from_source_run_ref,
)
from cognee.infrastructure.databases.provenance.source_ref_state import (
    ProvenanceColumns,
    provenance_after_attach,
    provenance_after_remove,
)

from .tables import _meta


def _prepare_node_rows(
    nodes: Union[List[Tuple[str, Dict]], List[DataPoint]],
) -> list[dict[str, Any]]:
    """Copy, sanitize, deduplicate, and sort nodes for one database write."""
    rows_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if isinstance(node, tuple):
            properties = {**(node[1] or {}), "id": node[0]}
        elif hasattr(node, "model_dump"):
            properties = dict(node.model_dump())
        else:
            properties = dict(vars(node))

        node_id = sanitize_relational_payload(str(properties.get("id", "")))
        extra = {
            key: value for key, value in properties.items() if key not in {"id", "name", "type"}
        }
        rows_by_id[node_id] = {
            "id": node_id,
            "name": sanitize_relational_payload(str(properties.get("name", ""))),
            "type": sanitize_relational_payload(str(properties.get("type", ""))),
            "properties": json.dumps(sanitize_relational_payload(extra), cls=JSONEncoder),
        }

    return [rows_by_id[node_id] for node_id in sorted(rows_by_id)]


def _prepare_edge_rows(
    edges: List[Tuple[str, str, str, Optional[Dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Copy, sanitize, deduplicate, and sort edges for one database write."""
    rows_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source_id, target_id, relationship_name, *property_values in edges:
        identity = (
            sanitize_relational_payload(str(source_id)),
            sanitize_relational_payload(str(target_id)),
            sanitize_relational_payload(str(relationship_name)),
        )
        properties = property_values[0] if property_values and property_values[0] else {}
        rows_by_identity[identity] = {
            "source_id": identity[0],
            "target_id": identity[1],
            "relationship_name": identity[2],
            "properties": json.dumps(
                sanitize_relational_payload(dict(properties)), cls=JSONEncoder
            ),
        }

    return [rows_by_identity[identity] for identity in sorted(rows_by_identity)]


def _edge_identities(edges: list[EdgeIdentity]) -> list[tuple[str, str, str]]:
    """Deduplicate edge identities, sorted so concurrent writers lock rows in one order."""
    return sorted(
        {(str(edge.source_id), str(edge.target_id), str(edge.relationship_name)) for edge in edges}
    )


# Every write in this adapter takes this one advisory lock first, so two write
# transactions never overlap. Row locks alone are not enough: concurrent batches
# inserting the same node wait on each other's unique index entry, and a cascading
# delete waits on rows in a different order, which Postgres reports as a deadlock.
# An xact-level advisory lock is released on commit and on rollback, so there is no
# unlock path to forget. Reads never take it.
_GRAPH_WRITE_LOCK_KEY = 5522063


async def _lock_graph_writes(session: AsyncSession) -> None:
    """Serialize graph writes for the rest of this transaction."""
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:key)"), {"key": _GRAPH_WRITE_LOCK_KEY}
    )


def _decode_properties(value: Any) -> dict[str, Any]:
    """Return a new dictionary for a JSONB value."""
    if not value:
        return {}
    return dict(value) if isinstance(value, dict) else json.loads(value)


def _component_sizes(node_ids: list[str], edge_pairs: list[tuple[str, str]]) -> list[int]:
    """Return the sizes of undirected connected components, largest first."""
    neighbors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for source_id, target_id in edge_pairs:
        neighbors.setdefault(source_id, set()).add(target_id)
        neighbors.setdefault(target_id, set()).add(source_id)

    unseen = set(node_ids)
    sizes = []
    while unseen:
        pending = [unseen.pop()]
        size = 0
        while pending:
            node_id = pending.pop()
            size += 1
            new_neighbors = neighbors.get(node_id, set()) & unseen
            unseen.difference_update(new_neighbors)
            pending.extend(new_neighbors)
        sizes.append(size)

    return sorted(sizes, reverse=True)


def _select_nodeset_neighbor_ids(
    primary_ids: set[str], edge_pairs: list[tuple[str, str]], operator: str
) -> set[str]:
    """Select neighbors connected to any or every primary node."""
    if operator not in {"OR", "AND"}:
        raise ValueError("node_name_filter_operator must be 'OR' or 'AND'")
    if not primary_ids:
        return set()

    connected_primaries: dict[str, set[str]] = {}
    for source_id, target_id in edge_pairs:
        if source_id in primary_ids:
            connected_primaries.setdefault(target_id, set()).add(source_id)
        if target_id in primary_ids:
            connected_primaries.setdefault(source_id, set()).add(target_id)

    if operator == "OR":
        return set(connected_primaries)
    return {
        node_id
        for node_id, connections in connected_primaries.items()
        if connections >= primary_ids
    }


_DEFAULT_POOL_ARGS = {
    "pool_size": 2,
    "max_overflow": 20,  # 22-connection ceiling, PER DATASET
    "pool_pre_ping": True,
    "pool_recycle": 280,
    "pool_timeout": 280,
}


def _resolve_engine_args(configured_pool_args) -> dict:
    """Turn the configured POOL_ARGS into create_async_engine kwargs.

    Defaults mirror PGVectorAdapter._ACCESS_CONTROL_DEFAULT_POOL_ARGS (2/20),
    NOT SqlAlchemyAdapter's 5/35. Like PGVector's, this engine is created per
    dataset — the engine-cache key includes graph_database_schema — so its
    ceiling is multiplied by dataset count, not paid once. dict() handles the
    config's tuple-of-pairs form (relational/config.py stores POOL_ARGS as
    tuple(sorted(parsed.items()))).

    Until now only ``poolclass == "nullpool"`` was honoured and every sizing key
    was dropped, so the adapter always ran SQLAlchemy's stock 5/10/30s pool.
    """
    pool_args = dict(configured_pool_args or {})
    if str(pool_args.get("poolclass", "")).lower() == "nullpool":
        return {"poolclass": NullPool}
    pool_args.pop("poolclass", None)
    for key, value in _DEFAULT_POOL_ARGS.items():
        pool_args.setdefault(key, value)
    return pool_args


class PostgresDemoAdapter(GraphDBInterface):
    """Reference graph adapter using one node table and one directed-edge table."""

    supports_cypher_queries = False

    # Chunk-level incremental updates: get_connections yields the true edge
    # endpoints, provenance lives in-graph, and update_chunk_index below is the
    # narrow single-property move the incremental path requires.
    supports_incremental_chunk_updates = True

    _ALLOWED_FILTER_ATTRS = {"id", "name", "type"}

    def __init__(self, connection_string: str, schema: str = "") -> None:
        """Create engine and sessionmaker from a Postgres connection string.

        When ``schema`` is set (shared-database isolation mode) the connection's
        ``search_path`` is pinned to that schema so the two fixed tables
        (``graph_node``/``graph_edge``) and every unqualified query resolve
        inside the dataset's own schema. This lets many datasets share one
        database — no per-dataset database and no rewriting of the adapter's
        SQL. The path is the dataset schema only (``pg_catalog`` is always
        searched implicitly for built-ins), so neither reads nor
        ``create_all(checkfirst=True)`` can ever fall through to ``public``.
        """
        self.db_uri = connection_string
        self.schema = schema or ""

        relational_config = get_relational_config()
        engine_args = _resolve_engine_args(relational_config.pool_args)
        connect_args: dict = (
            dict(relational_config.database_connect_args)
            if relational_config.database_connect_args
            else {}
        )

        if self.schema:
            server_settings = dict(connect_args.get("server_settings") or {})
            server_settings["search_path"] = self.schema
            connect_args["server_settings"] = server_settings

        self.engine = create_async_engine(
            self.db_uri,
            json_serializer=lambda obj: json.dumps(obj, cls=JSONEncoder),
            connect_args=connect_args,
            **engine_args,
        )
        self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)
        self._write_gate = None
        self._write_gate_loop = None
        self._initialized = False
        self._init_lock = None
        self._init_lock_loop = None

    def _get_write_gate(self) -> asyncio.Lock:
        """Per-running-loop write gate.

        The adapter is process-cached (closing_lru_cache) and can outlive the
        loop it was built on; an asyncio.Lock binds to the first loop that
        awaits it, so an __init__-time Lock is a latent "bound to a different
        event loop" RuntimeError under repeated asyncio.run (CLI, unit tests).
        Correctness never depends on this lock — pg_advisory_xact_lock does —
        so a fresh gate per loop is safe.
        """
        loop = asyncio.get_running_loop()
        if self._write_gate_loop is not loop:
            self._write_gate = asyncio.Lock()
            self._write_gate_loop = loop
        return self._write_gate

    def _get_init_lock(self) -> asyncio.Lock:
        """Per-running-loop lock for initialize(); same rationale as the write gate."""
        loop = asyncio.get_running_loop()
        if self._init_lock_loop is not loop:
            self._init_lock = asyncio.Lock()
            self._init_lock_loop = loop
        return self._init_lock

    @asynccontextmanager
    async def _write_session(self):
        """One serialized graph-write transaction.

        Writes are already serialized cluster-wide by pg_advisory_xact_lock.
        Without this process-local gate every concurrent writer checks a
        connection out of the pool and *then* blocks on that advisory lock, so
        N in-flight writers pin min(N, pool ceiling) connections while exactly
        one makes progress — starving the adapter's own reads (see
        get_graph_metadata). The gate puts the queue in front of the pool
        instead of behind it. It is an optimization, not the correctness
        mechanism: the advisory lock still orders writes across processes.
        """
        async with self._get_write_gate():
            async with self.sessionmaker() as session:
                await _lock_graph_writes(session)
                yield session

    async def close(self) -> None:
        """Dispose the database engine."""
        await self.engine.dispose(close=True)
        self._initialized = False

    async def initialize(self) -> None:
        """Create the graph schema when absent. Idempotent per adapter.

        Every get_graph_engine() builds a fresh _GraphEngineHandle whose
        _ensure_initialized calls this, and get_graph_metadata()/is_empty()
        call it again — so a 164-item batch ran hundreds of
        create_all(checkfirst=True) reflections, each holding a pooled
        connection for a stack of catalog round-trips. That is what exhausted
        the pool on the read path (job 98411831471).
        """
        if self._initialized:
            return
        async with self._get_init_lock():
            if self._initialized:
                return
            async with self.engine.begin() as conn:
                await conn.run_sync(_meta.create_all, checkfirst=True)
            self._initialized = True

    async def query(self, query_str: str, params: Optional[dict] = None) -> List[Any]:
        """Reject raw Cypher; callers must use the typed graph methods."""
        raise NotImplementedError(
            "The Postgres graph backend does not support raw Cypher queries. "
            "Use a Cypher-capable graph backend (Neo4j, Ladybug) for raw query support, "
            "or use the typed adapter methods (add_nodes, get_neighbors, etc.)."
        )

    async def is_empty(self) -> bool:
        """Return whether the graph contains no nodes."""
        await self.initialize()
        async with self.sessionmaker() as session:
            result = await session.execute(text("SELECT EXISTS(SELECT 1 FROM graph_node LIMIT 1)"))
            return not result.scalar()

    async def add_node(
        self, node: Union[DataPoint, str], properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add one node, given either a DataPoint or a node id with properties."""
        if isinstance(node, str):
            await self.add_nodes([(node, properties or {})])
        else:
            await self.add_nodes([node])

    async def add_nodes(
        self,
        nodes: Union[List[Tuple[str, Dict]], List[DataPoint]],
        source_ref_key: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> None:
        """Add or replace nodes, optionally attaching one provenance reference."""
        if not nodes:
            return

        rows = _prepare_node_rows(nodes)
        # Provenance columns are absent here on purpose: a rewrite without a
        # source ref must not erase existing ownership.
        upsert = text("""
            INSERT INTO graph_node (id, name, type, properties, created_at, updated_at)
            VALUES (:id, :name, :type, CAST(:properties AS jsonb), now(), now())
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                type = EXCLUDED.type,
                properties = EXCLUDED.properties,
                updated_at = now()
        """)
        async with self._write_session() as session:
            await session.execute(upsert, rows)
            if source_ref_key is not None:
                await self._update_node_provenance(
                    session,
                    [row["id"] for row in rows],
                    lambda keys, run_refs: provenance_after_attach(
                        keys, run_refs, [source_ref_key], pipeline_run_id
                    ),
                )
            await session.commit()

    async def update_chunk_index(self, chunk_indexes: dict) -> None:
        """Set ONLY ``chunk_index`` on the given chunk nodes.

        Node properties live in one JSONB column, so the move is a ``jsonb_set``
        of that single key; name, type, every other property and the provenance
        columns are untouched. Missing ids are skipped, like ``get_nodes``.
        """
        if not chunk_indexes:
            return
        rows = [
            {"id": str(node_id), "chunk_index": int(chunk_index)}
            for node_id, chunk_index in chunk_indexes.items()
        ]
        statement = text("""
            UPDATE graph_node
            SET properties = jsonb_set(
                    COALESCE(properties, '{}'::jsonb),
                    '{chunk_index}',
                    to_jsonb(CAST(:chunk_index AS integer)),
                    true
                ),
                updated_at = now()
            WHERE id = :id
        """)
        async with self.sessionmaker() as session:
            await _lock_graph_writes(session)
            await session.execute(statement, rows)
            await session.commit()

    async def delete_node(self, node_id: str) -> None:
        """Delete one node. Delegates to delete_nodes."""
        await self.delete_nodes([node_id])

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """Delete nodes by id; the schema's foreign keys remove their incident edges."""
        if not node_ids:
            return
        async with self._write_session() as session:
            await session.execute(
                text("DELETE FROM graph_node WHERE id = ANY(:ids)"),
                {"ids": [str(node_id) for node_id in node_ids]},
            )
            await session.commit()

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return one flat node dictionary, or None when the node does not exist."""
        results = await self.get_nodes([node_id])
        return results[0] if results else None

    async def has_node(self, node_id: str) -> bool:
        """Return True when a node with the given id exists."""
        async with self.sessionmaker() as session:
            result = await session.execute(
                text("SELECT EXISTS(SELECT 1 FROM graph_node WHERE id = :id)"),
                {"id": str(node_id)},
            )
            return bool(result.scalar())

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Return flat node dictionaries, omitting ids that do not exist."""
        if not node_ids:
            return []
        async with self.sessionmaker() as session:
            result = await session.execute(
                text("SELECT id, name, type, properties FROM graph_node WHERE id = ANY(:ids)"),
                {"ids": [str(node_id) for node_id in node_ids]},
            )
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "type": row["type"],
                    **_decode_properties(row["properties"]),
                }
                for row in result.mappings().all()
            ]

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add one directed edge. Delegates to add_edges."""
        await self.add_edges(
            [(str(source_id), str(target_id), relationship_name, properties or {})]
        )

    async def add_edges(
        self,
        edges: Union[List[Tuple[str, str, str, Optional[Dict[str, Any]]]], List],
        source_ref_key: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> None:
        """Add or replace edges, optionally attaching one provenance reference."""
        if not edges:
            return

        rows = _prepare_edge_rows(edges)
        # Provenance columns are absent here on purpose: a rewrite without a
        # source ref must not erase existing ownership.
        upsert = text("""
            INSERT INTO graph_edge (
                source_id, target_id, relationship_name, properties, created_at, updated_at
            )
            VALUES (
                :source_id, :target_id, :relationship_name,
                CAST(:properties AS jsonb), now(), now()
            )
            ON CONFLICT (source_id, target_id, relationship_name) DO UPDATE SET
                properties = EXCLUDED.properties,
                updated_at = now()
        """)
        async with self._write_session() as session:
            await session.execute(upsert, rows)
            if source_ref_key is not None:
                await self._update_edge_provenance(
                    session,
                    [
                        EdgeIdentity(
                            source_id=row["source_id"],
                            target_id=row["target_id"],
                            relationship_name=row["relationship_name"],
                        )
                        for row in rows
                    ],
                    lambda keys, run_refs: provenance_after_attach(
                        keys, run_refs, [source_ref_key], pipeline_run_id
                    ),
                )
            await session.commit()

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """Return whether one directed edge triple exists."""
        result = await self.has_edges([(str(source_id), str(target_id), relationship_name)])
        return len(result) > 0

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Return the subset of the requested directed triples that exist."""
        if not edges:
            return []

        found: List[Tuple[str, str, str]] = []
        statement = text("""
            SELECT EXISTS(
                SELECT 1 FROM graph_edge
                WHERE source_id = :source_id
                  AND target_id = :target_id
                  AND relationship_name = :relationship_name
            )
        """)
        async with self.sessionmaker() as session:
            for source_id, target_id, relationship_name in edges:
                identity = (str(source_id), str(target_id), str(relationship_name))
                result = await session.execute(
                    statement,
                    {
                        "source_id": identity[0],
                        "target_id": identity[1],
                        "relationship_name": identity[2],
                    },
                )
                if result.scalar():
                    found.append(identity)

        return found

    async def get_edges(self, node_id: str) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        """Return every incident edge with its directed source and target nodes."""
        rows = await self._fetch_incident_edge_rows(str(node_id))
        edges = []
        for row in rows:
            source = {
                "id": row["source_id"],
                "name": row["source_name"],
                "type": row["source_type"],
                **_decode_properties(row["source_properties"]),
            }
            target = {
                "id": row["target_id"],
                "name": row["target_name"],
                "type": row["target_type"],
                **_decode_properties(row["target_properties"]),
            }
            edges.append((source, row["relationship_name"], target))
        return edges

    async def _fetch_incident_edge_rows(self, node_id: str) -> list[dict[str, Any]]:
        """Read incident edges together with both endpoint nodes."""
        async with self.sessionmaker() as session:
            result = await session.execute(
                text("""
                    SELECT
                        source.id AS source_id,
                        source.name AS source_name,
                        source.type AS source_type,
                        source.properties AS source_properties,
                        edge.relationship_name,
                        edge.properties AS edge_properties,
                        target.id AS target_id,
                        target.name AS target_name,
                        target.type AS target_type,
                        target.properties AS target_properties
                    FROM graph_edge AS edge
                    JOIN graph_node AS source ON source.id = edge.source_id
                    JOIN graph_node AS target ON target.id = edge.target_id
                    WHERE edge.source_id = :node_id OR edge.target_id = :node_id
                """),
                {"node_id": node_id},
            )
            return list(result.mappings().all())

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Return unique incident neighbors, including the node for a self-loop."""
        requested_id = str(node_id)
        rows = await self._fetch_incident_edge_rows(requested_id)
        neighbors: dict[str, dict[str, Any]] = {}
        for row in rows:
            prefix = "target" if row["source_id"] == requested_id else "source"
            neighbor = {
                "id": row[f"{prefix}_id"],
                "name": row[f"{prefix}_name"],
                "type": row[f"{prefix}_type"],
                **_decode_properties(row[f"{prefix}_properties"]),
            }
            neighbors[neighbor["id"]] = neighbor
        return list(neighbors.values())

    async def get_connections(
        self, node_id: Union[str, UUID]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """Return every incident source-edge-target connection."""
        rows = await self._fetch_incident_edge_rows(str(node_id))
        connections = []
        for row in rows:
            source = {
                "id": row["source_id"],
                "name": row["source_name"],
                "type": row["source_type"],
                **_decode_properties(row["source_properties"]),
            }
            edge = {
                "relationship_name": row["relationship_name"],
                **_decode_properties(row["edge_properties"]),
            }
            target = {
                "id": row["target_id"],
                "name": row["target_name"],
                "type": row["target_type"],
                **_decode_properties(row["target_properties"]),
            }
            connections.append((source, edge, target))
        return connections

    @staticmethod
    async def _fetch_nodes_by_id(
        session: AsyncSession, node_ids: list[str]
    ) -> list[tuple[str, dict[str, Any]]]:
        if not node_ids:
            return []
        result = await session.execute(
            text("SELECT id, name, type, properties FROM graph_node WHERE id = ANY(:ids)"),
            {"ids": node_ids},
        )
        nodes = []
        for row in result.mappings().all():
            properties = {
                "name": row["name"],
                "type": row["type"],
                **_decode_properties(row["properties"]),
            }
            nodes.append((row["id"], properties))
        return nodes

    @staticmethod
    async def _fetch_edges_touching(
        session: AsyncSession, node_ids: list[str]
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        if not node_ids:
            return []
        result = await session.execute(
            text("""
                SELECT source_id, target_id, relationship_name, properties
                FROM graph_edge
                WHERE source_id = ANY(:ids) OR target_id = ANY(:ids)
            """),
            {"ids": node_ids},
        )
        return [
            (
                row["source_id"],
                row["target_id"],
                row["relationship_name"],
                _decode_properties(row["properties"]),
            )
            for row in result.mappings().all()
        ]

    @staticmethod
    async def _fetch_edges_within(
        session: AsyncSession, node_ids: list[str]
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        if not node_ids:
            return []
        result = await session.execute(
            text("""
                SELECT source_id, target_id, relationship_name, properties
                FROM graph_edge
                WHERE source_id = ANY(:ids) AND target_id = ANY(:ids)
            """),
            {"ids": node_ids},
        )
        return [
            (
                row["source_id"],
                row["target_id"],
                row["relationship_name"],
                _decode_properties(row["properties"]),
            )
            for row in result.mappings().all()
        ]

    async def get_graph_data(
        self,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Return every node as (id, properties) and every edge as (source, target, name, props)."""
        async with self.sessionmaker() as session:
            node_result = await session.execute(
                text("SELECT id, name, type, properties FROM graph_node")
            )
            nodes = []
            for row in node_result.mappings().all():
                properties = {
                    "name": row["name"],
                    "type": row["type"],
                    **_decode_properties(row["properties"]),
                }
                nodes.append((row["id"], properties))
            if not nodes:
                return [], []

            edge_result = await session.execute(
                text("SELECT source_id, target_id, relationship_name, properties FROM graph_edge")
            )
            edges = [
                (
                    row["source_id"],
                    row["target_id"],
                    row["relationship_name"],
                    _decode_properties(row["properties"]),
                )
                for row in edge_result.mappings().all()
            ]
            return nodes, edges

    async def get_id_filtered_graph_data(
        self, target_ids: List[str]
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Retrieve the subgraph touching target_ids: edges with either endpoint
        in the set, plus all endpoint nodes of those edges (edge-driven,
        matching the Ladybug/Neo4j contract). Lets CogneeGraph project only the
        vector-search neighborhood instead of the full graph.
        """
        if not target_ids:
            return [], []
        ids = [str(i) for i in target_ids]

        async with self.sessionmaker() as session:
            edges = await self._fetch_edges_touching(session, ids)
            endpoint_ids = {
                endpoint
                for source_id, target_id, _, _ in edges
                for endpoint in (source_id, target_id)
            }
            if not endpoint_ids:
                return [], []
            nodes = await self._fetch_nodes_by_id(session, list(endpoint_ids))
            return nodes, edges

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, str, str, Dict]]]:
        """Return core-field matches and the edges induced by those nodes."""
        if not attribute_filters:
            return await self.get_graph_data()

        filters: list[tuple[str, set[str]]] = []
        for filter_dict in attribute_filters:
            for attr, filter_values in filter_dict.items():
                if attr not in self._ALLOWED_FILTER_ATTRS:
                    raise ValueError(f"Invalid filter attribute: {attr!r}")
                filters.append((attr, {str(value) for value in filter_values}))

        if not filters:
            return await self.get_graph_data()

        async with self.sessionmaker() as session:
            result = await session.execute(
                text("SELECT id, name, type, properties FROM graph_node")
            )
            nodes = []
            for row in result.mappings().all():
                if all(str(row[attribute]) in values for attribute, values in filters):
                    properties = {
                        "name": row["name"],
                        "type": row["type"],
                        **_decode_properties(row["properties"]),
                    }
                    nodes.append((row["id"], properties))

            edges = await self._fetch_edges_within(session, [node_id for node_id, _ in nodes])
            return nodes, edges

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str], node_name_filter_operator: str = "OR"
    ) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, str, str, dict]]]:
        """Return matching primary nodes and their qualifying neighbors."""
        if node_name_filter_operator not in {"OR", "AND"}:
            raise ValueError("node_name_filter_operator must be 'OR' or 'AND'")
        if not node_name:
            return [], []

        async with self.sessionmaker() as session:
            result = await session.execute(
                text("SELECT id FROM graph_node WHERE type = :type AND name = ANY(:names)"),
                {"type": node_type.__name__, "names": [str(name) for name in node_name]},
            )
            primary_ids = {row["id"] for row in result.mappings().all()}
            if not primary_ids:
                return [], []

            incident_edges = await self._fetch_edges_touching(session, list(primary_ids))
            neighbor_ids = _select_nodeset_neighbor_ids(
                primary_ids,
                [(source_id, target_id) for source_id, target_id, _, _ in incident_edges],
                node_name_filter_operator,
            )
            subgraph_ids = list(primary_ids | neighbor_ids)
            nodes = await self._fetch_nodes_by_id(session, subgraph_ids)
            edges = await self._fetch_edges_within(session, subgraph_ids)
            return nodes, edges

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        """Compute the supported graph metrics in Python."""
        async with self.sessionmaker() as session:
            node_result = await session.execute(text("SELECT id FROM graph_node"))
            edge_result = await session.execute(text("SELECT source_id, target_id FROM graph_edge"))
            node_ids = [row["id"] for row in node_result.mappings().all()]
            edge_pairs = [
                (row["source_id"], row["target_id"]) for row in edge_result.mappings().all()
            ]
        num_nodes = len(node_ids)
        num_edges = len(edge_pairs)
        component_sizes = _component_sizes(node_ids, edge_pairs)

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "mean_degree": (2 * num_edges) / num_nodes if num_nodes else None,
            "edge_density": num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
            "num_connected_components": len(component_sizes),
            "sizes_of_connected_components": component_sizes,
            "num_selfloops": sum(source == target for source, target in edge_pairs)
            if include_optional
            else -1,
            "diameter": -1,
            "avg_shortest_path_length": -1,
            "avg_clustering": -1,
        }

    async def get_neighborhood(
        self,
        node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Walk incident edges breadth-first and return the induced subgraph."""
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if not node_ids:
            return [], []

        reached = {str(node_id) for node_id in node_ids}
        frontier = set(reached)
        unfiltered_hop = text("""
            SELECT source_id, target_id FROM graph_edge
            WHERE source_id = ANY(:ids) OR target_id = ANY(:ids)
        """)
        filtered_hop = text("""
            SELECT source_id, target_id FROM graph_edge
            WHERE (source_id = ANY(:ids) OR target_id = ANY(:ids))
              AND relationship_name = ANY(:edge_types)
        """)

        async with self.sessionmaker() as session:
            for _ in range(depth):
                if not frontier:
                    break
                params = {"ids": list(frontier)}
                statement = unfiltered_hop
                if edge_types:
                    statement = filtered_hop
                    params["edge_types"] = [str(edge_type) for edge_type in edge_types]
                result = await session.execute(statement, params)

                next_frontier = set()
                for row in result.mappings().all():
                    for endpoint in (row["source_id"], row["target_id"]):
                        if endpoint not in reached:
                            reached.add(endpoint)
                            next_frontier.add(endpoint)
                frontier = next_frontier

            subgraph_ids = list(reached)
            nodes = await self._fetch_nodes_by_id(session, subgraph_ids)
            edges = await self._fetch_edges_within(session, subgraph_ids)
            return nodes, edges

    async def delete_graph(self) -> None:
        """Delete all nodes and edges from the graph."""
        await self.initialize()
        async with self._write_session() as session:
            await session.execute(text("TRUNCATE graph_edge, graph_node CASCADE"))
            await session.commit()

    # Provenance is stored in four text-array columns on both graph tables.

    async def _update_node_provenance(
        self,
        session: AsyncSession,
        node_ids: list[str],
        transition: Callable[[list[str], list[str]], ProvenanceColumns],
    ) -> None:
        """Lock each node and apply a provenance state transition."""
        lock = text("""
            SELECT source_ref_keys, source_run_refs
            FROM graph_node
            WHERE id = :id
            FOR UPDATE
        """)
        update = text("""
            UPDATE graph_node SET
                source_ref_keys = CAST(:source_ref_keys AS text[]),
                source_dataset_ids = CAST(:source_dataset_ids AS text[]),
                source_run_ids = CAST(:source_run_ids AS text[]),
                source_run_refs = CAST(:source_run_refs AS text[]),
                updated_at = now()
            WHERE id = :id
        """)
        for node_id in sorted({str(node_id) for node_id in node_ids}):
            result = await session.execute(lock, {"id": node_id})
            current = result.first()
            if current is None:
                continue
            updated = transition(list(current[0] or []), list(current[1] or []))
            await session.execute(
                update,
                {
                    "id": node_id,
                    "source_ref_keys": updated.source_ref_keys,
                    "source_dataset_ids": updated.source_dataset_ids,
                    "source_run_ids": updated.source_run_ids,
                    "source_run_refs": updated.source_run_refs,
                },
            )

    async def _update_edge_provenance(
        self,
        session: AsyncSession,
        edges: list[EdgeIdentity],
        transition: Callable[[list[str], list[str]], ProvenanceColumns],
    ) -> None:
        """Lock each edge and apply a provenance state transition."""
        lock = text("""
            SELECT source_ref_keys, source_run_refs
            FROM graph_edge
            WHERE source_id = :source_id
              AND target_id = :target_id
              AND relationship_name = :relationship_name
            FOR UPDATE
        """)
        update = text("""
            UPDATE graph_edge SET
                source_ref_keys = CAST(:source_ref_keys AS text[]),
                source_dataset_ids = CAST(:source_dataset_ids AS text[]),
                source_run_ids = CAST(:source_run_ids AS text[]),
                source_run_refs = CAST(:source_run_refs AS text[]),
                updated_at = now()
            WHERE source_id = :source_id
              AND target_id = :target_id
              AND relationship_name = :relationship_name
        """)
        for source_id, target_id, relationship_name in _edge_identities(edges):
            identity = {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_name": relationship_name,
            }
            result = await session.execute(lock, identity)
            current = result.first()
            if current is None:
                continue
            updated = transition(list(current[0] or []), list(current[1] or []))
            await session.execute(
                update,
                {
                    **identity,
                    "source_ref_keys": updated.source_ref_keys,
                    "source_dataset_ids": updated.source_dataset_ids,
                    "source_run_ids": updated.source_run_ids,
                    "source_run_refs": updated.source_run_refs,
                },
            )

    async def attach_node_source_refs(
        self,
        node_ids: list[str],
        source_ref_keys: list[str],
        pipeline_run_id: str | None = None,
    ) -> None:
        if not source_ref_keys:
            return
        keys_to_add = list(source_ref_keys)
        async with self._write_session() as session:
            await self._update_node_provenance(
                session,
                node_ids,
                lambda keys, run_refs: provenance_after_attach(
                    keys, run_refs, keys_to_add, pipeline_run_id
                ),
            )
            await session.commit()

    async def attach_edge_source_refs(
        self,
        edges: list[EdgeIdentity],
        source_ref_keys: list[str],
        pipeline_run_id: str | None = None,
    ) -> None:
        if not source_ref_keys:
            return
        keys_to_add = list(source_ref_keys)
        async with self._write_session() as session:
            await self._update_edge_provenance(
                session,
                edges,
                lambda keys, run_refs: provenance_after_attach(
                    keys, run_refs, keys_to_add, pipeline_run_id
                ),
            )
            await session.commit()

    async def remove_node_source_refs(
        self,
        node_ids: list[str],
        source_ref_keys: list[str],
    ) -> None:
        if not source_ref_keys:
            return
        keys_to_remove = list(source_ref_keys)
        async with self._write_session() as session:
            await self._update_node_provenance(
                session,
                node_ids,
                lambda keys, run_refs: provenance_after_remove(keys, run_refs, keys_to_remove),
            )
            await session.commit()

    async def remove_edge_source_refs(
        self,
        edges: list[EdgeIdentity],
        source_ref_keys: list[str],
    ) -> None:
        if not source_ref_keys:
            return
        keys_to_remove = list(source_ref_keys)
        async with self._write_session() as session:
            await self._update_edge_provenance(
                session,
                edges,
                lambda keys, run_refs: provenance_after_remove(keys, run_refs, keys_to_remove),
            )
            await session.commit()

    async def delete_edge_triples(self, edges: list[EdgeIdentity]) -> None:
        """Delete edges by (source, target, relationship); keep the endpoint nodes."""
        if not edges:
            return
        statement = text("""
            DELETE FROM graph_edge
            WHERE source_id = :source_id
              AND target_id = :target_id
              AND relationship_name = :relationship_name
        """)
        async with self._write_session() as session:
            for source_id, target_id, relationship_name in _edge_identities(edges):
                await session.execute(
                    statement,
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relationship_name": relationship_name,
                    },
                )
            await session.commit()

    async def get_node_delete_data(self, node_ids: list[str]) -> dict[str, NodeDeleteData]:
        if not node_ids:
            return {}
        async with self.sessionmaker() as session:
            result = await session.execute(
                text("""
                    SELECT id, name, type, properties,
                           source_ref_keys, source_dataset_ids, source_run_ids, source_run_refs
                    FROM graph_node WHERE id = ANY(:ids)
                """),
                {"ids": [str(node_id) for node_id in node_ids]},
            )
            out: dict[str, NodeDeleteData] = {}
            for row in result.mappings().all():
                properties = _decode_properties(row["properties"])
                properties.update(id=row["id"], name=row["name"], type=row["type"])
                metadata = properties.get("metadata") or {}
                indexed_fields = (
                    list(metadata.get("index_fields") or []) if isinstance(metadata, dict) else []
                )
                out[row["id"]] = NodeDeleteData(
                    node_id=row["id"],
                    node_type=row["type"] or "",
                    indexed_fields=indexed_fields,
                    node_properties=properties,
                    source_ref_keys=list(row["source_ref_keys"] or []),
                    source_dataset_ids=list(row["source_dataset_ids"] or []),
                    source_run_ids=list(row["source_run_ids"] or []),
                    source_run_refs=list(row["source_run_refs"] or []),
                )
            return out

    async def get_edge_delete_data(
        self, edges: list[EdgeIdentity]
    ) -> dict[EdgeIdentity, EdgeDeleteData]:
        if not edges:
            return {}
        statement = text("""
            SELECT source_id, target_id, relationship_name, properties,
                   source_ref_keys, source_dataset_ids, source_run_ids, source_run_refs
            FROM graph_edge
            WHERE source_id = :source_id
              AND target_id = :target_id
              AND relationship_name = :relationship_name
        """)
        # Lazy import: prepare_edges_for_storage lives in the modules layer, whose
        # package __init__ imports get_graph_engine -> this adapter. Importing it at
        # module load would create a cycle; at delete-time it is safe.
        from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text

        out: dict[EdgeIdentity, EdgeDeleteData] = {}
        async with self.sessionmaker() as session:
            for source_id, target_id, relationship_name in _edge_identities(edges):
                result = await session.execute(
                    statement,
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "relationship_name": relationship_name,
                    },
                )
                row = result.mappings().first()
                if row is None:
                    continue

                edge = EdgeIdentity(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relationship_name=row["relationship_name"],
                )
                properties = _decode_properties(row["properties"])
                out[edge] = EdgeDeleteData(
                    edge=edge,
                    edge_text=get_edge_retrieval_text(
                        properties.get("edge_text"), edge.relationship_name
                    ),
                    edge_properties=properties,
                    source_ref_keys=list(row["source_ref_keys"] or []),
                    source_dataset_ids=list(row["source_dataset_ids"] or []),
                    source_run_ids=list(row["source_run_ids"] or []),
                    source_run_refs=list(row["source_run_refs"] or []),
                )
        return out

    async def find_nodes_by_source_ref(self, source_ref_key: str) -> list[str]:
        async with self.sessionmaker() as session:
            result = await session.execute(
                text("SELECT id FROM graph_node WHERE :source_ref = ANY(source_ref_keys)"),
                {"source_ref": source_ref_key},
            )
            return [row["id"] for row in result.mappings().all()]

    async def find_edges_by_source_ref(self, source_ref_key: str) -> list[EdgeIdentity]:
        async with self.sessionmaker() as session:
            result = await session.execute(
                text("""
                    SELECT source_id, target_id, relationship_name
                    FROM graph_edge
                    WHERE :source_ref = ANY(source_ref_keys)
                """),
                {"source_ref": source_ref_key},
            )
            return [
                EdgeIdentity(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relationship_name=row["relationship_name"],
                )
                for row in result.mappings().all()
            ]

    async def find_node_source_refs_by_dataset(self, dataset_id: str) -> dict[str, list[str]]:
        dataset_id = str(dataset_id)
        async with self.sessionmaker() as session:
            result = await session.execute(
                text("""
                    SELECT id, source_ref_keys
                    FROM graph_node
                    WHERE :dataset_id = ANY(source_dataset_ids)
                """),
                {"dataset_id": dataset_id},
            )
            out: dict[str, list[str]] = {}
            for row in result.mappings().all():
                owned = [
                    key
                    for key in (row["source_ref_keys"] or [])
                    if str(get_dataset_id_from_source_ref_key(key)) == dataset_id
                ]
                if owned:
                    out[row["id"]] = owned
            return out

    async def find_edge_source_refs_by_dataset(
        self, dataset_id: str
    ) -> dict[EdgeIdentity, list[str]]:
        dataset_id = str(dataset_id)
        async with self.sessionmaker() as session:
            result = await session.execute(
                text("""
                    SELECT source_id, target_id, relationship_name, source_ref_keys
                    FROM graph_edge
                    WHERE :dataset_id = ANY(source_dataset_ids)
                """),
                {"dataset_id": dataset_id},
            )
            out: dict[EdgeIdentity, list[str]] = {}
            for row in result.mappings().all():
                owned = [
                    key
                    for key in (row["source_ref_keys"] or [])
                    if str(get_dataset_id_from_source_ref_key(key)) == dataset_id
                ]
                if owned:
                    edge = EdgeIdentity(
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        relationship_name=row["relationship_name"],
                    )
                    out[edge] = owned
            return out

    async def find_node_source_refs_by_pipeline_run(
        self, pipeline_run_id: str
    ) -> dict[str, list[str]]:
        pipeline_run_id = str(pipeline_run_id)
        async with self.sessionmaker() as session:
            result = await session.execute(
                text("""
                    SELECT id, source_run_refs
                    FROM graph_node
                    WHERE :pipeline_run_id = ANY(source_run_ids)
                """),
                {"pipeline_run_id": pipeline_run_id},
            )
            out: dict[str, list[str]] = {}
            for row in result.mappings().all():
                contributed = [
                    get_source_ref_key_from_source_run_ref(ref)
                    for ref in (row["source_run_refs"] or [])
                    if str(get_pipeline_run_id_from_source_run_ref(ref)) == pipeline_run_id
                ]
                if contributed:
                    out[row["id"]] = contributed
            return out

    async def find_edge_source_refs_by_pipeline_run(
        self, pipeline_run_id: str
    ) -> dict[EdgeIdentity, list[str]]:
        pipeline_run_id = str(pipeline_run_id)
        async with self.sessionmaker() as session:
            result = await session.execute(
                text("""
                    SELECT source_id, target_id, relationship_name, source_run_refs
                    FROM graph_edge
                    WHERE :pipeline_run_id = ANY(source_run_ids)
                """),
                {"pipeline_run_id": pipeline_run_id},
            )
            out: dict[EdgeIdentity, list[str]] = {}
            for row in result.mappings().all():
                contributed = [
                    get_source_ref_key_from_source_run_ref(ref)
                    for ref in (row["source_run_refs"] or [])
                    if str(get_pipeline_run_id_from_source_run_ref(ref)) == pipeline_run_id
                ]
                if contributed:
                    edge = EdgeIdentity(
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        relationship_name=row["relationship_name"],
                    )
                    out[edge] = contributed
            return out

    async def set_graph_metadata(self, metadata: dict[str, str]) -> None:
        """Upsert graph-level metadata keys."""
        if not metadata:
            return
        await self.initialize()
        rows = [{"key": str(key), "value": str(value)} for key, value in sorted(metadata.items())]
        upsert = text("""
            INSERT INTO graph_metadata (key, value)
            VALUES (:key, :value)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """)
        async with self._write_session() as session:
            await session.execute(upsert, rows)
            await session.commit()

    async def get_graph_metadata(self) -> dict[str, str]:
        """Return every graph-level metadata key and value."""
        await self.initialize()
        async with self.sessionmaker() as session:
            result = await session.execute(text("SELECT key, value FROM graph_metadata"))
            return {row["key"]: row["value"] for row in result.mappings().all()}

    async def remove_belongs_to_set_tags(
        self,
        tags: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> None:
        """Strip ``tags`` from each node's ``belongs_to_set`` property array.

        Keeps the denormalized membership list consistent with the additive
        belongs_to_set edges after a NodeSet (or its dataset) is deleted. The tags
        live inside the JSONB ``properties`` blob rather than in a core column, so
        this is a read-modify-write: the rows stay locked in node-id order for the
        whole transaction so a concurrent cleanup cannot overwrite this one.
        """
        if not tags:
            return
        if node_ids is not None and not node_ids:
            return

        tags_to_remove = set(tags)
        select_scoped = text("""
            SELECT id, properties FROM graph_node
            WHERE id = ANY(:ids)
            ORDER BY id
            FOR UPDATE
        """)
        select_all = text("SELECT id, properties FROM graph_node ORDER BY id FOR UPDATE")
        update_properties = text("""
            UPDATE graph_node
            SET properties = CAST(:properties AS jsonb), updated_at = now()
            WHERE id = :id
        """)

        async with self._write_session() as session:
            if node_ids is None:
                result = await session.execute(select_all)
            else:
                result = await session.execute(
                    select_scoped, {"ids": [str(node_id) for node_id in node_ids]}
                )

            for row in result.mappings().all():
                properties = _decode_properties(row["properties"])
                current_tags = properties.get("belongs_to_set")
                if not isinstance(current_tags, list) or tags_to_remove.isdisjoint(current_tags):
                    continue
                properties["belongs_to_set"] = [
                    tag for tag in current_tags if tag not in tags_to_remove
                ]
                await session.execute(
                    update_properties,
                    {
                        "id": row["id"],
                        "properties": json.dumps(properties, cls=JSONEncoder),
                    },
                )
            await session.commit()

    async def get_triplets_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        """Return one page of source-edge-target triplets.

        Ordering by the full edge identity keeps pagination stable, so exporting
        the graph page by page visits every triplet exactly once.
        """
        if offset < 0:
            raise ValueError(f"Offset must be non-negative, got {offset}")
        if limit < 0:
            raise ValueError(f"Limit must be non-negative, got {limit}")

        async with self.sessionmaker() as session:
            result = await session.execute(
                text("""
                    SELECT
                        source.id AS source_id,
                        source.name AS source_name,
                        source.type AS source_type,
                        source.properties AS source_properties,
                        edge.relationship_name,
                        edge.properties AS edge_properties,
                        target.id AS target_id,
                        target.name AS target_name,
                        target.type AS target_type,
                        target.properties AS target_properties
                    FROM graph_edge AS edge
                    JOIN graph_node AS source ON source.id = edge.source_id
                    JOIN graph_node AS target ON target.id = edge.target_id
                    ORDER BY edge.source_id, edge.target_id, edge.relationship_name
                    OFFSET :offset LIMIT :limit
                """),
                {"offset": offset, "limit": limit},
            )

            return [
                {
                    "start_node": {
                        "id": row["source_id"],
                        "name": row["source_name"],
                        "type": row["source_type"],
                        **_decode_properties(row["source_properties"]),
                    },
                    "relationship_properties": {
                        "relationship_name": row["relationship_name"],
                        **_decode_properties(row["edge_properties"]),
                    },
                    "end_node": {
                        "id": row["target_id"],
                        "name": row["target_name"],
                        "type": row["target_type"],
                        **_decode_properties(row["target_properties"]),
                    },
                }
                for row in result.mappings().all()
            ]
