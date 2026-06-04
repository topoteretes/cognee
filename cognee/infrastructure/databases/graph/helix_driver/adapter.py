"""HelixDB graph adapter — implements GraphDBInterface over the v2 dynamic JSON route.

HelixDB is a unified graph + vector engine. This module implements the graph
half; :class:`HelixHybridAdapter` (``hybrid/helix``) subclasses it and layers the
vector half on top, mirroring how ``NeptuneAnalyticsAdapter`` subclasses
``NeptuneGraphDB``.

Modeling notes
--------------
- All Cognee nodes share a single Helix label (``COGNEE_NODE``); the DataPoint
  class name is kept in a ``type`` property. Helix's virtual ``$id`` is an
  auto-assigned u64 and cannot be set, so the Cognee UUID lives in an indexed
  ``id`` property and every lookup/edge resolves through it.
- Edges store ``source_id`` / ``target_id`` / ``relationship_name`` as properties
  (in addition to the structural edge + label) so traversal results map back to
  Cognee UUIDs without depending on Helix internal ids.
- Every node and edge carries a ``tenant_id`` property; every read is scoped by
  it. ``tenant_id`` defaults to ``"default"`` when access control is off.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from uuid import UUID

from cognee.shared.logging_utils import get_logger
from cognee.modules.storage.utils import JSONEncoder
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    NodeData,
    EdgeData,
    Node,
)

from .client import HelixClient
from . import ast

logger = get_logger("HelixGraphDB")

NODE_LABEL = "COGNEE_NODE"
ID_PROP = "id"
TENANT_PROP = "tenant_id"
DEFAULT_TENANT = "default"
# Vector properties are stored on the shared node under this prefix so they can be
# kept out of graph NodeData (value_map returns every property otherwise).
VECTOR_PROP_PREFIX = "vec__"


class HelixGraphDB(GraphDBInterface):
    """Graph operations against a HelixDB instance via dynamic JSON queries."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        self.client = HelixClient(base_url=base_url, api_key=api_key)
        self.tenant_id = tenant_id or DEFAULT_TENANT
        self._initialized = False
        self._init_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Create the equality indexes the adapter relies on (idempotent)."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self.client.query(
                request_type="write",
                queries=[
                    ast.query_entry(
                        "idx_id",
                        [ast.create_node_equality_index(NODE_LABEL, ID_PROP, unique=True)],
                    ),
                    ast.query_entry(
                        "idx_tenant",
                        [ast.create_node_equality_index(NODE_LABEL, TENANT_PROP)],
                    ),
                    ast.query_entry(
                        "idx_type",
                        [ast.create_node_equality_index(NODE_LABEL, "type")],
                    ),
                ],
                returns=[],
            )
            self._initialized = True

    # ------------------------------------------------------------------ #
    # Low-level helpers
    # ------------------------------------------------------------------ #

    async def _read(self, name: str, steps: List[Any]) -> List[Any]:
        await self.initialize()
        resp = await self.client.query(
            request_type="read",
            queries=[ast.query_entry(name, steps)],
            returns=[name],
        )
        return _as_rows(resp.get(name))

    async def _read_scalar(self, name: str, steps: List[Any], default: Any = 0) -> Any:
        await self.initialize()
        resp = await self.client.query(
            request_type="read",
            queries=[ast.query_entry(name, steps)],
            returns=[name],
        )
        return _count_value(resp.get(name), default)

    async def _write(self, queries: List[Dict[str, Any]], returns: List[str]) -> Dict[str, Any]:
        await self.initialize()
        return await self.client.query(request_type="write", queries=queries, returns=returns)

    def _tenant_scope(self, *predicates: Dict[str, Any]) -> Dict[str, Any]:
        """Build an AND predicate that always includes the tenant scope."""
        preds = [ast.eq(TENANT_PROP, self.tenant_id), *predicates]
        return preds[0] if len(preds) == 1 else ast.and_(preds)

    @staticmethod
    def _serialize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten properties for Helix storage (mirrors the Neptune adapter)."""
        serialized: Dict[str, Any] = {}
        for key, value in properties.items():
            if value is None:
                continue
            if isinstance(value, UUID):
                serialized[key] = str(value)
            elif isinstance(value, (dict, list)):
                serialized[key] = json.dumps(value, cls=JSONEncoder)
            else:
                serialized[key] = value
        return serialized

    def _node_properties(self, node: Union[DataPoint, str], properties: Optional[Dict]) -> Dict:
        if isinstance(node, DataPoint):
            props = self._serialize_properties(node.model_dump())
            props[ID_PROP] = str(node.id)
        else:
            props = self._serialize_properties(properties or {})
            props[ID_PROP] = str(node)
        props[TENANT_PROP] = self.tenant_id
        return props

    def _edge_properties(
        self, source_id: str, target_id: str, relationship_name: str, properties: Optional[Dict]
    ) -> Dict[str, Any]:
        props = self._serialize_properties(properties or {})
        props["source_id"] = str(source_id)
        props["target_id"] = str(target_id)
        props["relationship_name"] = relationship_name
        props[TENANT_PROP] = self.tenant_id
        return props

    # ------------------------------------------------------------------ #
    # Raw query — unsupported (Postgres parity; Helix v2 has no Cypher)
    # ------------------------------------------------------------------ #

    async def query(self, query: str, params: Optional[dict] = None) -> List[Any]:
        raise NotImplementedError(
            "The HelixDB backend does not support raw Cypher queries. "
            "Use the typed adapter methods (add_nodes, get_neighbors, ...), "
            "or a Cypher-native backend (Neo4j, Ladybug) for raw query support."
        )

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    async def is_empty(self) -> bool:
        count = await self._read_scalar(
            "is_empty", [ast.n_where(self._tenant_scope()), ast.COUNT], default=0
        )
        return int(count or 0) == 0

    async def add_node(
        self, node: Union[DataPoint, str], properties: Optional[Dict[str, Any]] = None
    ) -> None:
        if isinstance(node, str):
            props = self._node_properties(node, properties)
            await self._write(
                self._upsert_node_queries(0, props[ID_PROP], _encode_scalar_props(props)), []
            )
        else:
            await self.add_nodes([node])

    async def add_nodes(self, nodes: Union[List[Node], List[DataPoint]]) -> None:
        if not nodes:
            return
        queries: List[Dict[str, Any]] = []
        for idx, node in enumerate(nodes):
            props = self._node_properties(node, None)
            queries.extend(
                self._upsert_node_queries(idx, props[ID_PROP], _encode_scalar_props(props))
            )
        await self._write(queries, [])

    def _upsert_node_queries(
        self, idx: int, node_id: str, encoded_props: List[List[Any]]
    ) -> List[Dict[str, Any]]:
        """Three batch entries that merge ``encoded_props`` onto the node with
        ``node_id`` (creating it if absent), preserving any other properties
        already on the node (e.g. vectors written by a separate vector call).
        """
        existing_var = f"ex{idx}"
        return [
            ast.query_entry(
                existing_var,
                [ast.n_where(self._tenant_scope(ast.eq(ID_PROP, node_id)))],
            ),
            ast.query_entry(
                None,
                [ast.n_by_var(existing_var)]
                + [ast.set_property(name, value) for name, value in encoded_props],
                condition={"VarNotEmpty": existing_var},
            ),
            ast.query_entry(
                None,
                [ast.add_node_with_inputs(NODE_LABEL, encoded_props)],
                condition={"VarEmpty": existing_var},
            ),
        ]

    async def delete_node(self, node_id: str) -> None:
        await self.delete_nodes([node_id])

    async def delete_nodes(self, node_ids: List[str]) -> None:
        if not node_ids:
            return
        ids = [str(n) for n in node_ids]
        await self._write(
            [
                ast.query_entry(
                    None,
                    [
                        ast.n_where(self._tenant_scope()),
                        ast.where(ast.is_in(ID_PROP, ids)),
                        ast.DROP,
                    ],
                )
            ],
            [],
        )

    async def get_node(self, node_id: str) -> Optional[NodeData]:
        rows = await self._read(
            "node",
            [
                ast.n_where(self._tenant_scope(ast.eq(ID_PROP, str(node_id)))),
                ast.value_map(None),
            ],
        )
        return _node_data(rows[0]) if rows else None

    async def get_nodes(self, node_ids: List[str]) -> List[NodeData]:
        if not node_ids:
            return []
        ids = [str(n) for n in node_ids]
        rows = await self._read(
            "nodes",
            [
                ast.n_where(self._tenant_scope()),
                ast.where(ast.is_in(ID_PROP, ids)),
                ast.value_map(None),
            ],
        )
        return [_node_data(row) for row in rows]

    async def get_neighbors(self, node_id: str) -> List[NodeData]:
        rows = await self._read(
            "neighbors",
            [
                ast.n_where(self._tenant_scope(ast.eq(ID_PROP, str(node_id)))),
                ast.both(None),
                ast.DEDUP,
                ast.value_map(None),
            ],
        )
        return [_node_data(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Edges
    # ------------------------------------------------------------------ #

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self.add_edges([(source_id, target_id, relationship_name, properties or {})])

    async def add_edges(
        self, edges: Union[List[EdgeData], List[Tuple[str, str, str, Optional[Dict[str, Any]]]]]
    ) -> None:
        if not edges:
            return
        queries: List[Dict[str, Any]] = []
        for idx, edge in enumerate(edges):
            source_id, target_id, relationship_name = str(edge[0]), str(edge[1]), edge[2]
            props = edge[3] if len(edge) > 3 and edge[3] else {}
            edge_props = self._edge_properties(source_id, target_id, relationship_name, props)
            tgt_var = f"e{idx}_tgt"
            # Bind the target node, then create the edge from the source to it.
            queries.append(
                ast.query_entry(
                    tgt_var,
                    [ast.n_where(self._tenant_scope(ast.eq(ID_PROP, target_id)))],
                )
            )
            queries.append(
                ast.query_entry(
                    None,
                    [
                        ast.n_where(self._tenant_scope(ast.eq(ID_PROP, source_id))),
                        ast.add_edge(relationship_name, tgt_var, edge_props),
                    ],
                    condition={"VarNotEmpty": tgt_var},
                )
            )
        await self._write(queries, [])

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        count = await self._read_scalar(
            "has_edge",
            [
                ast.n_where(self._tenant_scope(ast.eq(ID_PROP, str(source_id)))),
                ast.out(relationship_name),
                ast.where(ast.eq(ID_PROP, str(target_id))),
                ast.COUNT,
            ],
            default=0,
        )
        return int(count or 0) > 0

    async def has_edges(self, edges: List[EdgeData]) -> List[EdgeData]:
        if not edges:
            return []
        await self.initialize()
        queries = []
        for idx, edge in enumerate(edges):
            queries.append(
                ast.query_entry(
                    f"e{idx}",
                    [
                        ast.n_where(self._tenant_scope(ast.eq(ID_PROP, str(edge[0])))),
                        ast.out(edge[2]),
                        ast.where(ast.eq(ID_PROP, str(edge[1]))),
                        ast.COUNT,
                    ],
                )
            )
        resp = await self.client.query(
            request_type="read",
            queries=queries,
            returns=[f"e{idx}" for idx in range(len(edges))],
        )
        existing = []
        for idx, edge in enumerate(edges):
            if int(_count_value(resp.get(f"e{idx}")) or 0) > 0:
                existing.append(edge)
        return existing

    async def get_edges(self, node_id: str) -> List[EdgeData]:
        # Edge props carry the UUID endpoints, so incident edges are derived from
        # a tenant edge scan filtered to those touching node_id.
        return await self._incident_edges(str(node_id))

    async def _all_edges(self) -> List[EdgeData]:
        rows = await self._read(
            "all_edges",
            [{"EWhere": self._tenant_scope()}, "EdgeProperties"],
        )
        return [_edge_data(row) for row in rows if isinstance(row, dict)]

    async def _incident_edges(self, node_id: str) -> List[EdgeData]:
        edges = await self._all_edges()
        return [e for e in edges if e[0] == node_id or e[1] == node_id]

    async def get_connections(
        self, node_id: Union[str, UUID]
    ) -> List[Tuple[NodeData, Dict[str, Any], NodeData]]:
        node_id = str(node_id)
        edges = await self._incident_edges(node_id)
        if not edges:
            return []
        endpoint_ids = {e[0] for e in edges} | {e[1] for e in edges}
        nodes_by_id = {n.get(ID_PROP): n for n in await self.get_nodes(list(endpoint_ids))}
        connections = []
        for source_id, target_id, relationship_name, props in edges:
            source = nodes_by_id.get(source_id, {ID_PROP: source_id})
            target = nodes_by_id.get(target_id, {ID_PROP: target_id})
            edge = {"relationship_name": relationship_name, **props}
            connections.append((source, edge, target))
        return connections

    # ------------------------------------------------------------------ #
    # Graph-wide reads
    # ------------------------------------------------------------------ #

    async def get_graph_data(self) -> Tuple[List[Node], List[EdgeData]]:
        node_rows = await self._read(
            "nodes", [ast.n_where(self._tenant_scope()), ast.value_map(None)]
        )
        nodes = [(row.get(ID_PROP), _node_data(row)) for row in node_rows if isinstance(row, dict)]
        edges = await self._all_edges()
        return (nodes, edges)

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ) -> Tuple[List[Node], List[EdgeData]]:
        if not attribute_filters:
            return await self.get_graph_data()
        predicates = [ast.is_in(attr, values) for attr, values in attribute_filters[0].items()]
        node_rows = await self._read(
            "filtered",
            [
                ast.n_where(self._tenant_scope()),
                ast.where(ast.and_(predicates) if len(predicates) > 1 else predicates[0]),
                ast.value_map(None),
            ],
        )
        nodes = [(row.get(ID_PROP), _node_data(row)) for row in node_rows if isinstance(row, dict)]
        node_ids = {n[0] for n in nodes}
        edges = [e for e in await self._all_edges() if e[0] in node_ids and e[1] in node_ids]
        return (nodes, edges)

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str], node_name_filter_operator: str = "OR"
    ) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]]:
        primary_rows = await self._read(
            "primary",
            [
                ast.n_where(self._tenant_scope(ast.eq("type", node_type.__name__))),
                ast.where(ast.is_in("name", node_name)),
                ast.value_map(None),
            ],
        )
        neighbor_rows = await self._read(
            "neighbors",
            [
                ast.n_where(self._tenant_scope(ast.eq("type", node_type.__name__))),
                ast.where(ast.is_in("name", node_name)),
                ast.both(None),
                ast.DEDUP,
                ast.value_map(None),
            ],
        )
        nodes_by_id: Dict[Any, dict] = {}
        for row in [*primary_rows, *neighbor_rows]:
            if isinstance(row, dict):
                data = _node_data(row)
                nodes_by_id[data.get(ID_PROP)] = data
        node_ids = set(nodes_by_id.keys())
        edges = [e for e in await self._all_edges() if e[0] in node_ids and e[1] in node_ids]
        nodes = [(nid, data) for nid, data in nodes_by_id.items()]
        return (nodes, edges)

    async def get_neighborhood(
        self,
        node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[List[Node], List[EdgeData]]:
        if not node_ids:
            return [], []
        all_ids = {str(n) for n in node_ids}
        frontier = set(all_ids)
        for _ in range(max(depth, 0)):
            if not frontier:
                break
            steps = [
                ast.n_where(self._tenant_scope()),
                ast.where(ast.is_in(ID_PROP, list(frontier))),
            ]
            if edge_types:
                steps.append({"Union": [{"steps": [ast.both(t)]} for t in edge_types]})
            else:
                steps.append(ast.both(None))
            steps += [ast.DEDUP, ast.value_map(None)]
            rows = await self._read("hop", steps)
            new_ids = {
                row.get(ID_PROP) for row in rows if isinstance(row, dict) and row.get(ID_PROP)
            }
            frontier = new_ids - all_ids
            all_ids |= new_ids
        nodes = [(n.get(ID_PROP), n) for n in await self.get_nodes(list(all_ids))]
        edges = [e for e in await self._all_edges() if e[0] in all_ids and e[1] in all_ids]
        if edge_types:
            edges = [e for e in edges if e[2] in set(edge_types)]
        return (nodes, edges)

    async def delete_graph(self) -> None:
        await self._write(
            [ast.query_entry(None, [ast.n_where(self._tenant_scope()), ast.DROP])], []
        )

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        num_nodes = int(
            await self._read_scalar("n", [ast.n_where(self._tenant_scope()), ast.COUNT], default=0)
            or 0
        )
        edges = await self._all_edges()
        num_edges = len(edges)
        mandatory = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "mean_degree": (2 * num_edges) / num_nodes if num_nodes else None,
            "edge_density": num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else None,
            "num_connected_components": -1,
            "sizes_of_connected_components": [],
        }
        optional = {
            "num_selfloops": -1,
            "diameter": -1,
            "avg_shortest_path_length": -1,
            "avg_clustering": -1,
        }
        if include_optional:
            optional["num_selfloops"] = sum(1 for e in edges if e[0] == e[1])
        return mandatory | optional

    async def get_triplets_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        edges = await self._all_edges()
        page = edges[offset : offset + limit]
        if not page:
            return []
        endpoint_ids = {e[0] for e in page} | {e[1] for e in page}
        nodes_by_id = {n.get(ID_PROP): n for n in await self.get_nodes(list(endpoint_ids))}
        return [
            {
                "source": nodes_by_id.get(e[0], {ID_PROP: e[0]}),
                "edge": {"relationship_name": e[2], **e[3]},
                "target": nodes_by_id.get(e[1], {ID_PROP: e[1]}),
            }
            for e in page
        ]


# --------------------------------------------------------------------------- #
# Response parsing helpers
# --------------------------------------------------------------------------- #


def _encode_scalar_props(props: Dict[str, Any]) -> List[List[Any]]:
    """Encode a flat scalar property dict as ``[name, PropertyInput]`` pairs."""
    return [[name, ast.input_value(value)] for name, value in props.items()]


def _as_rows(value: Any) -> List[Any]:
    """Extract the row list from a Helix response value.

    Row-producing queries return ``{"properties": [<row>, ...]}``; a bare list or
    scalar is tolerated for forward-compatibility and mocked tests.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        rows = value.get("properties")
        return rows if isinstance(rows, list) else [value]
    if isinstance(value, list):
        return value
    return [value]


def _count_value(value: Any, default: Any = 0) -> Any:
    """Read a scalar count from a Helix ``Count`` response (``{"count": N}``)."""
    if isinstance(value, dict):
        return value.get("count", default)
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _strip_internal(row: Dict[str, Any]) -> Dict[str, Any]:
    """Drop Helix virtual fields ($id/$label/$distance/...) and vector props."""
    return {
        k: v
        for k, v in row.items()
        if not k.startswith("$") and not k.startswith(VECTOR_PROP_PREFIX)
    }


def _node_data(row: Any) -> NodeData:
    """Normalize a value-map row into a flat NodeData dict (no virtuals/vectors)."""
    if not isinstance(row, dict):
        return {}
    return _strip_internal(row)


def _edge_data(row: Dict[str, Any]) -> EdgeData:
    """Build an EdgeData tuple from a stored edge-properties row."""
    props = {k: v for k, v in _strip_internal(row).items() if k != TENANT_PROP}
    return (props.get("source_id"), props.get("target_id"), props.get("relationship_name"), props)
