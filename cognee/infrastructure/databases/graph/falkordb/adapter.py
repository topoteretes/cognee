"""FalkorDB Graph Database Adapter for Cognee.

This adapter provides FalkorDB integration for storing graph nodes and edges,
with support for multi-agent isolation via per-agent graph routing.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from uuid import UUID

from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    record_graph_changes,
)
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger

logger = get_logger("FalkorDBAdapter")

BASE_LABEL = "__Node__"


def _sanitize_graph_name(raw: Optional[str]) -> Optional[str]:
    """Sanitize graph name for FalkorDB compatibility."""
    if not raw:
        return None
    return raw.replace(" ", "_").replace("'", "").replace("-", "_")


class FalkorDBAdapter(GraphDBInterface):
    """
    FalkorDB graph database adapter implementing GraphDBInterface.

    Features:
    - Store graph nodes and edges in FalkorDB
    - Per-agent graph isolation via context variable
    - Async operations using thread pool executor (FalkorDB client is sync)
    """

    def __init__(
        self,
        graph_database_url: str = "",
        graph_database_port: int = 6379,
        graph_database_password: Optional[str] = None,
        graph_database_name: str = "CogneeGraph",
        **kwargs,
    ):
        """Initialize FalkorDB adapter.

        Args:
            graph_database_url: FalkorDB host URL (e.g., 'localhost' or 'redis://host')
            graph_database_port: FalkorDB port (default: 6379)
            graph_database_password: Optional password for authentication
            graph_database_name: Default graph name (default: 'CogneeGraph')
        
        Environment Variables (preferred - unified config):
            FALKORDB_HOST: FalkorDB host (takes precedence over graph_database_url)
            FALKORDB_PORT: FalkorDB port (takes precedence over graph_database_port)
            FALKORDB_PASSWORD: FalkorDB password (takes precedence over graph_database_password)
            FALKORDB_GRAPH_NAME: Graph name (takes precedence over graph_database_name)
        """
        import os

        # Prefer unified FALKORDB_* env vars, fall back to constructor params
        raw_host = os.getenv("FALKORDB_HOST") or graph_database_url
        self.host = (
            raw_host.replace("redis://", "").split(":")[0]
            if raw_host and "redis://" in raw_host
            else (raw_host or "localhost")
        )
        self.port = int(os.getenv("FALKORDB_PORT") or graph_database_port or 6379)
        self.password = os.getenv("FALKORDB_PASSWORD") or graph_database_password
        self._default_graph_name = os.getenv("FALKORDB_GRAPH_NAME") or graph_database_name or "CogneeGraph"

        self.client = None
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._graphs_initialized: set = set()

    def _get_graph_name_from_ctx(self) -> str:
        """Get graph name from context variable or fall back to default."""
        # Import here to avoid circular imports
        from cognee.context_global_variables import agent_graph_name_ctx

        ctx_name = _sanitize_graph_name(agent_graph_name_ctx.get())
        default_name = _sanitize_graph_name(self._default_graph_name) or "CogneeGraph"
        return ctx_name or default_name

    def _connect_sync(self) -> None:
        """Synchronous connection to FalkorDB."""
        if self.client is not None:
            return

        try:
            from falkordb import FalkorDB
        except ImportError:
            raise ImportError(
                "FalkorDB is not installed. Please install with 'pip install cognee[falkordb]'"
            )

        kwargs: Dict[str, Any] = {"host": self.host, "port": self.port}
        if self.password:
            kwargs["password"] = self.password
        self.client = FalkorDB(**kwargs)

    async def initialize(self) -> None:
        """Initialize the FalkorDB connection asynchronously."""
        if self.client is not None:
            return
        await asyncio.get_running_loop().run_in_executor(self._executor, self._connect_sync)

    def _ensure_graph_initialized_sync(self, graph_name: str) -> None:
        """Ensure graph has required indices (sync)."""
        if graph_name in self._graphs_initialized:
            return
        if not self.client:
            self._connect_sync()
        assert self.client is not None

        graph = self.client.select_graph(graph_name)

        # Create indices; ignore "already exists" errors
        for q in (
            f"CREATE INDEX FOR (n:{BASE_LABEL}) ON (n.id)",
            f"CREATE INDEX FOR (n:{BASE_LABEL}) ON (n.updated_at)",
        ):
            try:
                graph.query(q)
            except Exception as e:
                msg = str(e).lower()
                if "already indexed" in msg or "already exists" in msg:
                    continue
        self._graphs_initialized.add(graph_name)

    async def _ensure_graph_initialized(self, graph_name: str) -> None:
        """Ensure graph has required indices (async)."""
        await asyncio.get_running_loop().run_in_executor(
            self._executor, self._ensure_graph_initialized_sync, graph_name
        )

    async def query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query against FalkorDB."""
        graph_name = self._get_graph_name_from_ctx()
        await self._ensure_graph_initialized(graph_name)
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._query_sync, query, params or {}, graph_name
        )

    def _query_sync(
        self, query: str, params: Dict[str, Any], graph_name: str
    ) -> List[Dict[str, Any]]:
        """Synchronous query execution."""
        if not self.client:
            self._connect_sync()
        assert self.client is not None

        graph = self.client.select_graph(graph_name)

        # FalkorDB rejects multi-statement strings, so execute sequentially
        statements = [s.strip() for s in str(query).split(";") if s.strip()]
        if not statements:
            return []

        res = None
        for stmt in statements:
            res = graph.query(stmt, params)

        data: List[Dict[str, Any]] = []
        if res is None or not getattr(res, "result_set", None):
            return data

        header = getattr(res, "header", None)
        for record in res.result_set:
            if header:
                row: Dict[str, Any] = {}
                for i, col_def in enumerate(header):
                    if isinstance(col_def, (list, tuple)) and len(col_def) >= 2:
                        col_name = col_def[1] if isinstance(col_def[0], int) else col_def[0]
                    else:
                        col_name = str(col_def)
                    row[col_name] = record[i]
                data.append(row)
            else:
                data.append({"value": record})
        return data

    async def is_empty(self) -> bool:
        """Check if the graph is empty."""
        try:
            res = await self.query("MATCH (n) RETURN count(n) as count")
            if not res:
                return True
            count_val = res[0].get("count", 0)
            return int(count_val) == 0
        except Exception:
            return True

    def _sanitize_props(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize property values for FalkorDB compatibility."""
        try:
            import numpy as np
        except ImportError:
            np = None  # type: ignore

        def sanitize_value(v):
            if np is not None and isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (UUID, datetime, date)):
                return str(v)
            if isinstance(v, dict):
                return {k: sanitize_value(val) for k, val in v.items()}
            if isinstance(v, list):
                return [sanitize_value(x) for x in v]
            return v

        sanitized: Dict[str, Any] = {}
        for k, v in (props or {}).items():
            val = sanitize_value(v)
            sanitized[k] = json.dumps(val) if isinstance(val, (dict, list)) else val
        return sanitized

    async def add_node(
        self, node: Union[DataPoint, str], properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a single node to the graph."""
        if hasattr(node, "id") and hasattr(node, "model_dump"):
            properties = node.model_dump()
            node_id = str(node.id)
            label = type(node).__name__
        else:
            node_id = str(node)
            label = "Node"
        
        # logger.warning(f"DEBUG: add_node id={node_id} label={label}")

        properties = properties or {}
        if "type" not in properties:
            properties["type"] = label

        props = self._sanitize_props(properties)
        query = (
            f"MERGE (n:`{BASE_LABEL}` {{id: $id}}) "
            f"ON CREATE SET n += $props "
            f"ON MATCH SET n += $props "
            f"SET n:`{label}` "
            f"RETURN n"
        )
        await self.query(query, {"id": node_id, "props": props})

    @record_graph_changes
    async def add_nodes(self, nodes: Union[List[Any], List[DataPoint]]) -> None:
        """Add multiple nodes to the graph."""
        # logger.warning(f"DEBUG: add_nodes called with {len(nodes)} nodes")
        for node in nodes:
            await self.add_node(node)

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an edge between two nodes."""
        props = self._sanitize_props(properties or {})
        query = (
            f"MATCH (a:`{BASE_LABEL}` {{id: $source_id}}), (b:`{BASE_LABEL}` {{id: $target_id}}) "
            f"MERGE (a)-[r:`{relationship_name}`]->(b) "
            f"SET r += $props"
        )
        await self.query(
            query,
            {"source_id": str(source_id), "target_id": str(target_id), "props": props},
        )

    @record_graph_changes
    async def add_edges(self, edges: Union[List[Tuple], List[Any]]) -> None:
        """Add multiple edges to the graph."""
        # logger.warning(f"DEBUG: add_edges called with {len(edges)} edges")
        for edge in edges:
            if isinstance(edge, tuple):
                if len(edge) >= 4:
                    await self.add_edge(edge[0], edge[1], edge[2], edge[3])
                elif len(edge) >= 3:
                    await self.add_edge(edge[0], edge[1], edge[2])
            elif hasattr(edge, "source_node_id"):
                await self.add_edge(
                    edge.source_node_id,
                    edge.target_node_id,
                    edge.relationship_name,
                    getattr(edge, "properties", {}),
                )

    async def delete_graph(self) -> None:
        """Delete all nodes and edges in the graph."""
        await self.query("MATCH (n) DETACH DELETE n")

    async def delete_node(self, node_id: str) -> None:
        """Delete a single node by ID."""
        await self.query(
            f"MATCH (n:`{BASE_LABEL}` {{id: $id}}) DETACH DELETE n", {"id": str(node_id)}
        )

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """Delete multiple nodes by IDs."""
        await self.query(
            f"MATCH (n:`{BASE_LABEL}`) WHERE n.id IN $ids DETACH DELETE n",
            {"ids": [str(x) for x in node_ids]},
        )

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node by ID."""
        res = await self.query(
            f"MATCH (n:`{BASE_LABEL}` {{id: $id}}) RETURN n", {"id": str(node_id)}
        )
        if not res:
            return None
        n = res[0].get("n") or res[0].get(0)
        if not n:
            return None
        return n.properties if hasattr(n, "properties") else (n if isinstance(n, dict) else None)

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple nodes by IDs."""
        if not node_ids:
            return []
        res = await self.query(
            f"MATCH (n:`{BASE_LABEL}`) WHERE n.id IN $ids RETURN n",
            {"ids": [str(x) for x in node_ids]},
        )
        out: List[Dict[str, Any]] = []
        for r in res:
            n = r.get("n") or r.get(0)
            if not n:
                continue
            out.append(
                n.properties if hasattr(n, "properties") else (n if isinstance(n, dict) else {})
            )
        return out

    async def get_graph_data(self) -> Tuple[List[Any], List[Any]]:
        """Get all nodes and edges in the graph."""
        nodes_result = await self.query("MATCH (n) RETURN n")
        nodes_dict: Dict[str, Any] = {}
        for r in nodes_result:
            node = r.get("n") or r.get(0)
            if not node:
                continue
            props = (
                node.properties
                if hasattr(node, "properties")
                else (node if isinstance(node, dict) else {})
            )
            node_id = props.get("id")
            if node_id:
                nodes_dict[str(node_id)] = props

        edges_query = (
            f"MATCH (a:`{BASE_LABEL}`)-[r]->(b:`{BASE_LABEL}`) "
            f"RETURN a.id as source_id, b.id as target_id, type(r) as rel_type, r"
        )
        edges_result = await self.query(edges_query)
        edges_data = []
        for r in edges_result:
            edge = r.get("r")
            props = edge.properties if hasattr(edge, "properties") else {}
            edges_data.append((r["source_id"], r["target_id"], r["rel_type"], props))

        return (list(nodes_dict.items()), edges_data)

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        """Get graph metrics (node and edge counts)."""
        nodes_res = await self.query("MATCH (n) RETURN count(n) as count", {})
        edges_res = await self.query("MATCH ()-[r]->() RETURN count(r) as count", {})
        nodes_count = int((nodes_res[0].get("count", 0) if nodes_res else 0) or 0)
        edges_count = int((edges_res[0].get("count", 0) if edges_res else 0) or 0)
        return {"nodes": nodes_count, "edges": edges_count}

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """Check if an edge exists between two nodes."""
        q = (
            f"MATCH (a:`{BASE_LABEL}` {{id: $source}})-[r:`{relationship_name}`]->"
            f"(b:`{BASE_LABEL}` {{id: $target}}) RETURN count(r) as count"
        )
        res = await self.query(q, {"source": str(source_id), "target": str(target_id)})
        return bool(res and int(res[0].get("count", 0) or 0) > 0)

    async def has_edges(self, edges: List[Any]) -> List[Any]:
        """Check which edges exist."""
        existing: List[Any] = []
        for e in edges:
            if isinstance(e, tuple) and len(e) >= 3:
                src, dst, rel = e[0], e[1], e[2]
            else:
                continue
            if await self.has_edge(str(src), str(dst), str(rel)):
                existing.append(e)
        return existing

    async def get_edges(self, node_id: str) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Get all edges connected to a node."""
        out: List[Tuple[str, str, str, Dict[str, Any]]] = []

        # Outgoing edges
        q_out = (
            f"MATCH (a:`{BASE_LABEL}` {{id: $id}})-[r]->(b:`{BASE_LABEL}`) "
            f"RETURN a.id as source_id, b.id as target_id, type(r) as rel_type, r"
        )
        for r in await self.query(q_out, {"id": str(node_id)}):
            edge = r.get("r")
            props = edge.properties if hasattr(edge, "properties") else {}
            out.append((r["source_id"], r["target_id"], r["rel_type"], props))

        # Incoming edges
        q_in = (
            f"MATCH (a:`{BASE_LABEL}`)-[r]->(b:`{BASE_LABEL}` {{id: $id}}) "
            f"RETURN a.id as source_id, b.id as target_id, type(r) as rel_type, r"
        )
        for r in await self.query(q_in, {"id": str(node_id)}):
            edge = r.get("r")
            props = edge.properties if hasattr(edge, "properties") else {}
            out.append((r["source_id"], r["target_id"], r["rel_type"], props))

        return out

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all neighboring nodes."""
        q = f"MATCH (a:`{BASE_LABEL}` {{id: $id}})--(b:`{BASE_LABEL}`) RETURN DISTINCT b"
        res = await self.query(q, {"id": str(node_id)})
        out: List[Dict[str, Any]] = []
        for r in res:
            n = r.get("b") or r.get(0)
            if not n:
                continue
            out.append(
                n.properties if hasattr(n, "properties") else (n if isinstance(n, dict) else {})
            )
        return out

    async def get_connections(
        self, node_id: Union[str, UUID]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """Get all connections for a node."""
        q = (
            f"MATCH (a:`{BASE_LABEL}` {{id: $id}})-[r]-(b:`{BASE_LABEL}`) "
            f"RETURN a as a, b as b, type(r) as rel_type, r"
        )
        res = await self.query(q, {"id": str(node_id)})
        out: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
        for row in res:
            a = row.get("a")
            b = row.get("b")
            r_obj = row.get("r")
            rel_type = row.get("rel_type")
            a_props = (
                a.properties if hasattr(a, "properties") else (a if isinstance(a, dict) else {})
            )
            b_props = (
                b.properties if hasattr(b, "properties") else (b if isinstance(b, dict) else {})
            )
            r_props = r_obj.properties if hasattr(r_obj, "properties") else {}
            rel = {"type": rel_type, **(r_props or {})}
            out.append((a_props, rel, b_props))
        return out

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str]
    ) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]]:
        """Get a subgraph containing nodes of a specific type."""
        label = getattr(node_type, "__name__", str(node_type))
        
        # Modified to fetch neighbors as well, matching Neo4j behavior
        if node_name:
            nodes_res = await self.query(
                f"MATCH (n:`{BASE_LABEL}`:`{label}`) WHERE n.name IN $names "
                f"OPTIONAL MATCH (n)--(nbr) "
                f"RETURN n, nbr",
                {"names": node_name},
            )
        else:
            nodes_res = await self.query(
                f"MATCH (n:`{BASE_LABEL}`:`{label}`) "
                f"OPTIONAL MATCH (n)--(nbr) "
                f"RETURN n, nbr",
                {},
            )

        unique_nodes: Dict[str, Any] = {}
        
        for r in nodes_res:
            # Process primary node 'n'
            n = r.get("n")
            if n:
               props = n.properties if hasattr(n, "properties") else (n if isinstance(n, dict) else {})
               nid = str(props.get("id"))
               if nid and nid not in unique_nodes:
                   unique_nodes[nid] = props
            
            # Process neighbor node 'nbr'
            nbr = r.get("nbr")
            if nbr:
               props = nbr.properties if hasattr(nbr, "properties") else (nbr if isinstance(nbr, dict) else {})
               nid = str(props.get("id"))
               if nid and nid not in unique_nodes:
                   unique_nodes[nid] = props

        nodes: List[Tuple[int, dict]] = []
        id_to_idx: Dict[str, int] = {}
        
        for i, (nid, props) in enumerate(unique_nodes.items()):
            nodes.append((i, props))
            id_to_idx[nid] = i

        node_ids = list(id_to_idx.keys())
        if not node_ids:
            return nodes, []

        edges_res = await self.query(
            f"MATCH (a:`{BASE_LABEL}`)-[r]-(b:`{BASE_LABEL}`) "
            f"WHERE a.id IN $ids AND b.id IN $ids "
            f"RETURN a.id as source_id, b.id as target_id, type(r) as rel_type, r",
            {"ids": node_ids},
        )
        edges: List[Tuple[int, int, str, dict]] = []
        for r in edges_res:
            edge = r.get("r")
            props = edge.properties if hasattr(edge, "properties") else {}
            s = id_to_idx.get(str(r["source_id"]))
            t = id_to_idx.get(str(r["target_id"]))
            if s is None or t is None:
                continue
            edges.append((s, t, r["rel_type"], props))
        return nodes, edges

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Get filtered graph data based on attribute filters."""
        nodes, edges = await self.get_graph_data()
        if not attribute_filters:
            return nodes, edges

        def matches(node_props: Dict[str, Any]) -> bool:
            for f in attribute_filters:
                for k, vals in f.items():
                    if k in node_props and str(node_props[k]) in {str(v) for v in vals}:
                        return True
            return False

        kept_ids = {node_id for node_id, props in nodes if matches(props)}
        filtered_nodes = [(nid, props) for nid, props in nodes if nid in kept_ids]
        filtered_edges = [e for e in edges if e[0] in kept_ids and e[1] in kept_ids]
        return filtered_nodes, filtered_edges
