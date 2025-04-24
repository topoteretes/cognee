"""Adapter for remote Kuzu graph database via REST API."""

from cognee.shared.logging_utils import get_logger
import json
from typing import Dict, Any, List, Union, Optional, Tuple
from datetime import datetime, timezone
from uuid import UUID
from contextlib import asynccontextmanager
import aiohttp
from aiohttp import BasicAuth

from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    record_graph_changes,
)
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import JSONEncoder

logger = get_logger()


class RemoteKuzuAdapter(GraphDBInterface):
    """Adapter for remote Kuzu graph database operations via REST API."""

    def __init__(self, api_url: str, username: str, password: str):
        """Initialize remote Kuzu database connection.

        Args:
            api_url: URL of the Kuzu REST API
            username: Optional username for API authentication
            password: Optional password for API authentication
        """
        self.api_url = api_url
        self.username = username
        self.password = password
        self._session = None
        self._schema_initialized = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the adapter and its session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _make_request(self, endpoint: str, data: dict) -> dict:
        """Make a request to the Kuzu API."""
        url = f"{self.api_url}{endpoint}"
        session = await self._get_session()
        try:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    error_detail = await response.text()
                    logger.error(
                        f"API request failed with status {response.status}: {error_detail}"
                    )
                    logger.error(f"Request data: {data}")
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=error_detail,
                    )
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {str(e)}")
            logger.error(f"Request data: {data}")
            raise

    async def _check_schema_exists(self) -> bool:
        """Check if the required schema exists without causing recursion."""
        try:
            # Make a direct request to check schema
            response = await self._make_request(
                "/query",
                {"query": "SELECT name FROM node_tables() WHERE name = 'Node'", "parameters": {}},
            )
            return bool(response.get("data") and response["data"][0][0])
        except Exception as e:
            logger.error(f"Failed to check schema: {e}")
            return False

    async def _create_schema(self):
        """Create the required schema tables."""
        try:
            # Create Node table if it doesn't exist
            try:
                await self._make_request(
                    "/query",
                    {
                        "query": """
                        CREATE NODE TABLE IF NOT EXISTS Node (
                            id STRING,
                            name STRING,
                            type STRING,
                            properties STRING,
                            created_at TIMESTAMP,
                            updated_at TIMESTAMP,
                            PRIMARY KEY (id)
                        )
                        """,
                        "parameters": {},
                    },
                )
            except aiohttp.ClientResponseError as e:
                if "already exists" not in str(e):
                    raise

            # Create EDGE table if it doesn't exist
            try:
                await self._make_request(
                    "/query",
                    {
                        "query": """
                        CREATE REL TABLE IF NOT EXISTS EDGE (
                            FROM Node TO Node,
                            relationship_name STRING,
                            properties STRING,
                            created_at TIMESTAMP,
                            updated_at TIMESTAMP
                        )
                        """,
                        "parameters": {},
                    },
                )
            except aiohttp.ClientResponseError as e:
                if "already exists" not in str(e):
                    raise

            self._schema_initialized = True
            logger.info("Schema initialized successfully")

        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise

    async def _initialize_schema(self):
        """Initialize the database schema if it doesn't exist."""
        if self._schema_initialized:
            return

        try:
            if not await self._check_schema_exists():
                await self._create_schema()
            else:
                self._schema_initialized = True
                logger.info("Schema already exists")

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise

    async def query(self, query: str, params: Optional[dict] = None) -> List[Tuple]:
        """Execute a Kuzu query via the REST API."""
        try:
            # Initialize schema if needed
            if not self._schema_initialized:
                await self._initialize_schema()

            response = await self._make_request(
                "/query", {"query": query, "parameters": params or {}}
            )

            # Convert response to list of tuples
            results = []
            if "data" in response:
                for row in response["data"]:
                    processed_row = []
                    for val in row:
                        if isinstance(val, dict) and "properties" in val:
                            try:
                                props = json.loads(val["properties"])
                                val.update(props)
                                del val["properties"]
                            except json.JSONDecodeError:
                                pass
                        processed_row.append(val)
                    results.append(tuple(processed_row))

            return results
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {params}")
            raise

    @asynccontextmanager
    async def get_session(self):
        """Get a database session context manager."""
        try:
            yield self
        finally:
            pass

    async def add_node(self, node: DataPoint) -> None:
        """Add a single node to the graph."""
        try:
            properties = node.model_dump() if hasattr(node, "model_dump") else vars(node)

            # Extract core fields with defaults
            core_properties = {
                "id": str(properties.get("id", "")),
                "name": str(properties.get("name", "")),
                "type": str(properties.get("type", "")),
            }

            # Remove core fields from other properties
            for key in core_properties:
                properties.pop(key, None)

            # Format timestamps in ISO format for properties
            if "created_at" in properties:
                properties["created_at"] = datetime.fromtimestamp(
                    properties["created_at"] / 1000, timezone.utc
                ).isoformat()
            if "updated_at" in properties:
                properties["updated_at"] = datetime.fromtimestamp(
                    properties["updated_at"] / 1000, timezone.utc
                ).isoformat()

            core_properties["properties"] = json.dumps(properties, cls=JSONEncoder)

            # Format timestamps for node fields in Kuzu's expected format
            now = datetime.now(timezone.utc)
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
            core_properties.update({"created_at": timestamp_str, "updated_at": timestamp_str})

            # Create node using MERGE
            query = """
            MERGE (n:Node {id: $id})
            ON CREATE SET
                n.name = $name,
                n.type = $type,
                n.properties = $properties,
                n.created_at = timestamp($created_at),
                n.updated_at = timestamp($updated_at)
            ON MATCH SET
                n.name = $name,
                n.type = $type,
                n.properties = $properties,
                n.updated_at = timestamp($updated_at)
            """

            await self.query(query, core_properties)

        except Exception as e:
            logger.error(f"Failed to add node: {e}")
            raise

    @record_graph_changes
    async def add_nodes(self, nodes: List[DataPoint]) -> None:
        """Add multiple nodes in a batch operation."""
        if not nodes:
            return

        try:
            now = datetime.now(timezone.utc)
            node_params = []

            for node in nodes:
                properties = node.model_dump() if hasattr(node, "model_dump") else vars(node)

                core_properties = {
                    "id": str(properties.get("id", "")),
                    "name": str(properties.get("name", "")),
                    "type": str(properties.get("type", "")),
                }

                for key in core_properties:
                    properties.pop(key, None)

                # Format timestamps in ISO format for properties
                if "created_at" in properties:
                    properties["created_at"] = datetime.fromtimestamp(
                        properties["created_at"] / 1000, timezone.utc
                    ).isoformat()
                if "updated_at" in properties:
                    properties["updated_at"] = datetime.fromtimestamp(
                        properties["updated_at"] / 1000, timezone.utc
                    ).isoformat()

                # Format timestamps for node fields in Kuzu's expected format
                timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

                node_params.append(
                    {
                        **core_properties,
                        "properties": json.dumps(properties, cls=JSONEncoder),
                        "created_at": timestamp_str,
                        "updated_at": timestamp_str,
                    }
                )

            if node_params:
                query = """
                UNWIND $nodes AS node
                MERGE (n:Node {id: node.id})
                ON CREATE SET
                    n.name = node.name,
                    n.type = node.type,
                    n.properties = node.properties,
                    n.created_at = timestamp(node.created_at),
                    n.updated_at = timestamp(node.updated_at)
                ON MATCH SET
                    n.name = node.name,
                    n.type = node.type,
                    n.properties = node.properties,
                    n.updated_at = timestamp(node.updated_at)
                """

                await self.query(query, {"nodes": node_params})

        except Exception as e:
            logger.error(f"Failed to add nodes in batch: {e}")
            raise

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its relationships."""
        query = "MATCH (n:Node) WHERE n.id = $id DETACH DELETE n"
        await self.query(query, {"id": node_id})

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """Delete multiple nodes at once."""
        query = "MATCH (n:Node) WHERE n.id IN $ids DETACH DELETE n"
        await self.query(query, {"ids": node_ids})

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node by ID."""
        query = """
        MATCH (n:Node)
        WHERE n.id = $id
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        }
        """
        try:
            result = await self.query(query, {"id": node_id})
            if result and result[0]:
                node_data = result[0][0]
                if isinstance(node_data, dict) and "properties" in node_data:
                    try:
                        props = json.loads(node_data["properties"])
                        node_data.update(props)
                        del node_data["properties"]
                    except json.JSONDecodeError:
                        pass
                return node_data
            return None
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple nodes by their IDs."""
        query = """
        MATCH (n:Node)
        WHERE n.id IN $node_ids
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        }
        """
        try:
            results = await self.query(query, {"node_ids": node_ids})
            nodes = []
            for row in results:
                if row and row[0]:
                    node_data = row[0]
                    if isinstance(node_data, dict) and "properties" in node_data:
                        try:
                            props = json.loads(node_data["properties"])
                            node_data.update(props)
                            del node_data["properties"]
                        except json.JSONDecodeError:
                            pass
                    nodes.append(node_data)
            return nodes
        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")
            return []

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an edge between two nodes."""
        try:
            now = datetime.now(timezone.utc)

            # Ensure properties are JSON serializable
            serializable_properties = {}
            for key, value in (properties or {}).items():
                if isinstance(value, UUID):
                    serializable_properties[key] = str(value)
                else:
                    serializable_properties[key] = value

            # Format timestamps in ISO format for properties
            if "created_at" in serializable_properties:
                serializable_properties["created_at"] = datetime.fromtimestamp(
                    serializable_properties["created_at"] / 1000, timezone.utc
                ).isoformat()
            if "updated_at" in serializable_properties:
                serializable_properties["updated_at"] = datetime.fromtimestamp(
                    serializable_properties["updated_at"] / 1000, timezone.utc
                ).isoformat()

            # Format timestamps for edge fields in Kuzu's expected format
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

            query = """
            MATCH (from:Node), (to:Node)
            WHERE from.id = $from_id AND to.id = $to_id
            MERGE (from)-[r:EDGE {
                relationship_name: $relationship_name
            }]->(to)
            ON CREATE SET
                r.created_at = timestamp($created_at),
                r.updated_at = timestamp($updated_at),
                r.properties = $properties
            ON MATCH SET
                r.updated_at = timestamp($updated_at),
                r.properties = $properties
            """

            await self.query(
                query,
                {
                    "from_id": source_id,
                    "to_id": target_id,
                    "relationship_name": relationship_name,
                    "properties": json.dumps(serializable_properties, cls=JSONEncoder),
                    "created_at": timestamp_str,
                    "updated_at": timestamp_str,
                },
            )

        except Exception as e:
            logger.error(f"Failed to add edge: {e}")
            raise

    @record_graph_changes
    async def add_edges(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]) -> None:
        """Add multiple edges in a batch operation."""
        if not edges:
            return

        try:
            now = datetime.now(timezone.utc)
            edge_params = []

            for source_id, target_id, relationship_name, properties in edges:
                # Convert UUIDs to strings if they are UUID objects
                source_id_str = str(source_id)
                target_id_str = str(target_id)

                # Ensure properties are JSON serializable
                serializable_properties = {}
                for key, value in (properties or {}).items():
                    if isinstance(value, UUID):
                        serializable_properties[key] = str(value)
                    else:
                        serializable_properties[key] = value

                # Format timestamps in ISO format for properties
                if "created_at" in serializable_properties:
                    created_at = serializable_properties["created_at"]
                    if isinstance(created_at, (int, float)):
                        serializable_properties["created_at"] = datetime.fromtimestamp(
                            created_at / 1000, timezone.utc
                        ).isoformat()
                    elif isinstance(created_at, str):
                        # If it's already a string, keep it as is
                        pass

                if "updated_at" in serializable_properties:
                    updated_at = serializable_properties["updated_at"]
                    if isinstance(updated_at, (int, float)):
                        serializable_properties["updated_at"] = datetime.fromtimestamp(
                            updated_at / 1000, timezone.utc
                        ).isoformat()
                    elif isinstance(updated_at, str):
                        # If it's already a string, keep it as is
                        pass

                # Format timestamps for edge fields in Kuzu's expected format
                timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

                edge_params.append(
                    {
                        "from_id": source_id_str,
                        "to_id": target_id_str,
                        "relationship_name": relationship_name,
                        "properties": json.dumps(serializable_properties, cls=JSONEncoder),
                        "created_at": timestamp_str,
                        "updated_at": timestamp_str,
                    }
                )

            if edge_params:
                query = """
                UNWIND $edges AS edge
                MATCH (from:Node), (to:Node)
                WHERE from.id = edge.from_id AND to.id = edge.to_id
                MERGE (from)-[r:EDGE {
                    relationship_name: edge.relationship_name
                }]->(to)
                ON CREATE SET
                    r.created_at = timestamp(edge.created_at),
                    r.updated_at = timestamp(edge.updated_at),
                    r.properties = edge.properties
                ON MATCH SET
                    r.updated_at = timestamp(edge.updated_at),
                    r.properties = edge.properties
                """

                await self.query(query, {"edges": edge_params})

        except Exception as e:
            logger.error(f"Failed to add edges in batch: {e}")
            raise

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """Check if an edge exists."""
        query = """
        MATCH (from:Node)-[r:EDGE]->(to:Node)
        WHERE from.id = $from_id AND to.id = $to_id AND r.relationship_name = $relationship_name
        RETURN COUNT(r) > 0
        """
        result = await self.query(
            query,
            {"from_id": source_id, "to_id": target_id, "relationship_name": relationship_name},
        )
        return result[0][0] if result else False

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Check if multiple edges exist."""
        if not edges:
            return []

        try:
            edge_params = [
                {"from_id": from_node, "to_id": to_node, "relationship_name": relationship_name}
                for from_node, to_node, relationship_name in edges
            ]

            query = """
            UNWIND $edges AS edge
            MATCH (from:Node)-[r:EDGE]->(to:Node)
            WHERE from.id = edge.from_id
            AND to.id = edge.to_id
            AND r.relationship_name = edge.relationship_name
            RETURN from.id, to.id, r.relationship_name
            """

            results = await self.query(query, {"edges": edge_params})
            return [(str(row[0]), str(row[1]), str(row[2])) for row in results]

        except Exception as e:
            logger.error(f"Failed to check edges in batch: {e}")
            return []

    async def get_edges(self, node_id: str) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        """Get all edges connected to a node."""
        query = """
        MATCH (n:Node)-[r:EDGE]-(m:Node)
        WHERE n.id = $node_id
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        },
        r.relationship_name,
        {
            id: m.id,
            name: m.name,
            type: m.type,
            properties: m.properties
        }
        """
        try:
            results = await self.query(query, {"node_id": node_id})
            edges = []
            for row in results:
                if row and len(row) == 3:
                    source_node = row[0]
                    target_node = row[2]

                    if isinstance(source_node, dict) and "properties" in source_node:
                        try:
                            props = json.loads(source_node["properties"])
                            source_node.update(props)
                            del source_node["properties"]
                        except json.JSONDecodeError:
                            pass

                    if isinstance(target_node, dict) and "properties" in target_node:
                        try:
                            props = json.loads(target_node["properties"])
                            target_node.update(props)
                            del target_node["properties"]
                        except json.JSONDecodeError:
                            pass

                    edges.append((source_node, row[1], target_node))
            return edges
        except Exception as e:
            logger.error(f"Failed to get edges for node {node_id}: {e}")
            return []

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all neighboring nodes."""
        return await self.get_neighbours(node_id)

    async def get_neighbours(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all neighbouring nodes."""
        query = """
        MATCH (n)-[r:EDGE]-(m)
        WHERE n.id = $id
        RETURN DISTINCT {
            id: m.id,
            name: m.name,
            type: m.type,
            properties: m.properties
        }
        """
        try:
            results = await self.query(query, {"id": node_id})
            nodes = []
            for row in results:
                if row and row[0]:
                    node_data = row[0]
                    if isinstance(node_data, dict) and "properties" in node_data:
                        try:
                            props = json.loads(node_data["properties"])
                            node_data.update(props)
                            del node_data["properties"]
                        except json.JSONDecodeError:
                            pass
                    nodes.append(node_data)
            return nodes
        except Exception as e:
            logger.error(f"Failed to get neighbours for node {node_id}: {e}")
            return []

    async def get_connections(
        self, node_id: str
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """Get all nodes connected to a given node."""
        query = """
        MATCH (n:Node)-[r:EDGE]-(m:Node)
        WHERE n.id = $node_id
        RETURN {
            id: n.id,
            name: n.name,
            type: n.type,
            properties: n.properties
        },
        {
            relationship_name: r.relationship_name,
            properties: r.properties
        },
        {
            id: m.id,
            name: m.name,
            type: m.type,
            properties: m.properties
        }
        """
        try:
            # Convert UUID to string if needed
            node_id_str = str(node_id)

            results = await self.query(query, {"node_id": node_id_str})
            connections = []
            for row in results:
                if row and len(row) == 3:
                    source_node = row[0]
                    relationship = row[1]
                    target_node = row[2]

                    if isinstance(source_node, dict) and "properties" in source_node:
                        try:
                            props = json.loads(source_node["properties"])
                            source_node.update(props)
                            del source_node["properties"]
                        except json.JSONDecodeError:
                            pass

                    if isinstance(relationship, dict) and "properties" in relationship:
                        try:
                            props = json.loads(relationship["properties"])
                            relationship.update(props)
                            del relationship["properties"]
                        except json.JSONDecodeError:
                            pass

                    if isinstance(target_node, dict) and "properties" in target_node:
                        try:
                            props = json.loads(target_node["properties"])
                            target_node.update(props)
                            del target_node["properties"]
                        except json.JSONDecodeError:
                            pass

                    connections.append((source_node, relationship, target_node))
            return connections
        except Exception as e:
            logger.error(f"Failed to get connections for node {node_id}: {e}")
            return []

    async def delete_graph(self) -> None:
        """Delete all data from the graph."""
        query = "MATCH (n:Node) DETACH DELETE n"
        await self.query(query)

    async def get_graph_data(
        self,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Get all nodes and edges in the graph."""
        try:
            # Get nodes
            nodes_query = """
            MATCH (n:Node)
            RETURN n.id, {
                name: n.name,
                type: n.type,
                properties: n.properties
            }
            """
            nodes = await self.query(nodes_query)
            formatted_nodes = []
            for n in nodes:
                if n[0]:
                    node_id = str(n[0])
                    props = n[1]
                    if isinstance(props, dict) and "properties" in props:
                        try:
                            additional_props = json.loads(props["properties"])
                            props.update(additional_props)
                            del props["properties"]
                        except json.JSONDecodeError:
                            pass
                    formatted_nodes.append((node_id, props))

            # Get edges
            edges_query = """
            MATCH (n:Node)-[r:EDGE]->(m:Node)
            RETURN n.id, m.id, r.relationship_name, r.properties
            """
            edges = await self.query(edges_query)
            formatted_edges = []
            for e in edges:
                if e and len(e) >= 3:
                    source_id = str(e[0])
                    target_id = str(e[1])
                    rel_type = str(e[2])
                    props = {}
                    if len(e) > 3 and e[3]:
                        try:
                            props = json.loads(e[3])
                        except json.JSONDecodeError:
                            pass
                    formatted_edges.append((source_id, target_id, rel_type, props))

            return formatted_nodes, formatted_edges

        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")
            return [], []

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        """Get graph metrics and statistics."""
        try:
            # Get basic metrics
            nodes, edges = await self.get_graph_data()
            num_nodes = len(nodes)
            num_edges = len(edges)

            metrics = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "mean_degree": (2 * num_edges) / num_nodes if num_nodes != 0 else None,
                "edge_density": num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
            }

            if include_optional:
                # Get connected components
                components_query = """
                MATCH (n:Node)
                WITH n.id AS node_id
                MATCH path = (n)-[:EDGE*1..3]-(m)
                WITH node_id, COLLECT(DISTINCT m.id) AS connected_nodes
                WITH COLLECT(DISTINCT connected_nodes + [node_id]) AS components
                RETURN SIZE(components) AS num_components
                """
                result = await self.query(components_query)
                num_components = result[0][0] if result else 0

                # Get component sizes
                sizes_query = """
                MATCH (n:Node)
                WITH n.id AS node_id
                MATCH path = (n)-[:EDGE*1..3]-(m)
                WITH node_id, COLLECT(DISTINCT m.id) AS connected_nodes
                WITH COLLECT(DISTINCT connected_nodes + [node_id]) AS components
                UNWIND components AS component
                RETURN SIZE(component) AS component_size
                """
                sizes = await self.query(sizes_query)
                component_sizes = [row[0] for row in sizes] if sizes else []

                # Get self-loops
                self_loops_query = """
                MATCH (n:Node)-[r:EDGE]->(n)
                RETURN COUNT(r) AS count
                """
                result = await self.query(self_loops_query)
                num_self_loops = result[0][0] if result else 0

                # Calculate diameter and average shortest path length
                diameter_query = """
                MATCH (n:Node), (m:Node)
                WHERE n.id < m.id
                MATCH path = shortestPath((n)-[:EDGE*]-(m))
                WITH n, m, path
                WHERE path IS NOT NULL
                RETURN MAX(LENGTH(path)) AS diameter, AVG(LENGTH(path)) AS avg_path_length
                """
                result = await self.query(diameter_query)
                diameter = result[0][0] if result and result[0][0] is not None else 0
                avg_shortest_path_length = (
                    result[0][1] if result and result[0][1] is not None else 0
                )

                # Calculate average clustering coefficient
                clustering_query = """
                MATCH (n:Node)-[:EDGE]-(m:Node)
                WITH n, COUNT(DISTINCT m) AS degree
                MATCH (n)-[:EDGE]-(m1:Node)-[:EDGE]-(m2:Node)-[:EDGE]-(n)
                WHERE m1.id < m2.id
                WITH n, degree, COUNT(DISTINCT {n1: m1.id, n2: m2.id}) AS triangles
                WITH AVG(CASE WHEN degree > 1 THEN 2.0 * triangles / (degree * (degree - 1)) ELSE 0 END) AS avg_clustering
                RETURN avg_clustering
                """
                result = await self.query(clustering_query)
                avg_clustering = result[0][0] if result and result[0][0] is not None else 0

                metrics.update(
                    {
                        "num_connected_components": num_components,
                        "sizes_of_connected_components": component_sizes,
                        "num_selfloops": num_self_loops,
                        "diameter": diameter,
                        "avg_shortest_path_length": avg_shortest_path_length,
                        "avg_clustering": avg_clustering,
                    }
                )

            return metrics

        except Exception as e:
            logger.error(f"Failed to get graph metrics: {e}")
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "mean_degree": 0,
                "edge_density": 0,
                "num_connected_components": 0,
                "sizes_of_connected_components": [],
                "num_selfloops": 0,
                "diameter": 0,
                "avg_shortest_path_length": 0,
                "avg_clustering": 0,
            }

    async def clear_database(self) -> None:
        """Clear all data from the database by deleting all nodes and relationships."""
        try:
            await self.delete_graph()
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            raise

    async def save_graph_to_file(self, file_path: str) -> None:
        """Not supported for remote operation."""
        logger.warning("save_graph_to_file() not supported for remote operation")
        pass

    async def load_graph_from_file(self, file_path: str) -> None:
        """Not supported for remote operation."""
        logger.warning("load_graph_from_file() not supported for remote operation")
        pass

    async def backup_to_s3(self, folder_name: str) -> None:
        """Not supported for remote operation."""
        logger.warning("backup_to_s3() not supported for remote operation")
        pass

    async def restore_from_s3(self, folder_name: str) -> None:
        """Not supported for remote operation."""
        logger.warning("restore_from_s3() not supported for remote operation")
        pass

    async def extract_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Extract a node by its ID or name.

        Args:
            node_id: The ID or name of the node to extract

        Returns:
            The node data if found, None otherwise
        """
        try:
            # First try to find by ID
            query = """
            MATCH (n:Node)
            WHERE n.id = $node_id
            RETURN {
                id: n.id,
                name: n.name,
                type: n.type,
                properties: n.properties
            }
            """
            result = await self.query(query, {"node_id": node_id})

            if result and result[0]:
                node_data = result[0][0]
                if isinstance(node_data, dict) and "properties" in node_data:
                    try:
                        props = json.loads(node_data["properties"])
                        node_data.update(props)
                        del node_data["properties"]
                    except json.JSONDecodeError:
                        pass
                return node_data

            # If not found by ID, try to find by name
            query = """
            MATCH (n:Node)
            WHERE n.name = $node_id
            RETURN {
                id: n.id,
                name: n.name,
                type: n.type,
                properties: n.properties
            }
            """
            result = await self.query(query, {"node_id": node_id})

            if result and result[0]:
                node_data = result[0][0]
                if isinstance(node_data, dict) and "properties" in node_data:
                    try:
                        props = json.loads(node_data["properties"])
                        node_data.update(props)
                        del node_data["properties"]
                    except json.JSONDecodeError:
                        pass
                return node_data

            return None

        except Exception as e:
            logger.error(f"Failed to extract node {node_id}: {e}")
            return None
