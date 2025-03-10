"""Adapter for Kuzu graph database."""

import logging
import json
import os
import shutil
import asyncio
from typing import Dict, Any, List, Union, Optional, Tuple
from datetime import datetime, timezone
from uuid import UUID
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import kuzu
from kuzu.database import Database
from kuzu import Connection
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.engine import DataPoint

logger = logging.getLogger(__name__)


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


class KuzuAdapter(GraphDBInterface):
    """Adapter for Kuzu graph database operations with improved consistency and async support."""

    def __init__(self, db_path: str):
        """Initialize Kuzu database connection and schema."""
        self.db_path = db_path  # Path for the database directory
        self.db: Optional[Database] = None
        self.conn: Optional[Connection] = None
        self.executor = ThreadPoolExecutor()
        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize the Kuzu database connection and schema."""
        try:
            os.makedirs(self.db_path, exist_ok=True)
            self.db = Database(self.db_path)
            self.db.init_database()
            self.conn = Connection(self.db)
            # Create node table with essential fields and timestamp
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Node(
                    id STRING PRIMARY KEY,
                    text STRING,
                    type STRING,
                    dataset STRING,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    properties STRING
                )
            """)
            # Create relationship table with timestamp
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS EDGE(
                    FROM Node TO Node,
                    relationship_name STRING,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    properties STRING
                )
            """)
            logger.debug("Kuzu database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kuzu database: {e}")
            raise

    async def query(self, query: str, params: dict = None) -> List[Tuple]:
        """Execute a Kuzu query asynchronously with automatic reconnection."""
        loop = asyncio.get_running_loop()

        def blocking_query():
            try:
                if not self.conn:
                    logger.debug("Reconnecting to Kuzu database...")
                    self._initialize_connection()
                result = self.conn.execute(query, params if params else {})
                rows = []
                while result.has_next():
                    row = result.get_next()
                    processed_row = []
                    for val in row:
                        if hasattr(val, "as_py"):
                            val = val.as_py()
                        processed_row.append(val)
                    rows.append(tuple(processed_row))
                return rows
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise

        return await loop.run_in_executor(self.executor, blocking_query)

    @asynccontextmanager
    async def get_session(self):
        """Get a database session.

        Kuzu doesn't have session management like Neo4j, so this provides API compatibility.
        """
        try:
            yield self.conn
        finally:
            pass

    # Helper method for parsing nodes

    def _parse_node(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw node result (with JSON properties) into a dictionary."""
        if data.get("properties"):
            try:
                props = json.loads(data["properties"])
                # Remove the JSON field and merge its contents
                data.pop("properties")
                data.update(props)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse properties JSON for node {data.get('id')}")
        return data

    # Helper method for building edge queries

    def _edge_query_and_params(
        self, from_node: str, to_node: str, relationship_name: str, properties: Dict[str, Any]
    ) -> Tuple[str, dict]:
        """Build the edge creation query and parameters."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        query = """
            MATCH (from:Node), (to:Node)
            WHERE from.id = $from_id AND to.id = $to_id
            CREATE (from)-[r:EDGE {
                relationship_name: $relationship_name,
                created_at: timestamp($created_at),
                updated_at: timestamp($updated_at),
                properties: $properties
            }]->(to)
        """
        params = {
            "from_id": from_node,
            "to_id": to_node,
            "relationship_name": relationship_name,
            "created_at": now,
            "updated_at": now,
            "properties": json.dumps(properties, cls=UUIDEncoder),
        }
        return query, params

    # Node Operations

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        query_str = "MATCH (n:Node) WHERE n.id = $id RETURN COUNT(n) > 0"
        result = await self.query(query_str, {"id": node_id})
        return result[0][0] if result else False

    async def add_node(self, node: DataPoint) -> None:
        """Add a single node to the graph if it doesn't exist."""
        try:
            properties = node.model_dump() if hasattr(node, "model_dump") else vars(node)

            # Extract core fields with defaults if not present
            core_properties = {
                "id": str(properties.get("id", "")),
                "text": str(properties.get("text", "")),
                "type": str(properties.get("type", "")),
                "dataset": str(properties.get("dataset", "")),
            }

            # Remove core fields from other properties
            for key in core_properties:
                properties.pop(key, None)

            core_properties["properties"] = json.dumps(properties, cls=UUIDEncoder)

            # Check if node exists
            match_query = """
            MATCH (n:Node)
            WHERE n.id = $id
            RETURN COUNT(n)
            """
            result = await self.query(match_query, {"id": core_properties["id"]})
            exists = result[0][0] > 0 if result else False

            if not exists:
                # Add timestamps for new node
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
                fields = []
                params = {}
                for key, value in core_properties.items():
                    if value is not None:
                        param_name = f"param_{key}"
                        fields.append(f"{key}: ${param_name}")
                        params[param_name] = value

                # Add timestamp fields
                fields.extend(
                    ["created_at: timestamp($created_at)", "updated_at: timestamp($updated_at)"]
                )
                params.update({"created_at": now, "updated_at": now})

                create_query = f"""
                CREATE (n:Node {{{", ".join(fields)}}})
                """
                await self.query(create_query, params)

        except Exception as e:
            logger.error(f"Failed to add node: {e}")
            raise

    async def add_nodes(self, nodes: List[DataPoint]) -> None:
        """Add multiple nodes to the graph."""
        for node in nodes:
            await self.add_node(node)

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and its relationships."""
        query_str = "MATCH (n:Node) WHERE n.id = $id DETACH DELETE n"
        await self.query(query_str, {"id": node_id})

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """Delete multiple nodes at once."""
        query_str = "MATCH (n:Node) WHERE n.id IN $ids DETACH DELETE n"
        await self.query(query_str, {"ids": node_ids})

    async def extract_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Extract a node by its ID."""
        query_str = """
        MATCH (n:Node)
        WHERE n.id = $id
        RETURN {
            id: n.id,
            text: n.text,
            type: n.type,
            dataset: n.dataset,
            properties: n.properties
        }
        """
        try:
            result = await self.query(query_str, {"id": node_id})
            if result and result[0]:
                # Use helper to parse JSON properties
                node_data = self._parse_node(result[0][0])
                return node_data
            return None
        except Exception as e:
            logger.error(f"Failed to extract node {node_id}: {e}")
            return None

    async def extract_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Extract multiple nodes by their IDs."""
        query_str = """
        MATCH (n:Node)
        WHERE n.id IN $node_ids
        RETURN {
            id: n.id,
            text: n.text,
            type: n.type,
            dataset: n.dataset,
            properties: n.properties
        }
        """
        try:
            results = await self.query(query_str, {"node_ids": node_ids})
            # Parse each node using the same helper function
            nodes = [self._parse_node(row[0]) for row in results if row[0]]
            return nodes
        except Exception as e:
            logger.error(f"Failed to extract nodes: {e}")
            return []

    # Edge Operations

    async def has_edge(self, from_node: str, to_node: str, edge_label: str) -> bool:
        """Check if an edge exists between nodes with the given relationship name."""
        query_str = """
        MATCH (from:Node)-[r:EDGE]->(to:Node)
        WHERE from.id = $from_id AND to.id = $to_id AND r.relationship_name = $edge_label
        RETURN COUNT(r) > 0
        """
        result = await self.query(
            query_str, {"from_id": from_node, "to_id": to_node, "edge_label": edge_label}
        )
        return result[0][0] if result else False

    async def has_edges(self, edges: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Check if multiple edges exist."""
        result_edges = []
        for from_node, to_node, edge_label in edges:
            if await self.has_edge(from_node, to_node, edge_label):
                result_edges.append((from_node, to_node, edge_label))
        return result_edges

    async def add_edge(
        self,
        from_node: str,
        to_node: str,
        relationship_name: str,
        edge_properties: Dict[str, Any] = {},
    ) -> None:
        """Add an edge between two nodes."""
        try:
            query, params = self._edge_query_and_params(
                from_node, to_node, relationship_name, edge_properties
            )
            await self.query(query, params)
        except Exception as e:
            logger.error(f"Failed to add edge: {e}")
            raise

    async def add_edges(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]) -> None:
        """Add multiple edges at once."""
        for from_node, to_node, relationship_name, properties in edges:
            query, params = self._edge_query_and_params(
                from_node, to_node, relationship_name, properties
            )
            await self.query(query, params)

    async def get_edges(self, node_id: str) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        """Get all edges connected to a node."""
        query_str = """
        MATCH (n:Node)-[r]-(m:Node)
        WHERE n.id = $node_id
        RETURN {
            id: n.id,
            text: n.text,
            type: n.type,
            dataset: n.dataset
        },
        r.relationship_name,
        {
            id: m.id,
            text: m.text,
            type: m.type,
            dataset: m.dataset,
            properties: r.properties
        }
        """
        try:
            results = await self.query(query_str, {"node_id": node_id})
            edges = []
            for row in results:
                if row and len(row) == 3:
                    # Parse edge properties JSON for the connected node if available
                    if isinstance(row[2], dict) and "properties" in row[2]:
                        try:
                            props = json.loads(row[2]["properties"]) if row[2]["properties"] else {}
                            row[2].update(props)
                            del row[2]["properties"]
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse properties JSON for edge connected to {node_id}"
                            )
                    edges.append((row[0], row[1], row[2]))
            if not edges:
                # Return a self-referential edge if none found
                edges.append(
                    (
                        {"id": node_id, "text": "", "type": "", "dataset": ""},
                        "SELF",
                        {"id": node_id, "text": "", "type": "", "dataset": ""},
                    )
                )
            return edges
        except Exception as e:
            logger.error(f"Failed to get edges for node {node_id}: {e}")
            return [
                (
                    {"id": node_id, "text": "", "type": "", "dataset": ""},
                    "SELF",
                    {"id": node_id, "text": "", "type": "", "dataset": ""},
                )
            ]

    # Neighbor Operations

    async def get_neighbours(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all neighbouring nodes."""
        query_str = """
        MATCH (n)-[r]-(m)
        WHERE n.id = $id
        RETURN DISTINCT properties(m)
        """
        result = await self.query(query_str, {"id": node_id})
        return [row[0] for row in result]

    async def get_predecessors(
        self, node_id: Union[str, UUID], edge_label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all predecessor nodes."""
        if edge_label:
            query_str = """
            MATCH (n)<-[r:EDGE]-(m)
            WHERE n.id = $id AND r.relationship_name = $edge_label
            RETURN properties(m)
            """
            params = {"id": str(node_id), "edge_label": edge_label}
        else:
            query_str = """
            MATCH (n)<-[r:EDGE]-(m)
            WHERE n.id = $id
            RETURN properties(m)
            """
            params = {"id": str(node_id)}
        result = await self.query(query_str, params)
        return [row[0] for row in result]

    async def get_successors(
        self, node_id: Union[str, UUID], edge_label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all successor nodes."""
        if edge_label:
            query_str = """
            MATCH (n)-[r:EDGE]->(m)
            WHERE n.id = $id AND r.relationship_name = $edge_label
            RETURN properties(m)
            """
            params = {"id": str(node_id), "edge_label": edge_label}
        else:
            query_str = """
            MATCH (n)-[r:EDGE]->(m)
            WHERE n.id = $id
            RETURN properties(m)
            """
            params = {"id": str(node_id)}
        result = await self.query(query_str, params)
        return [row[0] for row in result]

    async def get_connections(
        self, node_id: str
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """Get all nodes connected to a given node."""
        query_str = """
        MATCH (n:Node)-[r:EDGE]-(m:Node)
        WHERE n.id = $node_id
        RETURN {
            id: n.id,
            text: n.text,
            type: n.type,
            dataset: n.dataset,
            properties: n.properties
        },
        {
            relationship_name: r.relationship_name,
            properties: r.properties
        },
        {
            id: m.id,
            text: m.text,
            type: m.type,
            dataset: m.dataset,
            properties: m.properties
        }
        """
        try:
            results = await self.query(query_str, {"node_id": node_id})
            edges = []
            for row in results:
                if row and len(row) == 3:
                    processed_row = []
                    for item in row:
                        if isinstance(item, dict) and "properties" in item and item["properties"]:
                            try:
                                props = json.loads(item["properties"])
                                item.update(props)
                                del item["properties"]
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse JSON properties")
                        processed_row.append(item)
                    edges.append(tuple(processed_row))
            if not edges:
                edges.append(
                    (
                        {"id": node_id, "text": "", "type": "", "dataset": ""},
                        {"relationship_name": "SELF"},
                        {"id": node_id, "text": "", "type": "", "dataset": ""},
                    )
                )
            return edges
        except Exception as e:
            logger.error(f"Failed to get connections for node {node_id}: {e}")
            return [
                (
                    {"id": node_id, "text": "", "type": "", "dataset": ""},
                    {"relationship_name": "SELF"},
                    {"id": node_id, "text": "", "type": "", "dataset": ""},
                )
            ]

    async def remove_connection_to_predecessors_of(
        self, node_ids: List[str], edge_label: str
    ) -> None:
        """Remove all incoming edges of specified type for given nodes."""
        query_str = """
        MATCH (n)<-[r:EDGE]-(m)
        WHERE n.id IN $node_ids AND r.relationship_name = $edge_label
        DELETE r
        """
        await self.query(query_str, {"node_ids": node_ids, "edge_label": edge_label})

    async def remove_connection_to_successors_of(
        self, node_ids: List[str], edge_label: str
    ) -> None:
        """Remove all outgoing edges of specified type for given nodes."""
        query_str = """
        MATCH (n)-[r:EDGE]->(m)
        WHERE n.id IN $node_ids AND r.relationship_name = $edge_label
        DELETE r
        """
        await self.query(query_str, {"node_ids": node_ids, "edge_label": edge_label})

    # Graph-wide Operations

    async def get_graph_data(
        self,
    ) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, str, str, Dict[str, Any]]]]:
        """Get all nodes and edges in the graph."""
        try:
            nodes_query = """
            MATCH (n:Node)
            RETURN n.id, {
                text: n.text,
                type: n.type,
                dataset: n.dataset,
                properties: n.properties
            }
            """
            nodes = await self.query(nodes_query)
            formatted_nodes = []
            for n in nodes:
                if n[0]:
                    node_id = str(n[0])
                    props = n[1]
                    if props.get("properties"):
                        try:
                            additional_props = json.loads(props["properties"])
                            props.update(additional_props)
                            del props["properties"]
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse properties JSON for node {node_id}")
                    formatted_nodes.append((node_id, props))
            if not formatted_nodes:
                logger.warning("No nodes found in the database")
                return [], []

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
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(
                                f"Failed to parse edge properties for {source_id}->{target_id}"
                            )
                    formatted_edges.append((source_id, target_id, rel_type, props))

            if formatted_nodes and not formatted_edges:
                logger.debug("No edges found, creating self-referential edges for nodes")
                for node_id, _ in formatted_nodes:
                    formatted_edges.append(
                        (
                            node_id,
                            node_id,
                            "SELF",
                            {
                                "relationship_name": "SELF",
                                "relationship_type": "SELF",
                                "vector_distance": 0.0,
                            },
                        )
                    )
            return formatted_nodes, formatted_edges
        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")
            raise

    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ):
        """Get filtered nodes and relationships based on attributes."""
        where_clauses = []
        params = {}
        for i, filter_dict in enumerate(attribute_filters):
            for attr, values in filter_dict.items():
                param_name = f"values_{i}_{attr}"
                where_clauses.append(f"n.{attr} IN ${param_name}")
                params[param_name] = values
        where_clause = " AND ".join(where_clauses)
        nodes_query = f"MATCH (n:Node) WHERE {where_clause} RETURN properties(n)"
        edges_query = f"""
        MATCH (n1:Node)-[r:EDGE]->(n2:Node)
        WHERE {where_clause.replace("n.", "n1.")} AND {where_clause.replace("n.", "n2.")}
        RETURN properties(r)
        """
        nodes, edges = await asyncio.gather(
            self.query(nodes_query, params), self.query(edges_query, params)
        )
        return ([n[0] for n in nodes], [e[0] for e in edges])

    async def get_graph_metrics(self, include_optional=False) -> Dict[str, Any]:
        """Get basic graph metrics."""
        metrics = {}
        node_count = await self.query("MATCH (n:Node) RETURN COUNT(n)")
        metrics["num_nodes"] = node_count[0][0] if node_count else 0
        edge_count = await self.query("MATCH ()-[r:EDGE]->() RETURN COUNT(r)")
        metrics["num_edges"] = edge_count[0][0] if edge_count else 0
        metrics["mean_degree"] = (
            (2 * metrics["num_edges"]) / metrics["num_nodes"] if metrics["num_nodes"] > 0 else 0
        )
        metrics["edge_density"] = (
            (metrics["num_edges"]) / (metrics["num_nodes"] * (metrics["num_nodes"] - 1))
            if metrics["num_nodes"] > 1
            else 0
        )
        return metrics

    async def get_disconnected_nodes(self) -> List[str]:
        """Get nodes that are not connected to any other node."""
        query_str = """
        MATCH (n:Node)
        WHERE NOT EXISTS((n)-[]-())
        RETURN n.id
        """
        result = await self.query(query_str)
        return [str(row[0]) for row in result]

    # Graph Meta-Data Operations

    async def get_model_independent_graph_data(self) -> Dict[str, List[str]]:
        """Get graph data independent of any specific data model."""
        node_labels = await self.query("MATCH (n:Node) RETURN DISTINCT labels(n)")
        rel_types = await self.query("MATCH ()-[r:EDGE]->() RETURN DISTINCT r.relationship_name")
        return {
            "node_labels": [label[0] for label in node_labels],
            "relationship_types": [rel[0] for rel in rel_types],
        }

    async def get_node_labels_string(self) -> str:
        """Get all node labels as a string."""
        labels = await self.query("MATCH (n:Node) RETURN DISTINCT labels(n)")
        return "|".join(sorted(set([label[0] for label in labels])))

    async def get_relationship_labels_string(self) -> str:
        """Get all relationship types as a string."""
        types = await self.query("MATCH ()-[r:EDGE]->() RETURN DISTINCT r.relationship_name")
        return "|".join(sorted(set([t[0] for t in types])))

    async def delete_graph(self) -> None:
        """Delete all data from the graph while preserving the database structure."""
        try:
            # Delete relationships from the fixed table EDGE
            self.conn.execute("MATCH ()-[r:EDGE]->() DELETE r")
            # Then delete nodes
            self.conn.execute("MATCH (n:Node) DELETE n")
            logger.info("Cleared all data from graph while preserving structure")
        except Exception as e:
            logger.error(f"Failed to delete graph data: {e}")
            raise

    async def clear_database(self) -> None:
        """Clear all data from the database by deleting the database files and reinitializing."""
        try:
            if self.conn:
                self.conn = None
            if self.db:
                self.db.close()
                self.db = None
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
                logger.info(f"Deleted Kuzu database files at {self.db_path}")
            # Reinitialize the database
            self._initialize_connection()
            # Verify the database is empty
            result = self.conn.execute("MATCH (n:Node) RETURN COUNT(n)")
            count = result.get_next()[0] if result.has_next() else 0
            if count > 0:
                logger.warning(
                    f"Database still contains {count} nodes after clearing, forcing deletion"
                )
                self.conn.execute("MATCH (n:Node) DETACH DELETE n")
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Error during database clearing: {e}")
            raise
