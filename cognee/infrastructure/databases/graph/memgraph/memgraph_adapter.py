"""Memgraph Adapter for Graph Database"""

import json
from cognee.shared.logging_utils import get_logger, ERROR
import asyncio
from textwrap import dedent
from typing import Optional, Any, List, Dict, Type, Tuple
from contextlib import asynccontextmanager
from uuid import UUID
from neo4j import AsyncSession
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.modules.storage.utils import JSONEncoder
from cognee.infrastructure.databases.exceptions.exceptions import NodesetFilterNotSupportedError

logger = get_logger("MemgraphAdapter", level=ERROR)


class MemgraphAdapter(GraphDBInterface):
    """
    Handles interaction with a Memgraph database through various graph operations.

    Public methods include:
    - get_session
    - query
    - has_node
    - add_node
    - add_nodes
    - extract_node
    - extract_nodes
    - delete_node
    - delete_nodes
    - has_edge
    - has_edges
    - add_edge
    - add_edges
    - get_edges
    - get_disconnected_nodes
    - get_predecessors
    - get_successors
    - get_neighbours
    - get_connections
    - remove_connection_to_predecessors_of
    - remove_connection_to_successors_of
    - delete_graph
    - serialize_properties
    - get_model_independent_graph_data
    - get_graph_data
    - get_nodeset_subgraph
    - get_filtered_graph_data
    - get_node_labels_string
    - get_relationship_labels_string
    - get_graph_metrics
    """

    def __init__(
        self,
        graph_database_url: str,
        graph_database_username: Optional[str] = None,
        graph_database_password: Optional[str] = None,
        driver: Optional[Any] = None,
    ):
        # Only use auth if both username and password are provided
        auth = None
        if graph_database_username and graph_database_password:
            auth = (graph_database_username, graph_database_password)

        self.driver = driver or AsyncGraphDatabase.driver(
            graph_database_url,
            auth=auth,
            max_connection_lifetime=120,
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """
        Manage a session with the database, yielding the session for use in operations.
        """
        async with self.driver.session() as session:
            yield session

    async def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a provided query on the Memgraph database and return the results.

        Parameters:
        -----------

            - query (str): The Cypher query to be executed against the database.
            - params (Optional[Dict[str, Any]]): Optional parameters to be used in the query.
              (default None)

        Returns:
        --------

            - List[Dict[str, Any]]: A list of dictionaries representing the result set of the
              query.
        """
        try:
            async with self.get_session() as session:
                result = await session.run(query, params)
                data = await result.data()
                return data
        except Neo4jError as error:
            logger.error("Memgraph query error: %s", error, exc_info=True)
            raise error

    async def has_node(self, node_id: str) -> bool:
        """
        Determine if a node with the given ID exists in the database.

        Parameters:
        -----------

            - node_id (str): The ID of the node to check for existence.

        Returns:
        --------

            - bool: True if the node exists; otherwise, False.
        """
        results = await self.query(
            """
                MATCH (n)
                WHERE n.id = $node_id
                RETURN COUNT(n) > 0 AS node_exists
            """,
            {"node_id": node_id},
        )
        return results[0]["node_exists"] if len(results) > 0 else False

    async def add_node(self, node: DataPoint):
        """
        Add a new node to the database with specified properties.

        Parameters:
        -----------

            - node (DataPoint): The DataPoint object representing the node to add.

        Returns:
        --------

            The result of the node addition, including its internal ID and node ID.
        """
        serialized_properties = self.serialize_properties(node.model_dump())

        query = """
        MERGE (node {id: $node_id})
        ON CREATE SET node:$node_label, node += $properties, node.updated_at = timestamp()
        ON MATCH SET node:$node_label, node += $properties, node.updated_at = timestamp()
        RETURN ID(node) AS internal_id, node.id AS nodeId
        """

        params = {
            "node_id": str(node.id),
            "node_label": type(node).__name__,
            "properties": serialized_properties,
        }
        return await self.query(query, params)

    async def add_nodes(self, nodes: list[DataPoint]) -> None:
        """
        Add multiple nodes to the database in a single operation.

        Parameters:
        -----------

            - nodes (list[DataPoint]): A list of DataPoint objects representing the nodes to
              add.

        Returns:
        --------

            - None: None.
        """
        query = """
        UNWIND $nodes AS node
        MERGE (n {id: node.node_id})
        ON CREATE SET n:node.label, n += node.properties, n.updated_at = timestamp()
        ON MATCH SET n:node.label, n += node.properties, n.updated_at = timestamp()
        RETURN ID(n) AS internal_id, n.id AS nodeId
        """

        nodes = [
            {
                "node_id": str(node.id),
                "label": type(node).__name__,
                "properties": self.serialize_properties(node.model_dump()),
            }
            for node in nodes
        ]

        results = await self.query(query, dict(nodes=nodes))
        return results

    async def extract_node(self, node_id: str):
        """
        Retrieve a single node based on its ID.

        Parameters:
        -----------

            - node_id (str): The ID of the node to retrieve.

        Returns:
        --------

            The node corresponding to the provided ID, or None if not found.
        """
        results = await self.extract_nodes([node_id])

        return results[0] if len(results) > 0 else None

    async def extract_nodes(self, node_ids: List[str]):
        """
        Retrieve multiple nodes based on their IDs.

        Parameters:
        -----------

            - node_ids (List[str]): A list of IDs for the nodes to retrieve.

        Returns:
        --------

            A list of nodes corresponding to the provided IDs.
        """
        query = """
        UNWIND $node_ids AS id
        MATCH (node {id: id})
        RETURN node"""

        params = {"node_ids": node_ids}

        results = await self.query(query, params)

        return [result["node"] for result in results]

    async def delete_node(self, node_id: str):
        """
        Delete a node from the database based on its ID.

        Parameters:
        -----------

            - node_id (str): The ID of the node to delete.

        Returns:
        --------

            None.
        """
        sanitized_id = node_id.replace(":", "_")

        query = "MATCH (node: {{id: $node_id}}) DETACH DELETE node"
        params = {"node_id": sanitized_id}

        return await self.query(query, params)

    async def delete_nodes(self, node_ids: list[str]) -> None:
        """
        Delete multiple nodes from the database based on their IDs.

        Parameters:
        -----------

            - node_ids (list[str]): A list of IDs for the nodes to delete.

        Returns:
        --------

            - None: None.
        """
        query = """
        UNWIND $node_ids AS id
        MATCH (node {id: id})
        DETACH DELETE node"""

        params = {"node_ids": node_ids}

        return await self.query(query, params)

    async def has_edge(self, from_node: UUID, to_node: UUID, edge_label: str) -> bool:
        """
        Check if a directed edge exists between two nodes identified by their IDs.

        Parameters:
        -----------

            - from_node (UUID): The ID of the source node.
            - to_node (UUID): The ID of the target node.
            - edge_label (str): The label of the edge to check.

        Returns:
        --------

            - bool: True if the edge exists; otherwise, False.
        """
        query = """
            MATCH (from_node)-[relationship]->(to_node)
            WHERE from_node.id = $from_node_id AND to_node.id = $to_node_id AND type(relationship) = $edge_label
            RETURN COUNT(relationship) > 0 AS edge_exists
        """

        params = {
            "from_node_id": str(from_node),
            "to_node_id": str(to_node),
            "edge_label": edge_label,
        }

        records = await self.query(query, params)
        return records[0]["edge_exists"] if records else False

    async def has_edges(self, edges):
        """
        Check for the existence of multiple edges based on provided criteria.

        Parameters:
        -----------

            - edges: A list of edges to verify existence for.

        Returns:
        --------

            A list of boolean values indicating the existence of each edge.
        """
        query = """
            UNWIND $edges AS edge
            MATCH (a)-[r]->(b)
            WHERE id(a) = edge.from_node AND id(b) = edge.to_node AND type(r) = edge.relationship_name
            RETURN edge.from_node AS from_node, edge.to_node AS to_node, edge.relationship_name AS relationship_name, count(r) > 0 AS edge_exists
        """

        try:
            params = {
                "edges": [
                    {
                        "from_node": str(edge[0]),
                        "to_node": str(edge[1]),
                        "relationship_name": edge[2],
                    }
                    for edge in edges
                ],
            }

            results = await self.query(query, params)
            return [result["edge_exists"] for result in results]
        except Neo4jError as error:
            logger.error("Memgraph query error: %s", error, exc_info=True)
            raise error

    async def add_edge(
        self,
        from_node: UUID,
        to_node: UUID,
        relationship_name: str,
        edge_properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a directed edge between two nodes with optional properties.

        Parameters:
        -----------

            - from_node (UUID): The ID of the source node.
            - to_node (UUID): The ID of the target node.
            - relationship_name (str): The type/label of the relationship to create.
            - edge_properties (Optional[Dict[str, Any]]): Optional properties associated with
              the edge. (default None)

        Returns:
        --------

            The result of the edge addition operation, including relationship details.
        """

        exists = await asyncio.gather(self.has_node(str(from_node)), self.has_node(str(to_node)))

        if not all(exists):
            return None

        serialized_properties = self.serialize_properties(edge_properties or {})

        query = dedent(
            f"""\
            MATCH (from_node {{id: $from_node}}),
                  (to_node {{id: $to_node}})
            WHERE from_node IS NOT NULL AND to_node IS NOT NULL
            MERGE (from_node)-[r:{relationship_name}]->(to_node)
            ON CREATE SET r += $properties, r.updated_at = timestamp()
            ON MATCH SET r += $properties, r.updated_at = timestamp()
            RETURN r
            """
        )

        params = {
            "from_node": str(from_node),
            "to_node": str(to_node),
            "relationship_name": relationship_name,
            "properties": serialized_properties,
        }

        return await self.query(query, params)

    async def add_edges(self, edges: list[tuple[str, str, str, dict[str, Any]]]) -> None:
        """
        Batch add multiple edges between nodes, enforcing specified relationships.

        Parameters:
        -----------

            - edges (list[tuple[str, str, str, dict[str, Any]]): A list of tuples containing
              specifications for each edge to add.

        Returns:
        --------

            - None: None.
        """
        query = """
            UNWIND $edges AS edge
            MATCH (from_node {id: edge.from_node})
            MATCH (to_node {id: edge.to_node})
            CALL merge.relationship(
                from_node,
                edge.relationship_name,
                {
                    source_node_id: edge.from_node,
                    target_node_id: edge.to_node
                },
                edge.properties,
                to_node,
                {}
            ) YIELD rel
            RETURN rel"""

        edges = [
            {
                "from_node": str(edge[0]),
                "to_node": str(edge[1]),
                "relationship_name": edge[2],
                "properties": {
                    **(edge[3] if edge[3] else {}),
                    "source_node_id": str(edge[0]),
                    "target_node_id": str(edge[1]),
                },
            }
            for edge in edges
        ]

        try:
            results = await self.query(query, dict(edges=edges))
            return results
        except Neo4jError as error:
            logger.error("Memgraph query error: %s", error, exc_info=True)
            raise error

    async def get_edges(self, node_id: str):
        """
        Retrieve all edges connected to a specific node identified by its ID.

        Parameters:
        -----------

            - node_id (str): The ID of the node for which to retrieve connected edges.

        Returns:
        --------

            A list of tuples representing the edges connected to the node.
        """
        query = """
        MATCH (n {id: $node_id})-[r]-(m)
        RETURN n, r, m
        """

        results = await self.query(query, dict(node_id=node_id))

        return [
            (result["n"]["id"], result["m"]["id"], {"relationship_name": result["r"][1]})
            for result in results
        ]

    async def get_disconnected_nodes(self) -> list[str]:
        """
        Identify nodes in the graph that do not belong to the largest connected component.

        Returns:
        --------

            - list[str]: A list of IDs representing the disconnected nodes.
        """
        query = """
        // Step 1: Collect all nodes
        MATCH (n)
        WITH COLLECT(n) AS nodes

        // Step 2: Find all connected components
        WITH nodes
        CALL {
          WITH nodes
          UNWIND nodes AS startNode
          MATCH path = (startNode)-[*]-(connectedNode)
          WITH startNode, COLLECT(DISTINCT connectedNode) AS component
          RETURN component
        }

        // Step 3: Aggregate components
        WITH COLLECT(component) AS components

        // Step 4: Identify the largest connected component
        UNWIND components AS component
        WITH component
        ORDER BY SIZE(component) DESC
        LIMIT 1
        WITH component AS largestComponent

        // Step 5: Find nodes not in the largest connected component
        MATCH (n)
        WHERE NOT n IN largestComponent
        RETURN COLLECT(ID(n)) AS ids
        """

        results = await self.query(query)
        return results[0]["ids"] if len(results) > 0 else []

    async def get_predecessors(self, node_id: str, edge_label: str = None) -> list[str]:
        """
        Retrieve all predecessors of a node based on its ID and optional edge label.

        Parameters:
        -----------

            - node_id (str): The ID of the node to find predecessors for.
            - edge_label (str): Optional edge label to filter predecessors. (default None)

        Returns:
        --------

            - list[str]: A list of predecessor node IDs.
        """
        if edge_label is not None:
            query = """
            MATCH (node)<-[r]-(predecessor)
            WHERE node.id = $node_id AND type(r) = $edge_label
            RETURN predecessor
            """

            results = await self.query(
                query,
                dict(
                    node_id=node_id,
                    edge_label=edge_label,
                ),
            )

            return [result["predecessor"] for result in results]
        else:
            query = """
            MATCH (node)<-[r]-(predecessor)
            WHERE node.id = $node_id
            RETURN predecessor
            """

            results = await self.query(
                query,
                dict(
                    node_id=node_id,
                ),
            )

            return [result["predecessor"] for result in results]

    async def get_successors(self, node_id: str, edge_label: str = None) -> list[str]:
        """
        Retrieve all successors of a node based on its ID and optional edge label.

        Parameters:
        -----------

            - node_id (str): The ID of the node to find successors for.
            - edge_label (str): Optional edge label to filter successors. (default None)

        Returns:
        --------

            - list[str]: A list of successor node IDs.
        """
        if edge_label is not None:
            query = """
            MATCH (node)-[r]->(successor)
            WHERE node.id = $node_id AND type(r) = $edge_label
            RETURN successor
            """

            results = await self.query(
                query,
                dict(
                    node_id=node_id,
                    edge_label=edge_label,
                ),
            )

            return [result["successor"] for result in results]
        else:
            query = """
            MATCH (node)-[r]->(successor)
            WHERE node.id = $node_id
            RETURN successor
            """

            results = await self.query(
                query,
                dict(
                    node_id=node_id,
                ),
            )

            return [result["successor"] for result in results]

    async def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get both predecessors and successors of a node.

        Parameters:
        -----------

            - node_id (str): The ID of the node to find neighbors for.

        Returns:
        --------

            - List[Dict[str, Any]]: A combined list of neighbor node IDs.
        """
        predecessors, successors = await asyncio.gather(
            self.get_predecessors(node_id), self.get_successors(node_id)
        )

        return predecessors + successors

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node by ID."""
        query = """
        MATCH (node {id: $node_id})
        RETURN node
        """
        results = await self.query(query, {"node_id": node_id})
        return results[0]["node"] if results else None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple nodes by their IDs."""
        query = """
        UNWIND $node_ids AS id
        MATCH (node {id: id})
        RETURN node
        """
        results = await self.query(query, {"node_ids": node_ids})
        return [result["node"] for result in results]

    async def get_connections(self, node_id: UUID) -> list:
        """
        Retrieve connections for a given node, including both predecessors and successors.

        Parameters:
        -----------

            - node_id (UUID): The ID of the node for which to retrieve connections.

        Returns:
        --------

            - list: A list of connections associated with the node.
        """
        predecessors_query = """
        MATCH (node)<-[relation]-(neighbour)
        WHERE node.id = $node_id
        RETURN neighbour, relation, node
        """
        successors_query = """
        MATCH (node)-[relation]->(neighbour)
        WHERE node.id = $node_id
        RETURN node, relation, neighbour
        """

        predecessors, successors = await asyncio.gather(
            self.query(predecessors_query, dict(node_id=str(node_id))),
            self.query(successors_query, dict(node_id=str(node_id))),
        )

        connections = []

        for neighbour in predecessors:
            neighbour = neighbour["relation"]
            connections.append((neighbour[0], {"relationship_name": neighbour[1]}, neighbour[2]))

        for neighbour in successors:
            neighbour = neighbour["relation"]
            connections.append((neighbour[0], {"relationship_name": neighbour[1]}, neighbour[2]))

        return connections

    async def remove_connection_to_predecessors_of(
        self, node_ids: list[str], edge_label: str
    ) -> None:
        """
        Remove specified connections to the predecessors of the given node IDs.

        Parameters:
        -----------

            - node_ids (list[str]): A list of node IDs from which to remove predecessor
              connections.
            - edge_label (str): The label of the edges to remove.

        Returns:
        --------

            - None: None.
        """
        query = f"""
        UNWIND $node_ids AS nid
        MATCH (node {id: nid})-[r]->(predecessor)
        WHERE type(r) = $edge_label
        DELETE r;
        """

        params = {"node_ids": node_ids, "edge_label": edge_label}

        return await self.query(query, params)

    async def remove_connection_to_successors_of(
        self, node_ids: list[str], edge_label: str
    ) -> None:
        """
        Remove specified connections to the successors of the given node IDs.

        Parameters:
        -----------

            - node_ids (list[str]): A list of node IDs from which to remove successor
              connections.
            - edge_label (str): The label of the edges to remove.

        Returns:
        --------

            - None: None.
        """
        query = f"""
        UNWIND $node_ids AS id
        MATCH (node:`{id}`)<-[r:{edge_label}]-(successor)
        DELETE r;
        """

        params = {"node_ids": node_ids}

        return await self.query(query, params)

    async def delete_graph(self):
        """
        Completely delete the graph from the database, removing all nodes and edges.

        Returns:
        --------

            None.
        """
        query = """MATCH (node)
                DETACH DELETE node;"""

        return await self.query(query)

    def serialize_properties(self, properties=dict()):
        """
        Convert property values to a suitable representation for storage.

        Parameters:
        -----------

            - properties: A dictionary of properties to serialize. (default dict())

        Returns:
        --------

            A dictionary of serialized properties.
        """
        serialized_properties = {}

        for property_key, property_value in properties.items():
            if isinstance(property_value, UUID):
                serialized_properties[property_key] = str(property_value)
                continue

            if isinstance(property_value, dict):
                serialized_properties[property_key] = json.dumps(property_value, cls=JSONEncoder)
                continue

            serialized_properties[property_key] = property_value

        return serialized_properties

    async def get_model_independent_graph_data(self):
        """
        Fetch nodes and relationships without any specific model filtering.

        Returns:
        --------

            A tuple containing nodes and edges as collections.
        """
        query_nodes = "MATCH (n) RETURN collect(n) AS nodes"
        nodes = await self.query(query_nodes)

        query_edges = "MATCH (n)-[r]->(m) RETURN collect([n, r, m]) AS elements"
        edges = await self.query(query_edges)

        return (nodes, edges)

    async def get_graph_data(self):
        """
        Retrieve all nodes and edges from the graph, including their properties.

        Returns:
        --------

            A tuple containing lists of nodes and edges.
        """
        query = "MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, properties(n) AS properties"

        result = await self.query(query)

        nodes = [
            (
                record["id"],
                record["properties"],
            )
            for record in result
        ]

        query = """
        MATCH (n)-[r]->(m)
        RETURN ID(n) AS source, ID(m) AS target, TYPE(r) AS type, properties(r) AS properties
        """
        result = await self.query(query)
        edges = [
            (
                record["source"],
                record["target"],
                record["type"],
                record["properties"],
            )
            for record in result
        ]

        return (nodes, edges)

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str]
    ) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]]:
        """
        Throw an error indicating that node set filtering is not supported.

        Parameters:
        -----------

            - node_type (Type[Any]): The type of nodes to filter.
            - node_name (List[str]): A list of node names to filter.
        """
        raise NodesetFilterNotSupportedError

    async def get_filtered_graph_data(self, attribute_filters):
        """
        Fetch nodes and relationships based on specified attribute filters.

        Parameters:
        -----------

            - attribute_filters: A list of criteria to filter nodes and relationships.

        Returns:
        --------

            A tuple containing filtered nodes and edges.
        """
        where_clauses = []
        for attribute, values in attribute_filters[0].items():
            values_str = ", ".join(
                f"'{value}'" if isinstance(value, str) else str(value) for value in values
            )
            where_clauses.append(f"n.{attribute} IN [{values_str}]")

        where_clause = " AND ".join(where_clauses)

        query_nodes = f"""
        MATCH (n)
        WHERE {where_clause}
        RETURN ID(n) AS id, labels(n) AS labels, properties(n) AS properties
        """
        result_nodes = await self.query(query_nodes)

        nodes = [
            (
                record["id"],
                record["properties"],
            )
            for record in result_nodes
        ]

        query_edges = f"""
        MATCH (n)-[r]->(m)
        WHERE {where_clause} AND {where_clause.replace("n.", "m.")}
        RETURN ID(n) AS source, ID(m) AS target, TYPE(r) AS type, properties(r) AS properties
        """
        result_edges = await self.query(query_edges)

        edges = [
            (
                record["source"],
                record["target"],
                record["type"],
                record["properties"],
            )
            for record in result_edges
        ]

        return (nodes, edges)

    async def get_node_labels_string(self):
        """
        Retrieve a string representation of all unique node labels in the graph.

        Returns:
        --------

            A string containing unique node labels.
        """
        node_labels_query = """
        MATCH (n)
        WITH DISTINCT labels(n) AS labelList
        UNWIND labelList AS label
        RETURN collect(DISTINCT label) AS labels;
        """
        node_labels_result = await self.query(node_labels_query)
        node_labels = node_labels_result[0]["labels"] if node_labels_result else []

        if not node_labels:
            raise ValueError("No node labels found in the database")

        node_labels_str = "[" + ", ".join(f"'{label}'" for label in node_labels) + "]"
        return node_labels_str

    async def get_relationship_labels_string(self):
        """
        Retrieve a string representation of all unique relationship types in the graph.

        Returns:
        --------

            A string containing unique relationship types.
        """
        relationship_types_query = (
            "MATCH ()-[r]->() RETURN collect(DISTINCT type(r)) AS relationships;"
        )
        relationship_types_result = await self.query(relationship_types_query)
        relationship_types = (
            relationship_types_result[0]["relationships"] if relationship_types_result else []
        )

        if not relationship_types:
            raise ValueError("No relationship types found in the database.")

        relationship_types_undirected_str = (
            "{"
            + ", ".join(f"{rel}" + ": {orientation: 'UNDIRECTED'}" for rel in relationship_types)
            + "}"
        )
        return relationship_types_undirected_str

    async def get_graph_metrics(self, include_optional=False):
        """
        Calculate and return various metrics of the graph, including mandatory and optional
        metrics.

        Parameters:
        -----------

            - include_optional: Specify whether to include optional metrics in the results.
              (default False)

        Returns:
        --------

            A dictionary containing calculated graph metrics.
        """

        try:
            # Basic metrics
            node_count = await self.query("MATCH (n) RETURN count(n)")
            edge_count = await self.query("MATCH ()-[r]->() RETURN count(r)")
            num_nodes = node_count[0][0] if node_count else 0
            num_edges = edge_count[0][0] if edge_count else 0

            # Calculate mandatory metrics
            mandatory_metrics = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "mean_degree": (2 * num_edges) / num_nodes if num_nodes > 0 else 0,
                "edge_density": (num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
            }

            # Calculate connected components
            components_query = """
            MATCH (n:Node)
            WITH n.id AS node_id
            MATCH path = (n)-[:EDGE*0..]-()
            WITH COLLECT(DISTINCT node_id) AS component
            RETURN COLLECT(component) AS components
            """
            components_result = await self.query(components_query)
            component_sizes = (
                [len(comp) for comp in components_result[0][0]] if components_result else []
            )

            mandatory_metrics.update(
                {
                    "num_connected_components": len(component_sizes),
                    "sizes_of_connected_components": component_sizes,
                }
            )

            if include_optional:
                # Self-loops
                self_loops_query = """
                MATCH (n:Node)-[r:EDGE]->(n)
                RETURN COUNT(r)
                """
                self_loops = await self.query(self_loops_query)
                num_selfloops = self_loops[0][0] if self_loops else 0

                # Shortest paths (simplified for Kuzu)
                paths_query = """
                MATCH (n:Node), (m:Node)
                WHERE n.id < m.id
                MATCH path = (n)-[:EDGE*]-(m)
                RETURN MIN(LENGTH(path)) AS length
                """
                paths = await self.query(paths_query)
                path_lengths = [p[0] for p in paths if p[0] is not None]

                # Local clustering coefficient
                clustering_query = """
                /// Step 1: Get each node with its neighbors and degree
                MATCH (n:Node)-[:EDGE]-(neighbor)
                WITH n, COLLECT(DISTINCT neighbor) AS neighbors, COUNT(DISTINCT neighbor) AS degree

                // Step 2: Pair up neighbors and check if they are connected
                UNWIND neighbors AS n1
                UNWIND neighbors AS n2
                WITH n, degree, n1, n2
                WHERE id(n1) < id(n2)  // avoid duplicate pairs

                // Step 3: Use OPTIONAL MATCH to see if n1 and n2 are connected
                OPTIONAL MATCH (n1)-[:EDGE]-(n2)
                WITH n, degree, COUNT(n2) AS triangle_count

                // Step 4: Compute local clustering coefficient
                WITH n, degree,
                    CASE WHEN degree <= 1 THEN 0.0
                        ELSE (1.0 * triangle_count) / (degree * (degree - 1) / 2.0)
                    END AS local_cc

                // Step 5: Compute average
                RETURN AVG(local_cc) AS avg_clustering_coefficient
                """
                clustering = await self.query(clustering_query)

                optional_metrics = {
                    "num_selfloops": num_selfloops,
                    "diameter": max(path_lengths) if path_lengths else -1,
                    "avg_shortest_path_length": sum(path_lengths) / len(path_lengths)
                    if path_lengths
                    else -1,
                    "avg_clustering": clustering[0][0] if clustering and clustering[0][0] else -1,
                }
            else:
                optional_metrics = {
                    "num_selfloops": -1,
                    "diameter": -1,
                    "avg_shortest_path_length": -1,
                    "avg_clustering": -1,
                }

            return {**mandatory_metrics, **optional_metrics}

        except Exception as e:
            logger.error(f"Failed to get graph metrics: {e}")
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "mean_degree": 0,
                "edge_density": 0,
                "num_connected_components": 0,
                "sizes_of_connected_components": [],
                "num_selfloops": -1,
                "diameter": -1,
                "avg_shortest_path_length": -1,
                "avg_clustering": -1,
            }
