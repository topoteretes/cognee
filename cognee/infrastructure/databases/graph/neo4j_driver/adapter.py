"""Neo4j Adapter for Graph Database"""

import json
import asyncio
from uuid import UUID
from textwrap import dedent
from neo4j import AsyncSession
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import Neo4jError
from contextlib import asynccontextmanager
from typing import Optional, Any, List, Dict, Type, Tuple

from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger, ERROR
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    record_graph_changes,
)
from cognee.modules.storage.utils import JSONEncoder

from distributed.utils import override_distributed
from distributed.tasks.queued_add_nodes import queued_add_nodes
from distributed.tasks.queued_add_edges import queued_add_edges

from .neo4j_metrics_utils import (
    get_avg_clustering,
    get_edge_density,
    get_num_connected_components,
    get_shortest_path_lengths,
    get_size_of_connected_components,
    count_self_loops,
)
from .deadlock_retry import deadlock_retry


logger = get_logger("Neo4jAdapter", level=ERROR)

BASE_LABEL = "__Node__"


class Neo4jAdapter(GraphDBInterface):
    """
    Adapter for interacting with a Neo4j graph database, implementing the GraphDBInterface.
    This class provides methods for querying, adding, deleting nodes and edges, as well as
    managing sessions and projecting graphs.
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
        elif graph_database_username or graph_database_password:
            logger = get_logger(__name__)
            logger.warning("Neo4j credentials incomplete – falling back to anonymous connection.")

        self.driver = driver or AsyncGraphDatabase.driver(
            graph_database_url,
            auth=auth,
            max_connection_lifetime=120,
            notifications_min_severity="OFF",
        )

    async def initialize(self) -> None:
        """
        Initializes the database: adds uniqueness constraint on id and performs indexing
        """
        await self.query(
            (f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{BASE_LABEL}`) REQUIRE n.id IS UNIQUE;")
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """
        Get a session for database operations.
        """
        async with self.driver.session() as session:
            yield session

    @deadlock_retry()
    async def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against the Neo4j database and return the result.

        Parameters:
        -----------

            - query (str): A string containing the Cypher query to execute.
            - params (Optional[Dict[str, Any]]): A dictionary of parameters to be passed to the
              query. (default None)

        Returns:
        --------

            - List[Dict[str, Any]]: A list of dictionaries representing the result of the query
              execution.
        """
        try:
            async with self.get_session() as session:
                result = await session.run(query, parameters=params)
                data = await result.data()
                return data
        except Neo4jError as error:
            logger.error("Neo4j query error: %s", error, exc_info=True)
            raise error

    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the specified ID exists in the database.

        Parameters:
        -----------

            - node_id (str): The ID of the node to check for existence.

        Returns:
        --------

            - bool: True if the node exists, otherwise False.
        """
        results = self.query(
            f"""
                MATCH (n:`{BASE_LABEL}`)
                WHERE n.id = $node_id
                RETURN COUNT(n) > 0 AS node_exists
            """,
            {"node_id": node_id},
        )
        return results[0]["node_exists"] if len(results) > 0 else False

    async def add_node(self, node: DataPoint):
        """
        Add a new node to the database based on the provided DataPoint object.

        Parameters:
        -----------

            - node (DataPoint): An instance of DataPoint representing the node to add.

        Returns:
        --------

            The result of the query execution, typically the ID of the added node.
        """
        serialized_properties = self.serialize_properties(node.model_dump())

        query = dedent(
            f"""MERGE (node: `{BASE_LABEL}`{{id: $node_id}})
                ON CREATE SET node += $properties, node.updated_at = timestamp()
                ON MATCH SET node += $properties, node.updated_at = timestamp()
                WITH node, $node_label AS label
                CALL apoc.create.addLabels(node, [label]) YIELD node AS labeledNode
                RETURN ID(labeledNode) AS internal_id, labeledNode.id AS nodeId"""
        )

        params = {
            "node_id": str(node.id),
            "node_label": type(node).__name__,
            "properties": serialized_properties,
        }

        return await self.query(query, params)

    @record_graph_changes
    @override_distributed(queued_add_nodes)
    async def add_nodes(self, nodes: list[DataPoint]) -> None:
        """
        Add multiple nodes to the database in a single query.

        Parameters:
        -----------

            - nodes (list[DataPoint]): A list of DataPoint instances representing the nodes to
              add.

        Returns:
        --------

            - None: None
        """
        query = f"""
        UNWIND $nodes AS node
        MERGE (n: `{BASE_LABEL}`{{id: node.node_id}})
        ON CREATE SET n += node.properties, n.updated_at = timestamp()
        ON MATCH SET n += node.properties, n.updated_at = timestamp()
        WITH n, node.label AS label
        CALL apoc.create.addLabels(n, [label]) YIELD node AS labeledNode
        RETURN ID(labeledNode) AS internal_id, labeledNode.id AS nodeId
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
        Retrieve a single node from the database by its ID.

        Parameters:
        -----------

            - node_id (str): The ID of the node to retrieve.

        Returns:
        --------

            The node represented as a dictionary, or None if it does not exist.
        """
        results = await self.extract_nodes([node_id])

        return results[0] if len(results) > 0 else None

    async def extract_nodes(self, node_ids: List[str]):
        """
        Retrieve multiple nodes from the database by their IDs.

        Parameters:
        -----------

            - node_ids (List[str]): A list of IDs for the nodes to retrieve.

        Returns:
        --------

            A list of nodes represented as dictionaries.
        """
        query = f"""
        UNWIND $node_ids AS id
        MATCH (node: `{BASE_LABEL}`{{id: id}})
        RETURN node"""

        params = {"node_ids": node_ids}

        results = await self.query(query, params)

        return [result["node"] for result in results]

    async def delete_node(self, node_id: str):
        """
        Remove a node from the database identified by its ID.

        Parameters:
        -----------

            - node_id (str): The ID of the node to delete.

        Returns:
        --------

            The result of the query execution, typically indicating success or failure.
        """
        query = f"MATCH (node: `{BASE_LABEL}`{{id: $node_id}}) DETACH DELETE node"
        params = {"node_id": node_id}

        return await self.query(query, params)

    async def delete_nodes(self, node_ids: list[str]) -> None:
        """
        Delete multiple nodes from the database using their IDs.

        Parameters:
        -----------

            - node_ids (list[str]): A list of IDs of the nodes to delete.

        Returns:
        --------

            - None: None
        """
        query = f"""
        UNWIND $node_ids AS id
        MATCH (node: `{BASE_LABEL}`{{id: id}})
        DETACH DELETE node"""

        params = {"node_ids": node_ids}

        return await self.query(query, params)

    async def has_edge(self, from_node: UUID, to_node: UUID, edge_label: str) -> bool:
        """
        Check if an edge exists between two nodes with the specified IDs and edge label.

        Parameters:
        -----------

            - from_node (UUID): The ID of the node from which the edge originates.
            - to_node (UUID): The ID of the node to which the edge points.
            - edge_label (str): The label of the edge to check for existence.

        Returns:
        --------

            - bool: True if the edge exists, otherwise False.
        """
        query = f"""
            MATCH (from_node: `{BASE_LABEL}`)-[:`{edge_label}`]->(to_node: `{BASE_LABEL}`)
            WHERE from_node.id = $from_node_id AND to_node.id = $to_node_id
            RETURN COUNT(relationship) > 0 AS edge_exists
        """

        params = {
            "from_node_id": str(from_node),
            "to_node_id": str(to_node),
        }

        edge_exists = await self.query(query, params)
        return edge_exists

    async def has_edges(self, edges):
        """
        Check if multiple edges exist based on provided edge criteria.

        Parameters:
        -----------

            - edges: A list of edge specifications to check for existence.

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
            logger.error("Neo4j query error: %s", error, exc_info=True)
            raise error

    async def add_edge(
        self,
        from_node: UUID,
        to_node: UUID,
        relationship_name: str,
        edge_properties: Optional[Dict[str, Any]] = {},
    ):
        """
        Create a new edge between two nodes with specified properties.

        Parameters:
        -----------

            - from_node (UUID): The ID of the source node of the edge.
            - to_node (UUID): The ID of the target node of the edge.
            - relationship_name (str): The type/label of the edge to create.
            - edge_properties (Optional[Dict[str, Any]]): A dictionary of properties to assign
              to the edge. (default {})

        Returns:
        --------

            The result of the query execution, typically indicating the created edge.
        """
        serialized_properties = self.serialize_properties(edge_properties)

        query = dedent(
            f"""\
            MATCH (from_node :`{BASE_LABEL}`{{id: $from_node}}),
                  (to_node :`{BASE_LABEL}`{{id: $to_node}})
            MERGE (from_node)-[r:`{relationship_name}`]->(to_node)
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

    @record_graph_changes
    @override_distributed(queued_add_edges)
    async def add_edges(self, edges: list[tuple[str, str, str, dict[str, Any]]]) -> None:
        """
        Add multiple edges between nodes in a single query.

        Parameters:
        -----------

            - edges (list[tuple[str, str, str, dict[str, Any]]]): A list of tuples where each
              tuple contains edge details to add.

        Returns:
        --------

            - None: None
        """
        query = f"""
            UNWIND $edges AS edge
            MATCH (from_node: `{BASE_LABEL}`{{id: edge.from_node}})
            MATCH (to_node: `{BASE_LABEL}`{{id: edge.to_node}})
            CALL apoc.merge.relationship(
                from_node,
                edge.relationship_name,
                {{
                    source_node_id: edge.from_node,
                    target_node_id: edge.to_node
                }},
                edge.properties,
                to_node
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
            logger.error("Neo4j query error: %s", error, exc_info=True)
            raise error

    async def get_edges(self, node_id: str):
        """
        Retrieve all edges connected to a specified node.

        Parameters:
        -----------

            - node_id (str): The ID of the node for which edges are retrieved.

        Returns:
        --------

            A list of edges connecting to the specified node, represented as tuples of details.
        """
        query = f"""
        MATCH (n: `{BASE_LABEL}`{{id: $node_id}})-[r]-(m)
        RETURN n, r, m
        """

        results = await self.query(query, dict(node_id=node_id))

        return [
            (result["n"]["id"], result["m"]["id"], {"relationship_name": result["r"][1]})
            for result in results
        ]

    async def get_disconnected_nodes(self) -> list[str]:
        """
        Find and return nodes that are not connected to any other nodes in the graph.

        Returns:
        --------

            - list[str]: A list of IDs of disconnected nodes.
        """
        # return await self.query(
        #     "MATCH (node) WHERE NOT (node)<-[:*]-() RETURN node.id as id",
        # )
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
        Retrieve the predecessor nodes of a specified node based on an optional edge label.

        Parameters:
        -----------

            - node_id (str): The ID of the node whose predecessors are to be retrieved.
            - edge_label (str): Optional edge label to filter predecessors. (default None)

        Returns:
        --------

            - list[str]: A list of predecessor node IDs.
        """
        if edge_label is not None:
            query = f"""
            MATCH (node: `{BASE_LABEL}`)<-[r:`{edge_label}`]-(predecessor)
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
        else:
            query = f"""
            MATCH (node: `{BASE_LABEL}`)<-[r]-(predecessor)
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
        Retrieve the successor nodes of a specified node based on an optional edge label.

        Parameters:
        -----------

            - node_id (str): The ID of the node whose successors are to be retrieved.
            - edge_label (str): Optional edge label to filter successors. (default None)

        Returns:
        --------

            - list[str]: A list of successor node IDs.
        """
        if edge_label is not None:
            query = f"""
            MATCH (node: `{BASE_LABEL}`)-[r:`{edge_label}`]->(successor)
            WHERE node.id = $node_id
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
            query = f"""
            MATCH (node: `{BASE_LABEL}`)-[r]->(successor)
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
        Get all neighbors of a specified node, including all directly connected nodes.

        Parameters:
        -----------

            - node_id (str): The ID of the node for which neighbors are retrieved.

        Returns:
        --------

            - List[Dict[str, Any]]: A list of neighboring nodes represented as dictionaries.
        """
        return await self.get_neighbours(node_id)

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single node based on its ID.

        Parameters:
        -----------

            - node_id (str): The ID of the node to retrieve.

        Returns:
        --------

            - Optional[Dict[str, Any]]: The requested node as a dictionary, or None if it does
              not exist.
        """
        query = f"""
        MATCH (node: `{BASE_LABEL}`{{id: $node_id}})
        RETURN node
        """
        results = await self.query(query, {"node_id": node_id})
        return results[0]["node"] if results else None

    async def get_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple nodes based on their IDs.

        Parameters:
        -----------

            - node_ids (List[str]): A list of node IDs to retrieve.

        Returns:
        --------

            - List[Dict[str, Any]]: A list of nodes represented as dictionaries.
        """
        query = f"""
        UNWIND $node_ids AS id
        MATCH (node:`{BASE_LABEL}` {{id: id}})
        RETURN node
        """
        results = await self.query(query, {"node_ids": node_ids})
        return [result["node"] for result in results]

    async def get_connections(self, node_id: UUID) -> list:
        """
        Retrieve all connections (predecessors and successors) for a specified node.

        Parameters:
        -----------

            - node_id (UUID): The ID of the node for which connections are retrieved.

        Returns:
        --------

            - list: A list of connections represented as tuples of details.
        """
        predecessors_query = f"""
        MATCH (node:`{BASE_LABEL}`)<-[relation]-(neighbour)
        WHERE node.id = $node_id
        RETURN neighbour, relation, node
        """
        successors_query = f"""
        MATCH (node:`{BASE_LABEL}`)-[relation]->(neighbour)
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
        Remove connections (edges) to all predecessors of specified nodes based on edge label.

        Parameters:
        -----------

            - node_ids (list[str]): A list of IDs of nodes from which connections are to be
              removed.
            - edge_label (str): The label of the edges to remove.

        Returns:
        --------

            - None: None
        """
        # Not understanding
        query = f"""
        UNWIND $node_ids AS id
        MATCH (node:`{id}`)-[r:{edge_label}]->(predecessor)
        DELETE r;
        """

        params = {"node_ids": node_ids}

        return await self.query(query, params)

    async def remove_connection_to_successors_of(
        self, node_ids: list[str], edge_label: str
    ) -> None:
        """
        Remove connections (edges) to all successors of specified nodes based on edge label.

        Parameters:
        -----------

            - node_ids (list[str]): A list of IDs of nodes from which connections are to be
              removed.
            - edge_label (str): The label of the edges to remove.

        Returns:
        --------

            - None: None
        """
        # Not understanding
        query = f"""
        UNWIND $node_ids AS id
        MATCH (node:`{id}`)<-[r:{edge_label}]-(successor)
        DELETE r;
        """

        params = {"node_ids": node_ids}

        return await self.query(query, params)

    async def delete_graph(self):
        """
        Delete all nodes and edges from the graph database.

        Returns:
        --------

            The result of the query execution, typically indicating success or failure.
        """
        # query = """MATCH (node)
        #         DETACH DELETE node;"""

        # return await self.query(query)

        node_labels = await self.get_node_labels()

        for label in node_labels:
            query = f"""
            MATCH (node:`{label}`)
            DETACH DELETE node;
            """

            await self.query(query)

    def serialize_properties(self, properties=dict()):
        """
        Convert properties of a node or edge into a serializable format suitable for storage.

        Parameters:
        -----------

            - properties: A dictionary of properties to serialize, defaults to an empty
              dictionary. (default dict())

        Returns:
        --------

            A dictionary with serialized property values.
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
        Retrieve the basic graph data without considering the model specifics, returning nodes
        and edges.

        Returns:
        --------

            A tuple of nodes and edges data.
        """
        query_nodes = "MATCH (n) RETURN collect(n) AS nodes"
        nodes = await self.query(query_nodes)

        query_edges = "MATCH (n)-[r]->(m) RETURN collect([n, r, m]) AS elements"
        edges = await self.query(query_edges)

        return (nodes, edges)

    async def get_graph_data(self):
        """
        Retrieve comprehensive data about nodes and relationships within the graph.

        Returns:
        --------

            A tuple containing two lists: nodes and edges with their properties.
        """
        query = "MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, properties(n) AS properties"

        result = await self.query(query)

        nodes = [
            (
                record["properties"]["id"],
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
                record["properties"]["source_node_id"],
                record["properties"]["target_node_id"],
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
        Retrieve a subgraph based on specified node names and type, including their
        relationships.

        Parameters:
        -----------

            - node_type (Type[Any]): The type of nodes to include in the subgraph.
            - node_name (List[str]): A list of names for nodes to filter the subgraph.

        Returns:
        --------

            - Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]}: A tuple
              containing nodes and edges in the requested subgraph.
        """
        label = node_type.__name__

        query = f"""
        UNWIND $names AS wantedName
        MATCH (n:`{label}`)
        WHERE n.name = wantedName
        WITH collect(DISTINCT n) AS primary
        UNWIND primary AS p
        OPTIONAL MATCH (p)--(nbr)
        WITH primary, collect(DISTINCT nbr) AS nbrs
        WITH primary + nbrs AS nodelist
        UNWIND nodelist AS node
        WITH collect(DISTINCT node) AS nodes
        MATCH (a)-[r]-(b)
        WHERE a IN nodes AND b IN nodes
        WITH nodes, collect(DISTINCT r) AS rels
        RETURN
          [n IN nodes |
             {{ id: n.id,
                properties: properties(n) }}] AS rawNodes,
          [r IN rels  |
             {{ type: type(r),
                properties: properties(r) }}] AS rawRels
        """

        result = await self.query(query, {"names": node_name})
        if not result:
            return [], []

        raw_nodes = result[0]["rawNodes"]
        raw_rels = result[0]["rawRels"]

        nodes = [(n["properties"]["id"], n["properties"]) for n in raw_nodes]
        edges = [
            (
                r["properties"]["source_node_id"],
                r["properties"]["target_node_id"],
                r["type"],
                r["properties"],
            )
            for r in raw_rels
        ]

        return nodes, edges

    async def get_filtered_graph_data(self, attribute_filters):
        """
        Fetch nodes and edges filtered by specific attribute criteria.

        Parameters:
        -----------

            - attribute_filters: A list of dictionaries representing attributes and associated
              values for filtering.

        Returns:
        --------

            A tuple containing filtered nodes and edges based on the specified criteria.
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

    async def graph_exists(self, graph_name="myGraph"):
        """
        Check if a graph with a given name exists in the database.

        Parameters:
        -----------

            - graph_name: The name of the graph to check for existence, defaults to 'myGraph'.
              (default 'myGraph')

        Returns:
        --------

            True if the graph exists, otherwise False.
        """
        query = "CALL gds.graph.list() YIELD graphName RETURN collect(graphName) AS graphNames;"
        result = await self.query(query)
        graph_names = result[0]["graphNames"] if result else []
        return graph_name in graph_names

    async def get_node_labels(self):
        """
        Fetch all node labels from the database and return them.

        Returns:
        --------

            A list of node labels.
        """
        node_labels_query = "CALL db.labels()"
        node_labels_result = await self.query(node_labels_query)
        node_labels = [record["label"] for record in node_labels_result]

        return node_labels

    async def get_relationship_labels_string(self):
        """
        Fetch all relationship types from the database and return them as a formatted string.

        Returns:
        --------

            A formatted string of relationship types.
        """
        relationship_types_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS relationships;"
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

    async def project_entire_graph(self, graph_name="myGraph"):
        """
        Project all node labels and relationship types into an in-memory graph using GDS.

        Parameters:
        -----------

            - graph_name: The name of the graph to project, defaults to 'myGraph'. (default
              'myGraph')
        """
        if await self.graph_exists(graph_name):
            return

        node_labels = await self.get_node_labels()
        relationship_types_undirected_str = await self.get_relationship_labels_string()

        query = f"""
        CALL gds.graph.project(
            '{graph_name}',
            ['{"', '".join(node_labels)}'],
            {relationship_types_undirected_str}
        ) YIELD graphName;
        """

        await self.query(query)

    async def drop_graph(self, graph_name="myGraph"):
        """
        Drop an existing graph from the database based on its name.

        Parameters:
        -----------

            - graph_name: The name of the graph to drop, defaults to 'myGraph'. (default
              'myGraph')
        """
        if await self.graph_exists(graph_name):
            drop_query = f"CALL gds.graph.drop('{graph_name}');"
            await self.query(drop_query)

    async def get_graph_metrics(self, include_optional=False):
        """
        Retrieve metrics related to the graph such as number of nodes, edges, and connected
        components.

        Parameters:
        -----------

            - include_optional: Specify whether to include optional metrics; defaults to False.
              (default False)

        Returns:
        --------

            A dictionary containing graph metrics, both mandatory and optional based on the
            input flag.
        """

        nodes, edges = await self.get_model_independent_graph_data()
        graph_name = "myGraph"
        await self.drop_graph(graph_name)
        await self.project_entire_graph(graph_name)

        num_nodes = len(nodes[0]["nodes"])
        num_edges = len(edges[0]["elements"])

        mandatory_metrics = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "mean_degree": (2 * num_edges) / num_nodes if num_nodes != 0 else None,
            "edge_density": await get_edge_density(self),
            "num_connected_components": await get_num_connected_components(self, graph_name),
            "sizes_of_connected_components": await get_size_of_connected_components(
                self, graph_name
            ),
        }

        if include_optional:
            shortest_path_lengths = await get_shortest_path_lengths(self, graph_name)
            optional_metrics = {
                "num_selfloops": await count_self_loops(self),
                "diameter": max(shortest_path_lengths) if shortest_path_lengths else -1,
                "avg_shortest_path_length": sum(shortest_path_lengths) / len(shortest_path_lengths)
                if shortest_path_lengths
                else -1,
                "avg_clustering": await get_avg_clustering(self, graph_name),
            }
        else:
            optional_metrics = {
                "num_selfloops": -1,
                "diameter": -1,
                "avg_shortest_path_length": -1,
                "avg_clustering": -1,
            }

        return mandatory_metrics | optional_metrics

    async def get_document_subgraph(self, content_hash: str):
        """
        Retrieve a subgraph related to a document identified by its content hash, including
        related entities and chunks.

        Parameters:
        -----------

            - content_hash (str): The hash identifying the document whose subgraph should be
              retrieved.

        Returns:
        --------

            The subgraph data as a dictionary, or None if not found.
        """
        query = """
        MATCH (doc)
        WHERE (doc:TextDocument OR doc:PdfDocument)
        AND doc.name = 'text_' + $content_hash

        OPTIONAL MATCH (doc)<-[:is_part_of]-(chunk:DocumentChunk)
        OPTIONAL MATCH (chunk)-[:contains]->(entity:Entity)
        WHERE NOT EXISTS {
            MATCH (entity)<-[:contains]-(otherChunk:DocumentChunk)-[:is_part_of]->(otherDoc)
            WHERE (otherDoc:TextDocument OR otherDoc:PdfDocument)
            AND otherDoc.id <> doc.id
        }
        OPTIONAL MATCH (chunk)<-[:made_from]-(made_node:TextSummary)
        OPTIONAL MATCH (entity)-[:is_a]->(type:EntityType)
        WHERE NOT EXISTS {
            MATCH (type)<-[:is_a]-(otherEntity:Entity)<-[:contains]-(otherChunk:DocumentChunk)-[:is_part_of]->(otherDoc)
            WHERE (otherDoc:TextDocument OR otherDoc:PdfDocument)
            AND otherDoc.id <> doc.id
        }

        RETURN
            collect(DISTINCT doc) as document,
            collect(DISTINCT chunk) as chunks,
            collect(DISTINCT entity) as orphan_entities,
            collect(DISTINCT made_node) as made_from_nodes,
            collect(DISTINCT type) as orphan_types
        """
        result = await self.query(query, {"content_hash": content_hash})
        return result[0] if result else None

    async def get_degree_one_nodes(self, node_type: str):
        """
        Fetch nodes of a specified type that have exactly one connection.

        Parameters:
        -----------

            - node_type (str): The type of nodes to retrieve, must be 'Entity' or 'EntityType'.

        Returns:
        --------

            A list of nodes with exactly one connection of the specified type.
        """
        if not node_type or node_type not in ["Entity", "EntityType"]:
            raise ValueError("node_type must be either 'Entity' or 'EntityType'")

        query = f"""
        MATCH (n:{node_type})
        WHERE COUNT {{ MATCH (n)--() }} = 1
        RETURN n
        """
        result = await self.query(query)
        return [record["n"] for record in result] if result else []
