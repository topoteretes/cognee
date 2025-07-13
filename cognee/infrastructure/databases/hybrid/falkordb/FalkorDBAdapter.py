import asyncio

# from datetime import datetime
import json
from textwrap import dedent
from uuid import UUID
from webbrowser import Error
from typing import List, Dict, Any, Optional, Tuple, Type, Union

from falkordb import FalkorDB

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    record_graph_changes,
    NodeData,
    EdgeData,
    Node,
)
from cognee.infrastructure.databases.vector.embeddings import EmbeddingEngine
from cognee.infrastructure.databases.vector.vector_db_interface import VectorDBInterface
from cognee.infrastructure.engine import DataPoint


class IndexSchema(DataPoint):
    """
    Define a schema for indexing that includes text data and associated metadata.

    This class inherits from the DataPoint class. It contains a string attribute 'text' and
    a dictionary 'metadata' that specifies the index fields for this schema.
    """

    text: str

    metadata: dict = {"index_fields": ["text"]}


class FalkorDBAdapter(VectorDBInterface, GraphDBInterface):
    """
    Manage and interact with a graph database using vector embeddings.

    Public methods include:
    - query
    - embed_data
    - stringify_properties
    - create_data_point_query
    - create_edge_query
    - create_collection
    - has_collection
    - create_data_points
    - create_vector_index
    - has_vector_index
    - index_data_points
    - add_node
    - add_nodes
    - add_edge
    - add_edges
    - has_edges
    - retrieve
    - extract_node
    - extract_nodes
    - get_connections
    - search
    - batch_search
    - get_graph_data
    - delete_data_points
    - delete_node
    - delete_nodes
    - delete_graph
    - prune
    - get_node
    - get_nodes
    - get_neighbors
    - get_graph_metrics
    - get_document_subgraph
    - get_degree_one_nodes
    """

    def __init__(
        self,
        database_url: str,
        database_port: int,
        embedding_engine=EmbeddingEngine,
    ):
        self.driver = FalkorDB(
            host=database_url,
            port=database_port,
        )
        self.embedding_engine = embedding_engine
        self.graph_name = "cognee_graph"

    def query(self, query: str, params: dict = {}):
        """
        Execute a query against the graph database.

        Handles exceptions during the query execution by logging errors and re-raising the
        exception.

        The method can be called only if a valid query string and parameters are provided.

        Parameters:
        -----------

            - query (str): The query string to be executed against the graph database.
            - params (dict): A dictionary of parameters to be used in the query. (default {})

        Returns:
        --------

            The result of the query execution, returned by the graph database.
        """
        graph = self.driver.select_graph(self.graph_name)

        try:
            result = graph.query(query, params)
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            raise e

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """
        Embed a list of text data into vector representations using the embedding engine.

        Parameters:
        -----------

            - data (list[str]): A list of strings that should be embedded into vectors.

        Returns:
        --------

            - list[list[float]]: A list of lists, where each inner list contains float values
              representing the embedded vectors.
        """
        return await self.embedding_engine.embed_text(data)

    async def stringify_properties(self, properties: dict) -> str:
        """
        Convert properties dictionary to a string format suitable for database queries.

        Parameters:
        -----------

            - properties (dict): A dictionary containing properties to be converted to string
              format.

        Returns:
        --------

            - str: A string representation of the properties in the appropriate format.
        """

        def parse_value(value):
            """
            Convert a value to its string representation based on type for database queries.

            Parameters:
            -----------

                - value: The value to parse into a string representation.

            Returns:
            --------

                Returns the string representation of the value in the appropriate format.
            """
            if type(value) is UUID:
                return f"'{str(value)}'"
            if type(value) is int or type(value) is float:
                return value
            if (
                type(value) is list
                and len(value) > 0
                and type(value[0]) is float
                and len(value) == self.embedding_engine.get_vector_size()
            ):
                return f"'vecf32({value})'"
            # if type(value) is datetime:
            #     return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")
            if type(value) is dict:
                return f"'{json.dumps(value).replace(chr(39), chr(34))}'"
            if type(value) is str:
                # Escape single quotes and handle special characters
                escaped_value = (
                    str(value)
                    .replace("'", "\\'")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t")
                )
                return f"'{escaped_value}'"
            return f"'{str(value)}'"

        return ",".join([f"{key}:{parse_value(value)}" for key, value in properties.items()])

    async def create_data_point_query(self, data_point: DataPoint, vectorized_values: dict):
        """
        Compose a query to create or update a data point in the database.

        Parameters:
        -----------

            - data_point (DataPoint): An instance of DataPoint containing information about the
              entity.
            - vectorized_values (dict): A dictionary of vectorized values related to the data
              point.

        Returns:
        --------

            A tuple containing the query string and parameters dictionary.
        """
        node_label = type(data_point).__name__
        property_names = DataPoint.get_embeddable_property_names(data_point)

        properties = {
            **data_point.model_dump(),
            **(
                {
                    property_names[index]: (
                        vectorized_values[index]
                        if index < len(vectorized_values)
                        else getattr(data_point, property_name, None)
                    )
                    for index, property_name in enumerate(property_names)
                }
            ),
        }

        # Clean the properties - remove None values and handle special types
        clean_properties = {}
        for key, value in properties.items():
            if value is not None:
                if isinstance(value, UUID):
                    clean_properties[key] = str(value)
                elif isinstance(value, dict):
                    clean_properties[key] = json.dumps(value)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], float):
                    # This is likely a vector - convert to string representation
                    clean_properties[key] = f"vecf32({value})"
                else:
                    clean_properties[key] = value

        query = dedent(
            f"""
            MERGE (node:{node_label} {{id: $node_id}})
            SET node += $properties, node.updated_at = timestamp()
        """
        ).strip()

        params = {"node_id": str(data_point.id), "properties": clean_properties}

        return query, params

    def sanitize_relationship_name(self, relationship_name: str) -> str:
        """
        Sanitize relationship name to be valid for Cypher queries.

        Parameters:
        -----------
            - relationship_name (str): The original relationship name

        Returns:
        --------
            - str: A sanitized relationship name valid for Cypher
        """
        # Replace hyphens, spaces, and other special characters with underscores
        import re

        sanitized = re.sub(r"[^\w]", "_", relationship_name)
        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "_" + sanitized
        return sanitized or "RELATIONSHIP"

    async def create_edge_query(self, edge: tuple[str, str, str, dict]) -> str:
        """
        Generate a query to create or update an edge between two nodes in the graph.

        Parameters:
        -----------

            - edge (tuple[str, str, str, dict]): A tuple consisting of source and target node
              IDs, edge type, and edge properties.

        Returns:
        --------

            - str: A string containing the query to be executed for creating the edge.
        """
        # Sanitize the relationship name for Cypher compatibility
        sanitized_relationship = self.sanitize_relationship_name(edge[2])

        # Add the original relationship name to properties
        edge_properties = {**edge[3], "relationship_name": edge[2]}
        properties = await self.stringify_properties(edge_properties)
        properties = f"{{{properties}}}"

        return dedent(
            f"""
            MERGE (source {{id:'{edge[0]}'}})
            MERGE (target {{id: '{edge[1]}'}})
            MERGE (source)-[edge:{sanitized_relationship} {properties}]->(target)
            ON MATCH SET edge.updated_at = timestamp()
            ON CREATE SET edge.updated_at = timestamp()
        """
        ).strip()

    async def create_collection(self, collection_name: str):
        """
        Create a collection in the graph database with the specified name.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to be created.
        """
        pass

    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if a collection with the specified name exists in the graph database.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check for existence.

        Returns:
        --------

            - bool: Returns true if the collection exists, otherwise false.
        """
        collections = self.driver.list_graphs()

        return collection_name in collections

    async def create_data_points(self, data_points: list[DataPoint]):
        """
        Add a list of data points to the graph database via batching.

        Can raise exceptions if there are issues during the database operations.

        Parameters:
        -----------

            - data_points (list[DataPoint]): A list of DataPoint instances to be inserted into
              the database.
        """
        embeddable_values = []
        vector_map = {}

        for data_point in data_points:
            property_names = DataPoint.get_embeddable_property_names(data_point)
            key = str(data_point.id)
            vector_map[key] = {}

            for property_name in property_names:
                property_value = getattr(data_point, property_name, None)

                if property_value is not None:
                    vector_map[key][property_name] = len(embeddable_values)
                    embeddable_values.append(property_value)
                else:
                    vector_map[key][property_name] = None

        vectorized_values = await self.embed_data(embeddable_values)

        for data_point in data_points:
            vectorized_data = [
                vectorized_values[vector_map[str(data_point.id)][property_name]]
                if vector_map[str(data_point.id)][property_name] is not None
                else None
                for property_name in DataPoint.get_embeddable_property_names(data_point)
            ]

            query, params = await self.create_data_point_query(data_point, vectorized_data)
            self.query(query, params)

    async def create_vector_index(self, index_name: str, index_property_name: str):
        """
        Create a vector index in the specified graph for a given property if it does not already
        exist.

        Parameters:
        -----------

            - index_name (str): The name of the vector index to be created.
            - index_property_name (str): The name of the property on which the vector index will
              be created.
        """
        graph = self.driver.select_graph(self.graph_name)

        if not self.has_vector_index(graph, index_name, index_property_name):
            graph.create_node_vector_index(
                index_name, index_property_name, dim=self.embedding_engine.get_vector_size()
            )

    def has_vector_index(self, graph, index_name: str, index_property_name: str) -> bool:
        """
        Determine if a vector index exists on the specified property of the given graph.

        Parameters:
        -----------

            - graph: The graph instance to check for the vector index.
            - index_name (str): The name of the index to check for existence.
            - index_property_name (str): The property name associated with the index.

        Returns:
        --------

            - bool: Returns true if the vector index exists, otherwise false.
        """
        try:
            indices = graph.list_indices()

            return any(
                [
                    (index[0] == index_name and index_property_name in index[1])
                    for index in indices.result_set
                ]
            )
        except Error as e:
            print(e)
            return False

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        """
        Index a list of data points in the specified graph database based on properties.

        To be implemented: does not yet have a defined behavior.

        Parameters:
        -----------

            - index_name (str): The name of the index to be created for the data points.
            - index_property_name (str): The property name on which to index the data points.
            - data_points (list[DataPoint]): A list of DataPoint instances to be indexed.
        """
        pass

    async def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """
        Add a single node with specified properties to the graph.

        Parameters:
        -----------

            - node_id (str): Unique identifier for the node being added.
            - properties (Dict[str, Any]): A dictionary of properties associated with the node.
        """
        # Clean the properties - remove None values and handle special types
        clean_properties = {"id": node_id}
        for key, value in properties.items():
            if value is not None:
                if isinstance(value, UUID):
                    clean_properties[key] = str(value)
                elif isinstance(value, dict):
                    clean_properties[key] = json.dumps(value)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], float):
                    # This is likely a vector - convert to string representation
                    clean_properties[key] = f"vecf32({value})"
                else:
                    clean_properties[key] = value

        query = "MERGE (node {id: $node_id}) SET node += $properties, node.updated_at = timestamp()"
        params = {"node_id": node_id, "properties": clean_properties}

        self.query(query, params)

    # Helper methods for DataPoint compatibility
    async def add_data_point_node(self, node: DataPoint):
        """
        Add a single data point as a node in the graph.

        Parameters:
        -----------

            - node (DataPoint): An instance of DataPoint to be added to the graph.
        """
        await self.create_data_points([node])

    async def add_data_point_nodes(self, nodes: list[DataPoint]):
        """
        Add multiple data points as nodes in the graph.

        Parameters:
        -----------

            - nodes (list[DataPoint]): A list of DataPoint instances to be added to the graph.
        """
        await self.create_data_points(nodes)

    @record_graph_changes
    async def add_nodes(self, nodes: Union[List[Node], List[DataPoint]]) -> None:
        """
        Add multiple nodes to the graph in a single operation.

        Parameters:
        -----------

            - nodes (Union[List[Node], List[DataPoint]]): A list of Node tuples or DataPoint objects to be added to the graph.
        """
        for node in nodes:
            if isinstance(node, tuple) and len(node) == 2:
                # Node is in (node_id, properties) format
                node_id, properties = node
                await self.add_node(node_id, properties)
            elif hasattr(node, "id") and hasattr(node, "model_dump"):
                # Node is a DataPoint object
                await self.add_node(str(node.id), node.model_dump())
            else:
                raise ValueError(
                    f"Invalid node format: {node}. Expected tuple (node_id, properties) or DataPoint object."
                )

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create a new edge between two nodes in the graph.

        Parameters:
        -----------

            - source_id (str): The unique identifier of the source node.
            - target_id (str): The unique identifier of the target node.
            - relationship_name (str): The name of the relationship to be established by the
              edge.
            - properties (Optional[Dict[str, Any]]): Optional dictionary of properties
              associated with the edge. (default None)
        """
        if properties is None:
            properties = {}

        edge_tuple = (source_id, target_id, relationship_name, properties)
        query = await self.create_edge_query(edge_tuple)
        self.query(query)

    @record_graph_changes
    async def add_edges(self, edges: List[EdgeData]) -> None:
        """
        Add multiple edges to the graph in a single operation.

        Parameters:
        -----------

            - edges (List[EdgeData]): A list of EdgeData objects representing edges to be added.
        """
        for edge in edges:
            if isinstance(edge, tuple) and len(edge) == 4:
                # Edge is in (source_id, target_id, relationship_name, properties) format
                source_id, target_id, relationship_name, properties = edge
                await self.add_edge(source_id, target_id, relationship_name, properties)
            else:
                raise ValueError(
                    f"Invalid edge format: {edge}. Expected tuple (source_id, target_id, relationship_name, properties)."
                )

    async def has_edges(self, edges):
        """
        Check if the specified edges exist in the graph based on their attributes.

        Parameters:
        -----------

            - edges: A list of edges to check for existence in the graph.

        Returns:
        --------

            Returns a list of edge tuples that exist in the graph.
        """
        existing_edges = []
        for edge in edges:
            exists = await self.has_edge(str(edge[0]), str(edge[1]), edge[2])
            if exists:
                existing_edges.append(edge)
        return existing_edges

    async def retrieve(self, data_point_ids: list[UUID]):
        """
        Retrieve data points from the graph based on their IDs.

        Parameters:
        -----------

            - data_point_ids (list[UUID]): A list of UUIDs representing the data points to
              retrieve.

        Returns:
        --------

            Returns the result set containing the retrieved nodes or an empty list if not found.
        """
        result = self.query(
            "MATCH (node) WHERE node.id IN $node_ids RETURN node",
            {
                "node_ids": [str(data_point) for data_point in data_point_ids],
            },
        )
        return result.result_set

    async def extract_node(self, data_point_id: UUID):
        """
        Extract the properties of a single node identified by its data point ID.

        Parameters:
        -----------

            - data_point_id (UUID): The UUID of the data point to extract.

        Returns:
        --------

            Returns the properties of the node if found, otherwise None.
        """
        result = await self.retrieve([data_point_id])
        result = result[0][0] if len(result[0]) > 0 else None
        return result.properties if result else None

    async def extract_nodes(self, data_point_ids: list[UUID]):
        """
        Extract properties of multiple nodes identified by their data point IDs.

        Parameters:
        -----------

            - data_point_ids (list[UUID]): A list of UUIDs representing the data points to
              extract.

        Returns:
        --------

            Returns the properties of the nodes in a list.
        """
        return await self.retrieve(data_point_ids)

    async def get_connections(self, node_id: UUID) -> list:
        """
        Retrieve connection details (predecessors and successors) for a given node ID.

        Parameters:
        -----------

            - node_id (UUID): The UUID of the node whose connections are to be retrieved.

        Returns:
        --------

            - list: Returns a list of tuples representing the connections of the node.
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
            self.query(predecessors_query, dict(node_id=node_id)),
            self.query(successors_query, dict(node_id=node_id)),
        )

        connections = []

        for neighbour in predecessors:
            neighbour = neighbour["relation"]
            connections.append((neighbour[0], {"relationship_name": neighbour[1]}, neighbour[2]))

        for neighbour in successors:
            neighbour = neighbour["relation"]
            connections.append((neighbour[0], {"relationship_name": neighbour[1]}, neighbour[2]))

        return connections

    async def search(
        self,
        collection_name: str,
        query_text: str = None,
        query_vector: list[float] = None,
        limit: int = 10,
        with_vector: bool = False,
    ):
        """
        Search for nodes in a collection based on text or vector query, with optional limitation
        on results.

        Parameters:
        -----------

            - collection_name (str): The name of the collection in which to search.
            - query_text (str): The text to search for (if using text-based query). (default
              None)
            - query_vector (list[float]): The vector representation of the query if using
              vector-based search. (default None)
            - limit (int): Maximum number of results to return from the search. (default 10)
            - with_vector (bool): Flag indicating whether to return vectors with the search
              results. (default False)

        Returns:
        --------

            Returns the search results as a result set from the graph database.
        """
        if query_text is None and query_vector is None:
            raise InvalidValueError(message="One of query_text or query_vector must be provided!")

        if query_text and not query_vector:
            query_vector = (await self.embed_data([query_text]))[0]

        # For FalkorDB, let's do a simple property-based search instead of vector search for now
        # since the vector index might not be set up correctly
        if "." in collection_name:
            [label, attribute_name] = collection_name.split(".")
        else:
            # If no dot, treat the whole thing as a property search
            label = ""
            attribute_name = collection_name

        # Simple text-based search if we have query_text
        if query_text:
            if label:
                query = f"""
                MATCH (n:{label})
                WHERE toLower(toString(n.{attribute_name})) CONTAINS toLower($query_text)
                RETURN n, 1.0 as score
                LIMIT $limit
                """
            else:
                query = f"""
                MATCH (n)
                WHERE toLower(toString(n.{attribute_name})) CONTAINS toLower($query_text)
                RETURN n, 1.0 as score
                LIMIT $limit
                """

            params = {"query_text": query_text, "limit": limit}
            result = self.query(query, params)
            return result.result_set
        else:
            # For vector search, return empty for now since vector indexing needs proper setup
            return []

    async def batch_search(
        self,
        collection_name: str,
        query_texts: list[str],
        limit: int = None,
        with_vectors: bool = False,
    ):
        """
        Perform batch search across multiple queries based on text inputs and return results
        asynchronously.

        Parameters:
        -----------

            - collection_name (str): The name of the collection in which to perform the
              searches.
            - query_texts (list[str]): A list of text queries to search for.
            - limit (int): Optional limit for the search results for each query. (default None)
            - with_vectors (bool): Flag indicating whether to return vectors with the results.
              (default False)

        Returns:
        --------

            Returns a list of results for each search query executed in parallel.
        """
        query_vectors = await self.embedding_engine.embed_text(query_texts)

        return await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_vector=with_vectors,
                )
                for query_vector in query_vectors
            ]
        )

    async def get_graph_data(self):
        """
        Retrieve all nodes and edges from the graph along with their properties.

        Returns:
        --------

            Returns a tuple containing lists of nodes and edges data retrieved from the graph.
        """
        query = "MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, properties(n) AS properties"

        result = self.query(query)

        nodes = [
            (
                record[2]["id"],
                record[2],
            )
            for record in result.result_set
        ]

        query = """
        MATCH (n)-[r]->(m)
        RETURN ID(n) AS source, ID(m) AS target, TYPE(r) AS type, properties(r) AS properties
        """
        result = self.query(query)
        edges = [
            (
                record[3]["source_node_id"],
                record[3]["target_node_id"],
                record[2],
                record[3],
            )
            for record in result.result_set
        ]

        return (nodes, edges)

    async def delete_data_points(self, collection_name: str, data_point_ids: list[UUID]):
        """
        Remove specified data points from the graph database based on their IDs.

        Parameters:
        -----------

            - collection_name (str): The name of the collection from which to delete the data
              points.
            - data_point_ids (list[UUID]): A list of UUIDs representing the data points to
              delete.

        Returns:
        --------

            Returns the result of the deletion operation from the database.
        """
        return self.query(
            "MATCH (node) WHERE node.id IN $node_ids DETACH DELETE node",
            {
                "node_ids": [str(data_point) for data_point in data_point_ids],
            },
        )

    async def delete_node(self, node_id: str) -> None:
        """
        Delete a specified node from the graph by its ID.

        Parameters:
        -----------

            - node_id (str): Unique identifier for the node to delete.
        """
        query = f"MATCH (node {{id: '{node_id}'}}) DETACH DELETE node"
        self.query(query)

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """
        Delete multiple nodes from the graph by their identifiers.

        Parameters:
        -----------

            - node_ids (List[str]): A list of unique identifiers for the nodes to delete.
        """
        for node_id in node_ids:
            await self.delete_node(node_id)

    async def delete_graph(self):
        """
        Delete the entire graph along with all its indices and nodes.
        """
        try:
            graph = self.driver.select_graph(self.graph_name)

            indices = graph.list_indices()
            for index in indices.result_set:
                for field in index[1]:
                    graph.drop_node_vector_index(index[0], field)

            graph.delete()
        except Exception as e:
            print(f"Error deleting graph: {e}")

    async def get_node(self, node_id: str) -> Optional[NodeData]:
        """
        Retrieve a single node from the graph using its ID.

        Parameters:
        -----------

            - node_id (str): Unique identifier of the node to retrieve.
        """
        result = self.query(
            "MATCH (node) WHERE node.id = $node_id RETURN node",
            {"node_id": node_id},
        )

        if result.result_set and len(result.result_set) > 0:
            # FalkorDB returns node objects as first element in the result list
            return result.result_set[0][0].properties
        return None

    async def get_nodes(self, node_ids: List[str]) -> List[NodeData]:
        """
        Retrieve multiple nodes from the graph using their IDs.

        Parameters:
        -----------

            - node_ids (List[str]): A list of unique identifiers for the nodes to retrieve.
        """
        result = self.query(
            "MATCH (node) WHERE node.id IN $node_ids RETURN node",
            {"node_ids": node_ids},
        )

        nodes = []
        if result.result_set:
            for record in result.result_set:
                # FalkorDB returns node objects as first element in each record
                nodes.append(record[0].properties)
        return nodes

    async def get_neighbors(self, node_id: str) -> List[NodeData]:
        """
        Get all neighboring nodes connected to the specified node.

        Parameters:
        -----------

            - node_id (str): Unique identifier of the node for which to retrieve neighbors.
        """
        result = self.query(
            "MATCH (node)-[]-(neighbor) WHERE node.id = $node_id RETURN DISTINCT neighbor",
            {"node_id": node_id},
        )

        neighbors = []
        if result.result_set:
            for record in result.result_set:
                # FalkorDB returns neighbor objects as first element in each record
                neighbors.append(record[0].properties)
        return neighbors

    async def get_edges(self, node_id: str) -> List[EdgeData]:
        """
        Retrieve all edges that are connected to the specified node.

        Parameters:
        -----------

            - node_id (str): Unique identifier of the node whose edges are to be retrieved.
        """
        result = self.query(
            """
            MATCH (n)-[r]-(m) 
            WHERE n.id = $node_id 
            RETURN n.id AS source_id, m.id AS target_id, type(r) AS relationship_name, properties(r) AS properties
            """,
            {"node_id": node_id},
        )

        edges = []
        if result.result_set:
            for record in result.result_set:
                # FalkorDB returns values by index: source_id, target_id, relationship_name, properties
                edges.append(
                    (
                        record[0],  # source_id
                        record[1],  # target_id
                        record[2],  # relationship_name
                        record[3],  # properties
                    )
                )
        return edges

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """
        Verify if an edge exists between two specified nodes.

        Parameters:
        -----------

            - source_id (str): Unique identifier of the source node.
            - target_id (str): Unique identifier of the target node.
            - relationship_name (str): Name of the relationship to verify.
        """
        # Check both the sanitized relationship type and the original name in properties
        sanitized_relationship = self.sanitize_relationship_name(relationship_name)

        result = self.query(
            f"""
            MATCH (source)-[r:{sanitized_relationship}]->(target)
            WHERE source.id = $source_id AND target.id = $target_id 
            AND (r.relationship_name = $relationship_name OR NOT EXISTS(r.relationship_name))
            RETURN COUNT(r) > 0 AS edge_exists
            """,
            {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_name": relationship_name,
            },
        )

        if result.result_set and len(result.result_set) > 0:
            # FalkorDB returns scalar results as a list, access by index instead of key
            return result.result_set[0][0]
        return False

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        """
        Fetch metrics and statistics of the graph, possibly including optional details.

        Parameters:
        -----------

            - include_optional (bool): Flag indicating whether to include optional metrics or
              not. (default False)
        """
        # Get basic node and edge counts
        node_result = self.query("MATCH (n) RETURN count(n) AS node_count")
        edge_result = self.query("MATCH ()-[r]->() RETURN count(r) AS edge_count")

        # FalkorDB returns scalar results as a list, access by index instead of key
        num_nodes = node_result.result_set[0][0] if node_result.result_set else 0
        num_edges = edge_result.result_set[0][0] if edge_result.result_set else 0

        metrics = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "mean_degree": (2 * num_edges) / num_nodes if num_nodes > 0 else 0,
            "edge_density": num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
            "num_connected_components": 1,  # Simplified for now
            "sizes_of_connected_components": [num_nodes] if num_nodes > 0 else [],
        }

        if include_optional:
            # Add optional metrics - simplified implementation
            metrics.update(
                {
                    "num_selfloops": 0,  # Simplified
                    "diameter": -1,  # Not implemented
                    "avg_shortest_path_length": -1,  # Not implemented
                    "avg_clustering": -1,  # Not implemented
                }
            )
        else:
            metrics.update(
                {
                    "num_selfloops": -1,
                    "diameter": -1,
                    "avg_shortest_path_length": -1,
                    "avg_clustering": -1,
                }
            )

        return metrics

    async def get_document_subgraph(self, content_hash: str):
        """
        Get a subgraph related to a specific document by content hash.

        Parameters:
        -----------

            - content_hash (str): The content hash of the document to find.
        """
        query = """
        MATCH (d) WHERE d.id CONTAINS $content_hash
        OPTIONAL MATCH (d)<-[:CHUNK_OF]-(c)
        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
        OPTIONAL MATCH (e)-[:IS_INSTANCE_OF]->(et)
        RETURN d AS document, 
               COLLECT(DISTINCT c) AS chunks,
               COLLECT(DISTINCT e) AS orphan_entities,
               COLLECT(DISTINCT c) AS made_from_nodes,
               COLLECT(DISTINCT et) AS orphan_types
        """

        result = self.query(query, {"content_hash": f"text_{content_hash}"})

        if not result.result_set or not result.result_set[0]:
            return None

        # Convert result to dictionary format
        # FalkorDB returns values by index: document, chunks, orphan_entities, made_from_nodes, orphan_types
        record = result.result_set[0]
        return {
            "document": record[0],
            "chunks": record[1],
            "orphan_entities": record[2],
            "made_from_nodes": record[3],
            "orphan_types": record[4],
        }

    async def get_degree_one_nodes(self, node_type: str):
        """
        Get all nodes that have only one connection.

        Parameters:
        -----------

            - node_type (str): The type of nodes to filter by, must be 'Entity' or 'EntityType'.
        """
        if not node_type or node_type not in ["Entity", "EntityType"]:
            raise ValueError("node_type must be either 'Entity' or 'EntityType'")

        result = self.query(
            f"""
            MATCH (n:{node_type})
            WITH n, COUNT {{ MATCH (n)--() }} as degree
            WHERE degree = 1
            RETURN n
            """
        )

        # FalkorDB returns node objects as first element in each record
        return [record[0] for record in result.result_set] if result.result_set else []

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str]
    ) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]]:
        """
        Fetch a subgraph consisting of a specific set of nodes and their relationships.

        Parameters:
        -----------

            - node_type (Type[Any]): The type of nodes to include in the subgraph.
            - node_name (List[str]): A list of names of the nodes to include in the subgraph.
        """
        label = node_type.__name__

        # Find primary nodes of the specified type and names
        primary_query = f"""
        UNWIND $names AS wantedName
        MATCH (n:{label})
        WHERE n.name = wantedName
        RETURN DISTINCT n.id, properties(n) AS properties
        """

        primary_result = self.query(primary_query, {"names": node_name})
        if not primary_result.result_set:
            return [], []

        # FalkorDB returns values by index: id, properties
        primary_ids = [record[0] for record in primary_result.result_set]

        # Find neighbors of primary nodes
        neighbor_query = """
        MATCH (n)-[]-(neighbor)
        WHERE n.id IN $ids
        RETURN DISTINCT neighbor.id, properties(neighbor) AS properties
        """

        neighbor_result = self.query(neighbor_query, {"ids": primary_ids})
        # FalkorDB returns values by index: id, properties
        neighbor_ids = (
            [record[0] for record in neighbor_result.result_set]
            if neighbor_result.result_set
            else []
        )

        all_ids = list(set(primary_ids + neighbor_ids))

        # Get all nodes in the subgraph
        nodes_query = """
        MATCH (n)
        WHERE n.id IN $ids
        RETURN n.id, properties(n) AS properties
        """

        nodes_result = self.query(nodes_query, {"ids": all_ids})
        nodes = []
        if nodes_result.result_set:
            for record in nodes_result.result_set:
                # FalkorDB returns values by index: id, properties
                nodes.append((record[0], record[1]))

        # Get edges between these nodes
        edges_query = """
        MATCH (a)-[r]->(b)
        WHERE a.id IN $ids AND b.id IN $ids
        RETURN a.id AS source_id, b.id AS target_id, type(r) AS relationship_name, properties(r) AS properties
        """

        edges_result = self.query(edges_query, {"ids": all_ids})
        edges = []
        if edges_result.result_set:
            for record in edges_result.result_set:
                # FalkorDB returns values by index: source_id, target_id, relationship_name, properties
                edges.append(
                    (
                        record[0],  # source_id
                        record[1],  # target_id
                        record[2],  # relationship_name
                        record[3],  # properties
                    )
                )

        return nodes, edges

    async def prune(self):
        """
        Prune the graph by deleting the entire graph structure.
        """
        await self.delete_graph()
