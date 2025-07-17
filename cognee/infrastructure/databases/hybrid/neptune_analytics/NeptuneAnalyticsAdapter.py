import asyncio

# from datetime import datetime
import json
from textwrap import dedent
from uuid import UUID
from webbrowser import Error

from falkordb import FalkorDB

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
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
                and type(value[0]) is float
                and len(value) == self.embedding_engine.get_vector_size()
            ):
                return f"'vecf32({value})'"
            # if type(value) is datetime:
            #     return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")
            if type(value) is dict:
                return f"'{json.dumps(value)}'"
            return f"'{value}'"

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

            A string containing the query to be executed for the data point.
        """
        node_label = type(data_point).__name__
        property_names = DataPoint.get_embeddable_property_names(data_point)

        node_properties = await self.stringify_properties(
            {
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
        )

        return dedent(
            f"""
            MERGE (node:{node_label} {{id: '{str(data_point.id)}'}})
            ON CREATE SET node += ({{{node_properties}}}), node.updated_at = timestamp()
            ON MATCH SET node += ({{{node_properties}}}), node.updated_at = timestamp()
        """
        ).strip()

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
        properties = await self.stringify_properties(edge[3])
        properties = f"{{{properties}}}"

        return dedent(
            f"""
            MERGE (source {{id:'{edge[0]}'}})
            MERGE (target {{id: '{edge[1]}'}})
            MERGE (source)-[edge:{edge[2]} {properties}]->(target)
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

        queries = [
            await self.create_data_point_query(
                data_point,
                [
                    vectorized_values[vector_map[str(data_point.id)][property_name]]
                    if vector_map[str(data_point.id)][property_name] is not None
                    else None
                    for property_name in DataPoint.get_embeddable_property_names(data_point)
                ],
            )
            for data_point in data_points
        ]

        for query in queries:
            self.query(query)

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

    async def add_node(self, node: DataPoint):
        """
        Add a single data point as a node in the graph.

        Parameters:
        -----------

            - node (DataPoint): An instance of DataPoint to be added to the graph.
        """
        await self.create_data_points([node])

    async def add_nodes(self, nodes: list[DataPoint]):
        """
        Add multiple data points as nodes in the graph.

        Parameters:
        -----------

            - nodes (list[DataPoint]): A list of DataPoint instances to be added to the graph.
        """
        await self.create_data_points(nodes)

    async def add_edge(self, edge: tuple[str, str, str, dict]):
        """
        Add an edge between two existing nodes in the graph based on the provided details.

        Parameters:
        -----------

            - edge (tuple[str, str, str, dict]): A tuple containing details of the edge to be
              added.
        """
        query = await self.create_edge_query(edge)

        self.query(query)

    async def add_edges(self, edges: list[tuple[str, str, str, dict]]):
        """
        Add multiple edges to the graph in a batch operation.

        Parameters:
        -----------

            - edges (list[tuple[str, str, str, dict]]): A list of tuples, each containing
              details of the edges to be added.
        """
        queries = [await self.create_edge_query(edge) for edge in edges]

        for query in queries:
            self.query(query)

    async def has_edges(self, edges):
        """
        Check if the specified edges exist in the graph based on their attributes.

        Parameters:
        -----------

            - edges: A list of edges to check for existence in the graph.

        Returns:
        --------

            Returns a list of boolean values indicating the existence of each edge.
        """
        query = dedent(
            """
            UNWIND $edges AS edge
            MATCH (a)-[r]->(b)
            WHERE id(a) = edge.from_node AND id(b) = edge.to_node AND type(r) = edge.relationship_name
            RETURN edge.from_node AS from_node, edge.to_node AS to_node, edge.relationship_name AS relationship_name, count(r) > 0 AS edge_exists
        """
        ).strip()

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

        results = self.query(query, params).result_set

        return [result["edge_exists"] for result in results]

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

        [label, attribute_name] = collection_name.split(".")

        query = dedent(
            f"""
            CALL db.idx.vector.queryNodes(
                '{label}',
                '{attribute_name}',
                {limit},
                vecf32({query_vector})
            ) YIELD node, score
        """
        ).strip()

        result = self.query(query)

        return result.result_set

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

    async def delete_node(self, collection_name: str, data_point_id: str):
        """
        Delete a single node specified by its data point ID from the database.

        Parameters:
        -----------

            - collection_name (str): The name of the collection containing the node to be
              deleted.
            - data_point_id (str): The ID of the data point to delete.

        Returns:
        --------

            Returns the result of the deletion operation from the database.
        """
        return await self.delete_data_points([data_point_id])

    async def delete_nodes(self, collection_name: str, data_point_ids: list[str]):
        """
        Delete multiple nodes specified by their IDs from the database.

        Parameters:
        -----------

            - collection_name (str): The name of the collection containing the nodes to be
              deleted.
            - data_point_ids (list[str]): A list of IDs of the data points to delete from the
              collection.
        """
        self.delete_data_points(data_point_ids)

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

    async def prune(self):
        """
        Prune the graph by deleting the entire graph structure.
        """
        await self.delete_graph()
