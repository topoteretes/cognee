"""Neptune Analytics Adapter for Graph Database"""

import json
from typing import Optional, Any, List, Dict, Type, Tuple
from uuid import UUID
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    record_graph_changes,
    NodeData,
    EdgeData,
    Node,
)
from cognee.modules.storage.utils import JSONEncoder
from cognee.infrastructure.engine import DataPoint
from botocore.config import Config

from .exceptions import (
    NeptuneAnalyticsConfigurationError,
)
from .neptune_utils import (
    validate_graph_id,
    validate_aws_region,
    build_neptune_config,
    format_neptune_error,
)

logger = get_logger("NeptuneGraphDB")

try:
    from langchain_aws import NeptuneAnalyticsGraph

    LANGCHAIN_AWS_AVAILABLE = True
except ImportError:
    logger.warning("langchain_aws not available. Neptune Analytics functionality will be limited.")
    LANGCHAIN_AWS_AVAILABLE = False

NEPTUNE_ENDPOINT_URL = "neptune-graph://"


class NeptuneGraphDB(GraphDBInterface):
    """
    Adapter for interacting with Amazon Neptune Analytics graph store.
    This class provides methods for querying, adding, deleting nodes and edges using the aws_langchain library.
    """

    _GRAPH_NODE_LABEL = "COGNEE_NODE"

    def __init__(
        self,
        graph_id: str,
        region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        """
        Initialize the Neptune Analytics adapter.

        Parameters:
        -----------
            - graph_id (str): The Neptune Analytics graph identifier
            - region (Optional[str]): AWS region where the graph is located (default: us-east-1)
            - aws_access_key_id (Optional[str]): AWS access key ID
            - aws_secret_access_key (Optional[str]): AWS secret access key
            - aws_session_token (Optional[str]): AWS session token for temporary credentials

        Raises:
        -------
            - NeptuneAnalyticsConfigurationError: If configuration parameters are invalid
        """
        # validate import
        if not LANGCHAIN_AWS_AVAILABLE:
            raise ImportError(
                "langchain_aws is not available. Please install it to use Neptune Analytics."
            )

        # Validate configuration
        if not validate_graph_id(graph_id):
            raise NeptuneAnalyticsConfigurationError(message=f'Invalid graph ID: "{graph_id}"')

        if region and not validate_aws_region(region):
            raise NeptuneAnalyticsConfigurationError(message=f'Invalid AWS region: "{region}"')

        self.graph_id = graph_id
        self.region = region
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token

        # Build configuration
        self.config = build_neptune_config(
            graph_id=self.graph_id,
            region=self.region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        # Initialize Neptune Analytics client using langchain_aws
        self._client: NeptuneAnalyticsGraph = self._initialize_client()
        logger.info(
            f'Initialized Neptune Analytics adapter for graph: "{graph_id}" in region: "{self.region}"'
        )

    def _initialize_client(self) -> Optional[NeptuneAnalyticsGraph]:
        """
        Initialize the Neptune Analytics client using langchain_aws.

        Returns:
        --------
            - Optional[Any]: The Neptune Analytics client or None if not available
        """
        try:
            # Initialize the Neptune Analytics Graph client
            client_config = {
                "graph_identifier": self.graph_id,
                "config": Config(user_agent_appid="Cognee"),
            }
            # Add AWS credentials if provided
            if self.region:
                client_config["region_name"] = self.region
            if self.aws_access_key_id:
                client_config["aws_access_key_id"] = self.aws_access_key_id
            if self.aws_secret_access_key:
                client_config["aws_secret_access_key"] = self.aws_secret_access_key
            if self.aws_session_token:
                client_config["aws_session_token"] = self.aws_session_token

            client = NeptuneAnalyticsGraph(**client_config)
            logger.info("Successfully initialized Neptune Analytics client")
            return client

        except Exception as e:
            raise NeptuneAnalyticsConfigurationError(
                message=f"Failed to initialize Neptune Analytics client: {format_neptune_error(e)}"
            ) from e

    @staticmethod
    def _serialize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize properties for Neptune Analytics storage.
        Parameters:
        -----------
            - properties (Dict[str, Any]): Properties to serialize.
        Returns:
        --------
            - Dict[str, Any]: Serialized properties.
        """
        serialized_properties = {}

        for property_key, property_value in properties.items():
            if isinstance(property_value, UUID):
                serialized_properties[property_key] = str(property_value)
                continue

            if isinstance(property_value, dict) or isinstance(property_value, list):
                serialized_properties[property_key] = json.dumps(property_value, cls=JSONEncoder)
                continue

            serialized_properties[property_key] = property_value

        return serialized_properties

    async def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Execute a query against the Neptune Analytics database and return the results.

        Parameters:
        -----------
            - query (str): The query string to execute against the database.
            - params (Optional[Dict[str, Any]]): A dictionary of parameters to be used in the query.

        Returns:
        --------
            - List[Any]: A list of results from the query execution.
        """
        try:
            # Execute the query using the Neptune Analytics client
            # The langchain_aws NeptuneAnalyticsGraph supports openCypher queries
            if params is None:
                params = {}
            logger.debug(f"executing na query:\nquery={query}\n")
            result = self._client.query(query, params)

            # Convert the result to list format expected by the interface
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]
            else:
                return [{"result": result}]

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Neptune Analytics query failed: {error_msg}")
            raise Exception(f"Query execution failed: {error_msg}") from e

    async def add_node(self, node: DataPoint) -> None:
        """
        Add a single node with specified properties to the graph.

        Parameters:
        -----------
            - node (DataPoint): The DataPoint object to be added to the graph.
        """
        try:
            # Prepare node properties with the ID and graph type
            serialized_properties = self._serialize_properties(node.model_dump())

            query = f"""
            MERGE (n:{self._GRAPH_NODE_LABEL} {{`~id`: $node_id}})
            ON CREATE SET n = $properties, n.updated_at = timestamp()
            ON MATCH SET n += $properties, n.updated_at = timestamp()
            RETURN n
            """

            params = {
                "node_id": str(node.id),
                "properties": serialized_properties,
            }

            result = await self.query(query, params)
            logger.debug(f"Successfully added/updated node: {node.id}")
            logger.debug(f"Successfully added/updated node: {str(result)}")

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to add node {node.id}: {error_msg}")
            raise Exception(f"Failed to add node: {error_msg}") from e

    @record_graph_changes
    async def add_nodes(self, nodes: List[DataPoint]) -> None:
        """
        Add multiple nodes to the graph in a single operation.

        Parameters:
        -----------
            - nodes (List[DataPoint]): A list of DataPoint objects to be added to the graph.
        """
        if not nodes:
            logger.debug("No nodes to add")
            return

        try:
            # Build bulk node creation query using UNWIND
            query = f"""
            UNWIND $nodes AS node
            MERGE (n:{self._GRAPH_NODE_LABEL} {{`~id`: node.node_id}})
            ON CREATE SET n = node.properties, n.updated_at = timestamp()
            ON MATCH SET n += node.properties, n.updated_at = timestamp()
            RETURN count(n) AS nodes_processed
            """

            # Prepare node data for bulk operation
            params = {
                "nodes": [
                    {
                        "node_id": str(node.id),
                        "properties": self._serialize_properties(node.model_dump()),
                    }
                    for node in nodes
                ]
            }
            result = await self.query(query, params)

            processed_count = result[0].get("nodes_processed", 0) if result else 0
            logger.debug(f"Successfully processed {processed_count} nodes in bulk operation")

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to add nodes in bulk: {error_msg}")
            # Fallback to individual node creation
            logger.info("Falling back to individual node creation")
            for node in nodes:
                try:
                    await self.add_node(node)
                except Exception as node_error:
                    logger.error(
                        f"Failed to add individual node {node.id}: {format_neptune_error(node_error)}"
                    )
                    continue

    async def delete_node(self, node_id: str) -> None:
        """
        Delete a specified node from the graph by its ID.

        Parameters:
        -----------
            - node_id (str): Unique identifier for the node to delete.
        """
        try:
            # Build openCypher query to delete the node and all its relationships
            query = f"""
            MATCH (n:{self._GRAPH_NODE_LABEL})
            WHERE id(n) = $node_id
            DETACH DELETE n
            """

            params = {"node_id": node_id}

            await self.query(query, params)
            logger.debug(f"Successfully deleted node: {node_id}")

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to delete node {node_id}: {error_msg}")
            raise Exception(f"Failed to delete node: {error_msg}") from e

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """
        Delete multiple nodes from the graph by their identifiers.

        Parameters:
        -----------
            - node_ids (List[str]): A list of unique identifiers for the nodes to delete.
        """
        if not node_ids:
            logger.debug("No nodes to delete")
            return

        try:
            # Build bulk node deletion query using UNWIND
            query = f"""
            UNWIND $node_ids AS node_id
            MATCH (n:{self._GRAPH_NODE_LABEL})
            WHERE id(n) = node_id
            DETACH DELETE n
            """

            params = {"node_ids": node_ids}
            await self.query(query, params)
            logger.debug(f"Successfully deleted {len(node_ids)} nodes in bulk operation")

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to delete nodes in bulk: {error_msg}")
            # Fallback to individual node deletion
            logger.info("Falling back to individual node deletion")
            for node_id in node_ids:
                try:
                    await self.delete_node(node_id)
                except Exception as node_error:
                    logger.error(
                        f"Failed to delete individual node {node_id}: {format_neptune_error(node_error)}"
                    )
                    continue

    async def get_node(self, node_id: str) -> Optional[NodeData]:
        """
        Retrieve a single node from the graph using its ID.

        Parameters:
        -----------
            - node_id (str): Unique identifier of the node to retrieve.

        Returns:
        --------
            - Optional[NodeData]: The node data if found, None otherwise.
        """
        try:
            # Build openCypher query to retrieve the node
            query = f"""
            MATCH (n:{self._GRAPH_NODE_LABEL})
            WHERE id(n) = $node_id
            RETURN n
            """
            params = {"node_id": node_id}

            result = await self.query(query, params)

            if result and len(result) == 1:
                # Extract node properties from the result
                logger.debug(f"Successfully retrieved node: {node_id}")
                return result[0]["n"]
            else:
                if not result:
                    logger.debug(f"Node not found: {node_id}")
                elif len(result) > 1:
                    logger.debug(f"Only one node expected, multiple returned: {node_id}")
                return None

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get node {node_id}: {error_msg}")
            raise Exception(f"Failed to get node: {error_msg}") from e

    async def get_nodes(self, node_ids: List[str]) -> List[NodeData]:
        """
        Retrieve multiple nodes from the graph using their IDs.

        Parameters:
        -----------
            - node_ids (List[str]): A list of unique identifiers for the nodes to retrieve.

        Returns:
        --------
            - List[NodeData]: A list of node data for the found nodes.
        """
        if not node_ids:
            logger.debug("No node IDs provided")
            return []

        try:
            # Build bulk node-retrieval OpenCypher query using UNWIND
            query = f"""
            UNWIND $node_ids AS node_id
            MATCH (n:{self._GRAPH_NODE_LABEL})
            WHERE id(n) = node_id
            RETURN n
            """

            params = {"node_ids": node_ids}
            result = await self.query(query, params)

            # Extract node data from results
            nodes = [record["n"] for record in result]

            logger.debug(
                f"Successfully retrieved {len(nodes)} nodes out of {len(node_ids)} requested"
            )
            return nodes

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get nodes in bulk: {error_msg}")
            # Fallback to individual node retrieval
            logger.info("Falling back to individual node retrieval")
            nodes = []
            for node_id in node_ids:
                try:
                    node_data = await self.get_node(node_id)
                    if node_data:
                        nodes.append(node_data)
                except Exception as node_error:
                    logger.error(
                        f"Failed to get individual node {node_id}: {format_neptune_error(node_error)}"
                    )
                    continue
            return nodes

    async def extract_node(self, node_id: str):
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
        MATCH (node :{self._GRAPH_NODE_LABEL}) WHERE id(node) = id
        RETURN node"""

        params = {"node_ids": node_ids}

        results = await self.query(query, params)

        return [result["node"] for result in results]

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
            - relationship_name (str): The name of the relationship to be established by the edge.
            - properties (Optional[Dict[str, Any]]): Optional dictionary of properties associated with the edge.
        """
        try:
            # Build openCypher query to create the edge
            # First ensure both nodes exist, then create the relationship

            # Prepare edge properties
            edge_props = properties or {}
            serialized_properties = self._serialize_properties(edge_props)

            query = f"""
            MATCH (source:{self._GRAPH_NODE_LABEL})
            WHERE id(source) = $source_id
            MATCH (target:{self._GRAPH_NODE_LABEL})
            WHERE id(target) = $target_id
            MERGE (source)-[r:{relationship_name}]->(target)
            ON CREATE SET r = $properties, r.updated_at = timestamp()
            ON MATCH SET r = $properties, r.updated_at = timestamp()
            RETURN r
            """

            params = {
                "source_id": source_id,
                "target_id": target_id,
                "properties": serialized_properties,
            }
            await self.query(query, params)
            logger.debug(
                f"Successfully added edge: {source_id} -[{relationship_name}]-> {target_id}"
            )

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to add edge {source_id} -> {target_id}: {error_msg}")
            raise Exception(f"Failed to add edge: {error_msg}") from e

    @record_graph_changes
    async def add_edges(self, edges: List[Tuple[str, str, str, Optional[Dict[str, Any]]]]) -> None:
        """
        Add multiple edges to the graph in a single operation.

        Parameters:
        -----------
            - edges (List[EdgeData]): A list of EdgeData objects representing edges to be added.
        """
        if not edges:
            logger.debug("No edges to add")
            return

        edges_by_relationship: dict[str, list] = {}
        for edge in edges:
            relationship_name = edge[2]
            if edges_by_relationship.get(relationship_name, None):
                edges_by_relationship[relationship_name].append(edge)
            else:
                edges_by_relationship[relationship_name] = [edge]

        results = {}
        for relationship_name, edges_for_relationship in edges_by_relationship.items():
            try:
                # Create the bulk-edge OpenCypher query using UNWIND
                query = f"""
                    UNWIND $edges AS edge
                    MATCH (source:{self._GRAPH_NODE_LABEL})
                    WHERE id(source) = edge.from_node
                    MATCH (target:{self._GRAPH_NODE_LABEL})
                    WHERE id(target) = edge.to_node
                    MERGE (source)-[r:{relationship_name}]->(target)
                    ON CREATE SET r = edge.properties, r.updated_at = timestamp()
                    ON MATCH SET r = edge.properties, r.updated_at = timestamp()
                    RETURN count(*) AS edges_processed
                    """

                # Prepare edges data for bulk operation
                params = {
                    "edges": [
                        {
                            "from_node": str(edge[0]),
                            "to_node": str(edge[1]),
                            "relationship_name": relationship_name,
                            "properties": self._serialize_properties(
                                edge[3] if len(edge) > 3 and edge[3] else {}
                            ),
                        }
                        for edge in edges_for_relationship
                    ]
                }
                results[relationship_name] = await self.query(query, params)
            except Exception as e:
                logger.error(
                    f"Failed to add edges for relationship {relationship_name}: {format_neptune_error(e)}"
                )
                logger.info("Falling back to individual edge creation")
                for edge in edges_by_relationship:
                    try:
                        source_id, target_id, relationship_name = edge[0], edge[1], edge[2]
                        properties = edge[3] if len(edge) > 3 else {}
                        await self.add_edge(source_id, target_id, relationship_name, properties)
                    except Exception as edge_error:
                        logger.error(
                            f"Failed to add individual edge {edge[0]} -> {edge[1]}: {format_neptune_error(edge_error)}"
                        )
                        continue

        processed_count = 0
        for result in results.values():
            processed_count += result[0].get("edges_processed", 0) if result else 0
        logger.debug(f"Successfully processed {processed_count} edges in bulk operation")

    async def delete_graph(self) -> None:
        """
        Delete all nodes and edges from the graph database.

        Returns:
        --------
            The result of the query execution, typically indicating success or failure.
        """
        try:
            # Build openCypher query to delete the graph
            query = f"MATCH (n:{self._GRAPH_NODE_LABEL}) DETACH DELETE n"
            await self.query(query)

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to delete graph: {error_msg}")
            raise Exception(f"Failed to delete graph: {error_msg}") from e

    async def get_graph_data(self) -> Tuple[List[Node], List[EdgeData]]:
        """
        Retrieve all nodes and edges within the graph.

        Returns:
        --------
            - Tuple[List[Node], List[EdgeData]]: A tuple containing all nodes and edges in the graph.
        """
        try:
            # Query to get all nodes
            nodes_query = f"""
            MATCH (n:{self._GRAPH_NODE_LABEL})
            RETURN id(n) AS node_id, properties(n) AS properties
            """

            # Query to get all edges
            edges_query = f"""
            MATCH (source:{self._GRAPH_NODE_LABEL})-[r]->(target:{self._GRAPH_NODE_LABEL})
            RETURN id(source) AS source_id, id(target) AS target_id, type(r) AS relationship_name, properties(r) AS properties
            """

            # Execute both queries
            nodes_result = await self.query(nodes_query)
            edges_result = await self.query(edges_query)

            # Format nodes as (node_id, properties) tuples
            nodes = [(result["node_id"], result["properties"]) for result in nodes_result]

            # Format edges as (source_id, target_id, relationship_name, properties) tuples
            edges = [
                (
                    result["source_id"],
                    result["target_id"],
                    result["relationship_name"],
                    result["properties"],
                )
                for result in edges_result
            ]

            logger.debug(f"Retrieved {len(nodes)} nodes and {len(edges)} edges from graph")
            return (nodes, edges)

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get graph data: {error_msg}")
            raise Exception(f"Failed to get graph data: {error_msg}") from e

    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        """
        Fetch metrics and statistics of the graph, possibly including optional details.

        Parameters:
        -----------
            - include_optional (bool): Flag indicating whether to include optional metrics or not.

        Returns:
        --------
            - Dict[str, Any]: A dictionary containing graph metrics and statistics.
        """
        num_nodes, num_edges = await self._get_model_independent_graph_data()
        num_cluster, list_clsuter_size = await self._get_connected_components_stat()

        mandatory_metrics = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "mean_degree": (2 * num_edges) / num_nodes if num_nodes != 0 else None,
            "edge_density": num_edges * 1.0 / (num_nodes * (num_nodes - 1))
            if num_nodes != 0
            else None,
            "num_connected_components": num_cluster,
            "sizes_of_connected_components": list_clsuter_size,
        }

        optional_metrics = {
            "num_selfloops": -1,
            "diameter": -1,
            "avg_shortest_path_length": -1,
            "avg_clustering": -1,
        }

        if include_optional:
            optional_metrics["num_selfloops"] = await self._count_self_loops()
            # Unsupported due to long-running queries when computing the shortest path for each node in the graph:
            # optional_metrics['diameter']
            # optional_metrics['avg_shortest_path_length']
            #
            # Unsupported due to incompatible algorithm: localClusteringCoefficient
            # optional_metrics['avg_clustering']

        return mandatory_metrics | optional_metrics

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """
        Verify if an edge exists between two specified nodes.

        Parameters:
        -----------
            - source_id (str): Unique identifier of the source node.
            - target_id (str): Unique identifier of the target node.
            - relationship_name (str): Name of the relationship to verify.

        Returns:
        --------
            - bool: True if the edge exists, False otherwise.
        """
        try:
            # Build openCypher query to check if the edge exists
            query = f"""
            MATCH (source:{self._GRAPH_NODE_LABEL})-[r:{relationship_name}]->(target:{self._GRAPH_NODE_LABEL})
            WHERE id(source) = $source_id AND id(target) = $target_id
            RETURN COUNT(r) > 0 AS edge_exists
            """

            params = {
                "source_id": source_id,
                "target_id": target_id,
            }

            result = await self.query(query, params)

            if result and len(result) > 0:
                edge_exists = result.pop().get("edge_exists", False)
                logger.debug(
                    f"Edge existence check for "
                    f"{source_id} -[{relationship_name}]-> {target_id}: {edge_exists}"
                )
                return edge_exists
            else:
                return False

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to check edge existence {source_id} -> {target_id}: {error_msg}")
            return False

    async def has_edges(self, edges: List[EdgeData]) -> List[EdgeData]:
        """
        Determine the existence of multiple edges in the graph.

        Parameters:
        -----------
            - edges (List[EdgeData]): A list of EdgeData objects to check for existence in the graph.

        Returns:
        --------
            - List[EdgeData]: A list of EdgeData objects that exist in the graph.
        """
        query = f"""
        UNWIND $edges AS edge
        MATCH (a:{self._GRAPH_NODE_LABEL})-[r]->(b:{self._GRAPH_NODE_LABEL})
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
            logger.debug(f"Found {len(results)} existing edges out of {len(edges)} checked")
            return [result["edge_exists"] for result in results]

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to check edges existence: {error_msg}")
            return []

    async def get_edges(self, node_id: str) -> List[EdgeData]:
        """
        Retrieve all edges that are connected to the specified node.

        Parameters:
        -----------
            - node_id (str): Unique identifier of the node whose edges are to be retrieved.

        Returns:
        --------
            - List[EdgeData]: A list of EdgeData objects representing edges connected to the node.
        """
        try:
            # Query to get all edges connected to the node (both incoming and outgoing)
            query = f"""
            MATCH (n:{self._GRAPH_NODE_LABEL})-[r]-(m:{self._GRAPH_NODE_LABEL})
            WHERE id(n) = $node_id
            RETURN
                id(n) AS source_id,
                id(m) AS target_id,
                type(r) AS relationship_name,
                properties(r) AS properties
            """

            params = {"node_id": node_id}
            result = await self.query(query, params)

            # Format edges as EdgeData tuples: (source_id, target_id, relationship_name, properties)
            edges = [self._convert_relationship_to_edge(record) for record in result]

            logger.debug(f"Retrieved {len(edges)} edges for node: {node_id}")
            return edges

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get edges for node {node_id}: {error_msg}")
            raise Exception(f"Failed to get edges: {error_msg}") from e

    async def get_disconnected_nodes(self) -> list[str]:
        """
        Find and return nodes that are not connected to any other nodes in the graph.

        Returns:
        --------

            - list[str]: A list of IDs of disconnected nodes.
        """
        query = f"""
            MATCH(n :{self._GRAPH_NODE_LABEL})
            WHERE NOT (n)--()
            RETURN COLLECT(ID(n)) as ids
        """

        results = await self.query(query)
        return results[0]["ids"] if len(results) > 0 else []

    async def get_predecessors(self, node_id: str, edge_label: str = "") -> list[str]:
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

        edge_label = f" :{edge_label}" if edge_label is not None else ""
        query = f"""
        MATCH (node)<-[r{edge_label}]-(predecessor)
        WHERE node.id = $node_id
        RETURN predecessor
        """

        results = await self.query(query, {"node_id": node_id})

        return [result["predecessor"] for result in results]

    async def get_successors(self, node_id: str, edge_label: str = "") -> list[str]:
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

        edge_label = f" :{edge_label}" if edge_label is not None else ""
        query = f"""
        MATCH (node)-[r {edge_label}]->(successor)
        WHERE node.id = $node_id
        RETURN successor
        """

        results = await self.query(query, {"node_id": node_id})

        return [result["successor"] for result in results]

    async def get_neighbors(self, node_id: str) -> List[NodeData]:
        """
        Get all neighboring nodes connected to the specified node.

        Parameters:
        -----------
            - node_id (str): Unique identifier of the node for which to retrieve neighbors.

        Returns:
        --------
            - List[NodeData]: A list of NodeData objects representing neighboring nodes.
        """
        try:
            # Query to get all neighboring nodes (both incoming and outgoing connections)
            query = f"""
            MATCH (n:{self._GRAPH_NODE_LABEL})-[r]-(neighbor:{self._GRAPH_NODE_LABEL})
            WHERE id(n) = $node_id
            RETURN DISTINCT id(neighbor) AS neighbor_id, properties(neighbor) AS properties
            """

            params = {"node_id": node_id}
            result = await self.query(query, params)

            # Format neighbors as NodeData objects
            neighbors = [
                {"id": neighbor["neighbor_id"], **neighbor["properties"]} for neighbor in result
            ]

            logger.debug(f"Retrieved {len(neighbors)} neighbors for node: {node_id}")
            return neighbors

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get neighbors for node {node_id}: {error_msg}")
            raise Exception(f"Failed to get neighbors: {error_msg}") from e

    async def get_nodeset_subgraph(
        self, node_type: Type[Any], node_name: List[str]
    ) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]]:
        """
        Fetch a subgraph consisting of a specific set of nodes and their relationships.

        Parameters:
        -----------
            - node_type (Type[Any]): The type of nodes to include in the subgraph.
            - node_name (List[str]): A list of names of the nodes to include in the subgraph.

        Returns:
        --------
            - Tuple[List[Tuple[int, dict]], List[Tuple[int, int, str, dict]]]: A tuple containing nodes and edges of the subgraph.
        """
        try:
            # Query to get nodes by name and their connected subgraph
            query = f"""
            UNWIND $names AS wantedName
            MATCH (n:{self._GRAPH_NODE_LABEL})
            WHERE n.name = wantedName AND n.type = $type
            WITH collect(DISTINCT n) AS primary
            UNWIND primary AS p
            OPTIONAL MATCH (p)-[r]-(nbr:{self._GRAPH_NODE_LABEL})
            WITH primary, collect(DISTINCT nbr) AS nbrs, collect(DISTINCT r) AS rels
            WITH primary + nbrs AS nodelist, rels
            UNWIND nodelist AS node
            WITH collect(DISTINCT node) AS nodes, rels
            MATCH (a:{self._GRAPH_NODE_LABEL})-[r]-(b:{self._GRAPH_NODE_LABEL})
            WHERE a IN nodes AND b IN nodes
            WITH nodes, collect(DISTINCT r) AS all_rels
            RETURN
              [n IN nodes | {{
                id: id(n),
                properties: properties(n)
              }}] AS rawNodes,
              [r IN all_rels | {{
                source_id: id(startNode(r)),
                target_id: id(endNode(r)),
                type: type(r),
                properties: properties(r)
              }}] AS rawRels
            """

            params = {"names": node_name, "type": node_type.__name__}

            result = await self.query(query, params)

            if not result:
                logger.debug(f"No subgraph found for node type {node_type} with names {node_name}")
                return ([], [])

            raw_nodes = result[0]["rawNodes"]
            raw_rels = result[0]["rawRels"]

            # Format nodes as (node_id, properties) tuples
            nodes = [(n["id"], n["properties"]) for n in raw_nodes]

            # Format edges as (source_id, target_id, relationship_name, properties) tuples
            edges = [(r["source_id"], r["target_id"], r["type"], r["properties"]) for r in raw_rels]

            logger.debug(
                f"Retrieved subgraph with {len(nodes)} nodes and {len(edges)} edges for type {node_type.__name__}"
            )
            return (nodes, edges)

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get nodeset subgraph for type {node_type}: {error_msg}")
            raise Exception(f"Failed to get nodeset subgraph: {error_msg}") from e

    async def get_connections(self, node_id: UUID) -> list:
        """
        Get all nodes connected to a specified node and their relationship details.

        Parameters:
        -----------
            - node_id (str): Unique identifier of the node for which to retrieve connections.

        Returns:
        --------
            - List[Tuple[NodeData, Dict[str, Any], NodeData]]: A list of tuples containing connected nodes and relationship details.
        """
        try:
            # Query to get all connections (both incoming and outgoing)
            query = f"""
            MATCH (source:{self._GRAPH_NODE_LABEL})-[r]->(target:{self._GRAPH_NODE_LABEL})
            WHERE id(source) = $node_id OR id(target) = $node_id
            RETURN
                id(source) AS source_id,
                properties(source) AS source_props,
                id(target) AS target_id,
                properties(target) AS target_props,
                type(r) AS relationship_name,
                properties(r) AS relationship_props
            """

            params = {"node_id": str(node_id)}
            result = await self.query(query, params)

            connections = []
            for record in result:
                # Return as (source_node, relationship, target_node)
                connections.append(
                    (
                        {"id": record["source_id"], **record["source_props"]},
                        {
                            "relationship_name": record["relationship_name"],
                            **record["relationship_props"],
                        },
                        {"id": record["target_id"], **record["target_props"]},
                    )
                )

            logger.debug(f"Retrieved {len(connections)} connections for node: {node_id}")
            return connections

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get connections for node {node_id}: {error_msg}")
            raise Exception(f"Failed to get connections: {error_msg}") from e

    async def remove_connection_to_predecessors_of(self, node_ids: list[str], edge_label: str):
        """
        Remove connections (edges) to all predecessors of specified nodes based on edge label.

        Parameters:
        -----------

            - node_ids (list[str]): A list of IDs of nodes from which connections are to be
              removed.
            - edge_label (str): The label of the edges to remove.

        """
        query = f"""
        UNWIND $node_ids AS node_id
        MATCH ({{`~id`: node_id}})-[r:{edge_label}]->(predecessor)
        DELETE r;
        """
        params = {"node_ids": node_ids}
        await self.query(query, params)

    async def remove_connection_to_successors_of(self, node_ids: list[str], edge_label: str):
        """
        Remove connections (edges) to all successors of specified nodes based on edge label.

        Parameters:
        -----------

            - node_ids (list[str]): A list of IDs of nodes from which connections are to be
              removed.
            - edge_label (str): The label of the edges to remove.

        """
        query = f"""
        UNWIND $node_ids AS node_id
        MATCH ({{`~id`: node_id}})<-[r:{edge_label}]-(successor)
        DELETE r;
        """
        params = {"node_ids": node_ids}
        await self.query(query, params)

    async def get_node_labels_string(self):
        """
        Fetch all node labels from the database and return them as a formatted string.

        Returns:
        --------

            A formatted string of node labels.

        Raises:
        -------
            ValueError: If no node labels are found in the database.
        """
        node_labels_query = (
            "CALL neptune.graph.pg_schema() YIELD schema RETURN schema.nodeLabels as labels "
        )
        node_labels_result = await self.query(node_labels_query)
        node_labels = node_labels_result[0]["labels"] if node_labels_result else []

        if not node_labels:
            raise ValueError("No node labels found in the database")

        return str(node_labels)

    async def get_relationship_labels_string(self):
        """
        Fetch all relationship types from the database and return them as a formatted string.

        Returns:
        --------

            A formatted string of relationship types.
        """
        relationship_types_query = (
            "CALL neptune.graph.pg_schema() YIELD schema RETURN schema.edgeLabels as relationships "
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

    async def drop_graph(self, graph_name="myGraph"):
        """
        Drop an existing graph from the database based on its name.

        Note: This method is currently a placeholder because GDS (Graph Data Science)
        projection is not supported in Neptune Analytics.

        Parameters:
        -----------

            - graph_name: The name of the graph to drop, defaults to 'myGraph'. (default
              'myGraph')
        """
        pass

    async def graph_exists(self, graph_name="myGraph"):
        """
        Check if a graph with a given name exists in the database.

        Note: This method is currently a placeholder because GDS (Graph Data Science)
        projection is not supported in Neptune Analytics.

        Parameters:
        -----------

            - graph_name: The name of the graph to check for existence, defaults to 'myGraph'.
              (default 'myGraph')

        Returns:
        --------

            True if the graph exists, otherwise False.
        """
        pass

    async def project_entire_graph(self, graph_name="myGraph"):
        """
        Project all node labels and relationship types into an in-memory graph using GDS.

        Note: This method is currently a placeholder because GDS (Graph Data Science)
        projection is not supported in Neptune Anlaytics.
        """
        pass

    async def get_filtered_graph_data(self, attribute_filters: list[dict[str, list]]):
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
        where_clauses_n = []
        where_clauses_m = []
        for attribute, values in attribute_filters[0].items():
            values_str = ", ".join(
                f"'{value}'" if isinstance(value, str) else str(value) for value in values
            )
            where_clauses_n.append(f"n.{attribute} IN [{values_str}]")
            where_clauses_m.append(f"m.{attribute} IN [{values_str}]")

        node_where_clauses_n_str = " AND ".join(where_clauses_n)
        node_where_clauses_m_str = " AND ".join(where_clauses_m)
        edge_where_clause = f"{node_where_clauses_n_str} AND {node_where_clauses_m_str}"

        query_nodes = f"""
           MATCH (n :{self._GRAPH_NODE_LABEL})
           WHERE {node_where_clauses_n_str}
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
           MATCH (n :{self._GRAPH_NODE_LABEL})-[r]->(m :{self._GRAPH_NODE_LABEL})
           WHERE {edge_where_clause}
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
                MATCH (n :{self._GRAPH_NODE_LABEL})
                WHERE size((n)--()) = 1
                AND n.type = $node_type
                RETURN n
                """
        result = await self.query(query, {"node_type": node_type})
        return [record["n"] for record in result] if result else []

    async def get_document_subgraph(self, data_id: str):
        """
        Retrieve a subgraph related to a document identified by its content hash, including
        related entities and chunks.

        Parameters:
        -----------

            - data_id (str): The document_id identifying the document whose subgraph should be
              retrieved.

        Returns:
        --------

            The subgraph data as a dictionary, or None if not found.
        """
        query = f"""

        MATCH (doc)
        WHERE (doc:{self._GRAPH_NODE_LABEL})
        AND doc.type in ['TextDocument', 'PdfDocument']
        AND doc.id = $data_id

        OPTIONAL MATCH (doc)<-[:is_part_of]-(chunk {{type: 'DocumentChunk'}})

        // Alternative to WHERE NOT EXISTS
        OPTIONAL MATCH (chunk)-[:contains]->(entity {{type: 'Entity'}})
        OPTIONAL MATCH (entity)<-[:contains]-(otherChunk {{type: 'DocumentChunk'}})-[:is_part_of]->(otherDoc)
          WHERE otherDoc.type in ['TextDocument', 'PdfDocument']
          AND otherDoc.id <> doc.id
                OPTIONAL MATCH (chunk)<-[:made_from]-(made_node {{type: 'TextSummary'}})

        OPTIONAL MATCH (chunk)<-[:made_from]-(made_node {{type: 'TextSummary'}})

        // Alternative to WHERE NOT EXISTS
        OPTIONAL MATCH (entity)-[:is_a]->(type {{type: 'EntityType'}})
        OPTIONAL MATCH (type)<-[:is_a]-(otherEntity {{type: 'Entity'}})<-[:contains]-(otherChunk {{type: 'DocumentChunk'}})-[:is_part_of]->(otherDoc)
          WHERE otherDoc.type in ['TextDocument', 'PdfDocument']
          AND otherDoc.id <> doc.id

        // Alternative to WHERE NOT EXISTS
        WITH doc, entity, chunk, made_node, type, otherDoc
        WHERE otherDoc IS NULL

        RETURN
            collect(DISTINCT doc) as document,
            collect(DISTINCT chunk) as chunks,
            collect(DISTINCT entity) as orphan_entities,
            collect(DISTINCT made_node) as made_from_nodes,
            collect(DISTINCT type) as orphan_types
        """
        result = await self.query(query, {"data_id": data_id})
        return result[0] if result else None

    async def _get_model_independent_graph_data(self):
        """
        Retrieve the basic graph data without considering the model specifics, returning nodes
        and edges.

        Returns:
        --------

            A tuple of nodes and edges data.
        """
        query_string = f"""
            MATCH (n :{self._GRAPH_NODE_LABEL})
            WITH count(n) AS nodeCount
            MATCH (a :{self._GRAPH_NODE_LABEL})-[r]->(b :{self._GRAPH_NODE_LABEL})
            RETURN nodeCount AS numVertices, count(r) AS numEdges
        """
        query_response = await self.query(query_string)
        num_nodes = query_response[0].get("numVertices")
        num_edges = query_response[0].get("numEdges")

        return (num_nodes, num_edges)

    async def _get_connected_components_stat(self):
        """
        Retrieve statistics about connected components in the graph.

        This method analyzes the graph to find all connected components
        and returns both the sizes of each component and the total number of components.


        Returns:
        --------
            tuple[list[int], int]
            A tuple containing:
              - A list of sizes for each connected component (descending order).
              - The total number of connected components.
            Returns ([], 0) if no connected components are found.
        """
        query = f"""
        MATCH(n :{self._GRAPH_NODE_LABEL})
        CALL neptune.algo.wcc(n,{{}})
        YIELD node, component
        RETURN component, count(*) AS size
        ORDER BY size DESC
        """

        result = await self.query(query)
        size_connected_components = [record["size"] for record in result] if result else []
        num_connected_components = len(result)

        return (size_connected_components, num_connected_components)

    async def _count_self_loops(self):
        """
        Count the number of self-loop relationships in the Neptune Anlaytics graph backend.

        This function executes a OpenCypher query to find and count all edge relationships that
        begin and end at the same node (self-loops). It returns the count of such relationships
        or 0 if no results are found.

        Returns:
        --------

            The count of self-loop relationships found in the database, or 0 if none were found.
        """
        query = f"""
        MATCH (n :{self._GRAPH_NODE_LABEL})-[r]->(n :{self._GRAPH_NODE_LABEL})
        RETURN count(r) AS adapter_loop_count;
        """
        result = await self.query(query)
        return result[0]["adapter_loop_count"] if result else 0

    @staticmethod
    def _convert_relationship_to_edge(relationship: dict) -> EdgeData:
        return (
            relationship["source_id"],
            relationship["target_id"],
            relationship["relationship_name"],
            relationship["properties"],
        )
