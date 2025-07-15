"""Neptune Analytics Adapter for Graph Database"""

import json
from typing import Optional, Any, List, Dict, Type, Tuple
from uuid import UUID
from cognee.shared.logging_utils import get_logger, ERROR
from cognee.infrastructure.databases.graph.graph_db_interface import (
    GraphDBInterface,
    record_graph_changes,
    NodeData,
    EdgeData,
    Node,
)
from cognee.modules.storage.utils import JSONEncoder
from cognee.infrastructure.engine import DataPoint

from .exceptions import (
    NeptuneAnalyticsConfigurationError,
)
from .neptune_analytics_utils import (
    validate_graph_id,
    validate_aws_region,
    build_neptune_config,
    format_neptune_error,
)

logger = get_logger("NeptuneAnalyticsAdapter")

try:
    from langchain_aws import NeptuneAnalyticsGraph
    LANGCHAIN_AWS_AVAILABLE = True
except ImportError:
    logger.warning("langchain_aws not available. Neptune Analytics functionality will be limited.")
    LANGCHAIN_AWS_AVAILABLE = False

class NeptuneAnalyticsAdapter(GraphDBInterface):
    """
    Adapter for interacting with Amazon Neptune Analytics graph store.
    This class provides methods for querying, adding, deleting nodes and edges using the aws_langchain library.
    """
    _GRAPH_NODE_LABEL = "COGNEE_GRAPH_NODE"

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
            raise ImportError("langchain_aws is not available. Please install it to use Neptune Analytics.")

        # Validate configuration
        if not validate_graph_id(graph_id):
            raise NeptuneAnalyticsConfigurationError(f"Invalid graph ID: \"{graph_id}\"")
        
        if region and not validate_aws_region(region):
            raise NeptuneAnalyticsConfigurationError(f"Invalid AWS region: \"{region}\"")
        
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
        logger.info(f"Initialized Neptune Analytics adapter for graph: \"{graph_id}\" in region: \"{self.region}\"")

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
            raise NeptuneAnalyticsConfigurationError(f"Failed to initialize Neptune Analytics client: {format_neptune_error(e)}")

    @staticmethod
    def serialize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
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

            if isinstance(property_value, dict):
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
            logger.info(f"executing na query:\nquery={query}\n")
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
            raise Exception(f"Query execution failed: {error_msg}")

    async def add_node(self, node: DataPoint) -> None:
        """
        Add a single node with specified properties to the graph.

        Parameters:
        -----------
            - node_id (str): Unique identifier for the node being added.
            - properties (Dict[str, Any]): A dictionary of properties associated with the node.
        """
        try:
            # Prepare node properties with the ID and graph type
            serialized_properties = self.serialize_properties(node.model_dump())

            query = f"""
            MERGE (n:{self._GRAPH_NODE_LABEL} {{`~id`: $node_id}})
            ON CREATE SET n = $properties, n.updated_at = timestamp()
            ON MATCH SET n = $properties, n.updated_at = timestamp()
            RETURN n
            """

            params = {
                "node_id": str(node.id),
                "properties": serialized_properties,
            }
            
            result = await self.query(query, params)
            logger.debug(f"Successfully added/updated node: {node.id}")
            
        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to add node {node.id}: {error_msg}")
            raise Exception(f"Failed to add node: {error_msg}")

    @record_graph_changes
    async def add_nodes(self, nodes: List[DataPoint]) -> None:
        """
        Add multiple nodes to the graph in a single operation.

        Parameters:
        -----------
            - nodes (List[DataPoint]): A list of DataPoint objects to be added to the graph.
        """
        if not self._client:
            logger.error("Neptune Analytics client not initialized")
            return

        if not nodes:
            logger.debug("No nodes to add")
            return

        try:
            # Build bulk node creation query using UNWIND
            query = f"""
            UNWIND $nodes AS node
            MERGE (n:{self._GRAPH_NODE_LABEL} {{`~id`: node.node_id}})
            ON CREATE SET n = node.properties, n.updated_at = timestamp()
            ON MATCH SET n = node.properties, n.updated_at = timestamp()
            RETURN count(n) AS nodes_processed
            """

            # Prepare node data for bulk operation
            params = {
                "nodes": [
                    {
                        "node_id": str(node.id),
                        "properties": self.serialize_properties(node.model_dump()),
                    }
                    for node in nodes
                ]
            }
            result = await self.query(query, params)

            processed_count = result[0].get('nodes_processed', 0) if result else 0
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
                    logger.error(f"Failed to add individual node {node.id}: {format_neptune_error(node_error)}")
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

            params = {
                "node_id": node_id
            }
            
            await self.query(query, params)
            logger.debug(f"Successfully deleted node: {node_id}")
            
        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to delete node {node_id}: {error_msg}")
            raise Exception(f"Failed to delete node: {error_msg}")

    async def delete_nodes(self, node_ids: List[str]) -> None:
        """
        Delete multiple nodes from the graph by their identifiers.

        Parameters:
        -----------
            - node_ids (List[str]): A list of unique identifiers for the nodes to delete.
        """
        if not self._client:
            logger.error("Neptune Analytics client not initialized")
            return

        if not node_ids:
            logger.debug("No nodes to delete")
            return

        try:
            # Build bulk node deletion query using UNWIND
            query = """
            UNWIND $node_ids AS node_id
            MATCH (n)
            WHERE id(n) = node_id
            DETACH DELETE n
            """

            params = {"node_ids": node_ids}
            result = await self.query(query, params)
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
                    logger.error(f"Failed to delete individual node {node_id}: {format_neptune_error(node_error)}")
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
            params = {'node_id': node_id}

            result = await self.query(query, params)
            
            if result and len(result) == 1:
                # Extract node properties from the result
                node_data = result.pop().get('n', {})
                logger.debug(f"Successfully retrieved node: {node_id}")
                return node_data
            else:
                if not result:
                    logger.debug(f"Node not found: {node_id}")
                elif len(result) > 1:
                    logger.debug(f"Only one node expected, multiple returned: {node_id}")
                return None
                
        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get node {node_id}: {error_msg}")
            raise Exception(f"Failed to get node: {error_msg}")

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
        if not self._client:
            logger.error("Neptune Analytics client not initialized")
            return []

        if not node_ids:
            logger.debug("No node IDs provided")
            return []

        try:
            # Build bulk node-retrieval OpenCypher query using UNWIND
            query = """
            UNWIND $node_ids AS node_id
            MATCH (n)
            WHERE id(n) = node_id
            RETURN n
            """

            params = {"node_ids": node_ids}
            result = await self.query(query, params)

            # Extract node data from results
            nodes = []
            if result:
                for record in result:
                    node_data = record.get('n', {})
                    if node_data:
                        nodes.append(node_data)

            logger.debug(f"Successfully retrieved {len(nodes)} nodes out of {len(node_ids)} requested")
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
                    logger.error(f"Failed to get individual node {node_id}: {format_neptune_error(node_error)}")
                    continue
            return nodes

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
            serialized_properties = self.serialize_properties(edge_props)

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
            logger.debug(f"Successfully added edge: {source_id} -[{relationship_name}]-> {target_id}")
            
        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to add edge {source_id} -> {target_id}: {error_msg}")
            raise Exception(f"Failed to add edge: {error_msg}")

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
        for relationship_name, edges_by_relationship in edges_by_relationship.items():
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
                params = {"edges":
                    [
                        {
                            "from_node": str(edge[0]),
                            "to_node": str(edge[1]),
                            "relationship_name": relationship_name,
                            "properties": self.serialize_properties(edge[3] if len(edge) > 3 and edge[3] else {}),
                        }
                        for edge in edges_by_relationship
                    ]
                }
                results[relationship_name] = await self.query(query, params)
            except Exception as e:
                logger.error(f"Failed to add edges for relationship {relationship_name}: {format_neptune_error(e)}")
                logger.info("Falling back to individual edge creation")
                for edge in edges_by_relationship:
                    try:
                        source_id, target_id, relationship_name = edge[0], edge[1], edge[2]
                        properties = edge[3] if len(edge) > 3 else {}
                        await self.add_edge(source_id, target_id, relationship_name, properties)
                    except Exception as edge_error:
                        logger.error(f"Failed to add individual edge {edge[0]} -> {edge[1]}: {format_neptune_error(edge_error)}")
                        continue

        processed_count = 0
        for result in results.values():
            processed_count += result[0].get('edges_processed', 0) if result else 0
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
            logger.debug(f"Successfully deleted all edges and nodes from the graph")

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to delete graph: {error_msg}")
            raise Exception(f"Failed to delete graph: {error_msg}")


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
            nodes = [
                (
                    result["node_id"],
                    result["properties"]
                )
                for result in nodes_result
            ]

            # Format edges as (source_id, target_id, relationship_name, properties) tuples
            edges = [
                (
                    result["source_id"],
                    result["target_id"],
                    result["relationship_name"],
                    result["properties"]
                )
                for result in edges_result
            ]

            logger.debug(f"Retrieved {len(nodes)} nodes and {len(edges)} edges from graph")
            return (nodes, edges)

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get graph data: {error_msg}")
            raise Exception(f"Failed to get graph data: {error_msg}")

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
        # TODO: Implement using aws_langchain Neptune Analytics metrics retrieval
        logger.warning("Neptune Analytics get_graph_metrics method not yet implemented")
        return {}

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
            MATCH (source)-[r:{relationship_name}]->(target)
            WHERE id(source) = $source_id AND id(target) = $target_id
            RETURN COUNT(r) > 0 AS edge_exists
            """

            params = {
                "source_id": source_id,
                "target_id": target_id,
            }
            
            result = await self.query(query, params)
            
            if result and len(result) > 0:
                edge_exists = result.pop().get('edge_exists', False)
                logger.debug(f"Edge existence check for "
                             f"{source_id} -[{relationship_name}]-> {target_id}: {edge_exists}")
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
            raise Exception(f"Failed to get edges: {error_msg}")

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
                {
                    "id": neighbor["neighbor_id"],
                    **neighbor["properties"]
                }
                for neighbor in result
            ]

            logger.debug(f"Retrieved {len(neighbors)} neighbors for node: {node_id}")
            return neighbors

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get neighbors for node {node_id}: {error_msg}")
            raise Exception(f"Failed to get neighbors: {error_msg}")

    async def get_nodeset_subgraph(
            self,
            node_type: Type[Any],
            node_name: List[str]
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
            MATCH (a)-[r]-(b)
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

            params = {
                "names": node_name,
                "type": node_type.__name__
            }

            result = await self.query(query, params)

            if not result:
                logger.debug(f"No subgraph found for node type {node_type} with names {node_name}")
                return ([], [])

            raw_nodes = result[0]["rawNodes"]
            raw_rels = result[0]["rawRels"]

            # Format nodes as (node_id, properties) tuples
            nodes = [
                (n["id"], n["properties"])
                for n in raw_nodes
            ]

            # Format edges as (source_id, target_id, relationship_name, properties) tuples
            edges = [
                (r["source_id"], r["target_id"], r["type"], r["properties"])
                for r in raw_rels
            ]

            logger.debug(f"Retrieved subgraph with {len(nodes)} nodes and {len(edges)} edges for type {node_type.__name__}")
            return (nodes, edges)

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get nodeset subgraph for type {node_type}: {error_msg}")
            raise Exception(f"Failed to get nodeset subgraph: {error_msg}")

    async def get_connections(
        self, node_id: str
    ) -> List[Tuple[NodeData, Dict[str, Any], NodeData]]:
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

            params = {"node_id": node_id}
            result = await self.query(query, params)

            connections = []
            for record in result:
                # Return as (source_node, relationship, target_node)
                connections.append(
                    (
                        {
                            "id": record["source_id"],
                            **record["source_props"]
                        },
                        {
                            "relationship_name": record["relationship_name"],
                            **record["relationship_props"]
                        },
                        {
                            "id": record["target_id"],
                            **record["target_props"]
                        }
                    )
                )

            logger.debug(f"Retrieved {len(connections)} connections for node: {node_id}")
            return connections

        except Exception as e:
            error_msg = format_neptune_error(e)
            logger.error(f"Failed to get connections for node {node_id}: {error_msg}")
            raise Exception(f"Failed to get connections: {error_msg}")

    @staticmethod
    def _convert_relationship_to_edge(relationship: dict) -> EdgeData:
        return relationship["source_id"], relationship["target_id"], relationship["relationship_name"], relationship["properties"]
