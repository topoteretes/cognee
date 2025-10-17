import inspect
from functools import wraps
from abc import abstractmethod, ABC
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple, Type, Union
from uuid import NAMESPACE_OID, UUID, uuid5
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.models.graph_relationship_ledger import GraphRelationshipLedger
from cognee.infrastructure.databases.relational.get_relational_engine import get_relational_engine

logger = get_logger()

# Type aliases for better readability
NodeData = Dict[str, Any]
EdgeData = Tuple[
    str, str, str, Dict[str, Any]
]  # (source_id, target_id, relationship_name, properties)
Node = Tuple[str, NodeData]  # (node_id, properties)


def record_graph_changes(func):
    """
    Decorator to record graph changes in the relationship database.

    Parameters:
    -----------

        - func: The asynchronous function to wrap, which likely modifies graph data.

    Returns:
    --------

        Returns the wrapped function that manages database relationships.
    """

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        """
        Wraps the given asynchronous function to handle database relationships.

        Tracks the caller's function and class name for context. When the wrapped function is
        called, it manages database relationships for nodes or edges by adding entries to a
        ledger and committing the changes to the database session. Errors during relationship
        addition or session commit are logged and will not disrupt the execution of the wrapped
        function.

        Parameters:
        -----------

            - *args: Positional arguments passed to the wrapped function.
            - **kwargs: Keyword arguments passed to the wrapped function.

        Returns:
        --------

            Returns the result of the wrapped function call.
        """
        db_engine = get_relational_engine()
        frame = inspect.currentframe()
        while frame:
            if frame.f_back and frame.f_back.f_code.co_name != "wrapper":
                caller_frame = frame.f_back
                break
            frame = frame.f_back

        caller_name = caller_frame.f_code.co_name
        caller_class = (
            caller_frame.f_locals.get("self", None).__class__.__name__
            if caller_frame.f_locals.get("self", None)
            else None
        )
        creator = f"{caller_class}.{caller_name}" if caller_class else caller_name

        result = await func(self, *args, **kwargs)

        async with db_engine.get_async_session() as session:
            if func.__name__ == "add_nodes":
                nodes: List[DataPoint] = args[0]

                relationship_ledgers = []

                for node in nodes:
                    node_id = UUID(str(node.id))
                    relationship_ledgers.append(
                        GraphRelationshipLedger(
                            id=uuid5(NAMESPACE_OID, f"{datetime.now(timezone.utc).timestamp()}"),
                            source_node_id=node_id,
                            destination_node_id=node_id,
                            creator_function=f"{creator}.node",
                            node_label=getattr(node, "name", None) or str(node.id),
                        )
                    )

                try:
                    session.add_all(relationship_ledgers)
                    await session.flush()
                except Exception as e:
                    logger.debug(f"Error adding relationship: {e}")
                    await session.rollback()

            elif func.__name__ == "add_edges":
                edges = args[0]

                relationship_ledgers = []

                for edge in edges:
                    source_id = UUID(str(edge[0]))
                    target_id = UUID(str(edge[1]))
                    rel_type = str(edge[2])
                    relationship_ledgers.append(
                        GraphRelationshipLedger(
                            id=uuid5(NAMESPACE_OID, f"{datetime.now(timezone.utc).timestamp()}"),
                            source_node_id=source_id,
                            destination_node_id=target_id,
                            creator_function=f"{creator}.{rel_type}",
                        )
                    )

                try:
                    session.add_all(relationship_ledgers)
                    await session.flush()
                except Exception as e:
                    logger.debug(f"Error adding relationship: {e}")
                    await session.rollback()

            try:
                await session.commit()
            except Exception as e:
                logger.debug(f"Error committing session: {e}")

        return result

    return wrapper


class GraphDBInterface(ABC):
    """
    Define an interface for graph database operations to be implemented by concrete classes.

    Public methods include:
    - query
    - add_node
    - add_nodes
    - delete_node
    - delete_nodes
    - get_node
    - get_nodes
    - add_edge
    - add_edges
    - delete_graph
    - get_graph_data
    - get_graph_metrics
    - has_edge
    - has_edges
    - get_edges
    - get_neighbors
    - get_nodeset_subgraph
    - get_connections
    """

    @abstractmethod
    async def is_empty(self) -> bool:
        logger.warning("is_empty() is not implemented")
        return True

    @abstractmethod
    async def query(self, query: str, params: dict) -> List[Any]:
        """
        Execute a raw database query and return the results.

        Parameters:
        -----------

            - query (str): The query string to execute against the database.
            - params (dict): A dictionary of parameters to be used in the query.
        """
        raise NotImplementedError

    @abstractmethod
    async def add_node(
        self, node: Union[DataPoint, str], properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a single node with specified properties to the graph.

        Parameters:
        -----------

            - node (Union[DataPoint, str]): Either a DataPoint object or a string identifier for the node being added.
            - properties (Optional[Dict[str, Any]]): A dictionary of properties associated with the node.
              Required when node is a string, ignored when node is a DataPoint.
        """
        raise NotImplementedError

    @abstractmethod
    @record_graph_changes
    async def add_nodes(self, nodes: Union[List[Node], List[DataPoint]]) -> None:
        """
        Add multiple nodes to the graph in a single operation.

        Parameters:
        -----------

            - nodes (Union[List[Node], List[DataPoint]]): A list of Node objects or DataPoint objects to be added to the graph.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """
        Delete a specified node from the graph by its ID.

        Parameters:
        -----------

            - node_id (str): Unique identifier for the node to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_nodes(self, node_ids: List[str]) -> None:
        """
        Delete multiple nodes from the graph by their identifiers.

        Parameters:
        -----------

            - node_ids (List[str]): A list of unique identifiers for the nodes to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[NodeData]:
        """
        Retrieve a single node from the graph using its ID.

        Parameters:
        -----------

            - node_id (str): Unique identifier of the node to retrieve.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_nodes(self, node_ids: List[str]) -> List[NodeData]:
        """
        Retrieve multiple nodes from the graph using their IDs.

        Parameters:
        -----------

            - node_ids (List[str]): A list of unique identifiers for the nodes to retrieve.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    @record_graph_changes
    async def add_edges(
        self, edges: Union[List[EdgeData], List[Tuple[str, str, str, Optional[Dict[str, Any]]]]]
    ) -> None:
        """
        Add multiple edges to the graph in a single operation.

        Parameters:
        -----------

            - edges (Union[List[EdgeData], List[Tuple[str, str, str, Optional[Dict[str, Any]]]]]): A list of EdgeData objects or tuples representing edges to be added.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_graph(self) -> None:
        """
        Remove the entire graph, including all nodes and edges.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_graph_data(self) -> Tuple[List[Node], List[EdgeData]]:
        """
        Retrieve all nodes and edges within the graph.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_graph_metrics(self, include_optional: bool = False) -> Dict[str, Any]:
        """
        Fetch metrics and statistics of the graph, possibly including optional details.

        Parameters:
        -----------

            - include_optional (bool): Flag indicating whether to include optional metrics or
              not. (default False)
        """
        raise NotImplementedError

    @abstractmethod
    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """
        Verify if an edge exists between two specified nodes.

        Parameters:
        -----------

            - source_id (str): Unique identifier of the source node.
            - target_id (str): Unique identifier of the target node.
            - relationship_name (str): Name of the relationship to verify.
        """
        raise NotImplementedError

    @abstractmethod
    async def has_edges(self, edges: List[EdgeData]) -> List[EdgeData]:
        """
        Determine the existence of multiple edges in the graph.

        Parameters:
        -----------

            - edges (List[EdgeData]): A list of EdgeData objects to check for existence in the
              graph.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_edges(self, node_id: str) -> List[EdgeData]:
        """
        Retrieve all edges that are connected to the specified node.

        Parameters:
        -----------

            - node_id (str): Unique identifier of the node whose edges are to be retrieved.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_neighbors(self, node_id: str) -> List[NodeData]:
        """
        Get all neighboring nodes connected to the specified node.

        Parameters:
        -----------

            - node_id (str): Unique identifier of the node for which to retrieve neighbors.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    async def get_connections(
        self, node_id: Union[str, UUID]
    ) -> List[Tuple[NodeData, Dict[str, Any], NodeData]]:
        """
        Get all nodes connected to a specified node and their relationship details.

        Parameters:
        -----------

            - node_id (Union[str, UUID]): Unique identifier of the node for which to retrieve connections.
        """
        raise NotImplementedError
