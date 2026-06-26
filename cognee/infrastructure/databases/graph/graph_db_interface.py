from uuid import UUID
from abc import abstractmethod, ABC
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Type, Union
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.provenance import (
    EdgeDeleteData,
    EdgeIdentity,
    NodeDeleteData,
)

logger = get_logger()

# Type aliases for better readability
NodeData = Dict[str, Any]
EdgeData = Tuple[
    str, str, str, Dict[str, Any]
]  # (source_id, target_id, relationship_name, properties)
Node = Tuple[str, NodeData]  # (node_id, properties)


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
        """Return True when the graph contains no nodes."""
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
    async def add_nodes(
        self,
        nodes: Union[List[Node], List[DataPoint]],
        source_ref_key: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> None:
        """
        Add multiple nodes to the graph in a single operation.

        Parameters:
        -----------

            - nodes (Union[List[Node], List[DataPoint]]): A list of Node objects or DataPoint objects to be added to the graph.
            - source_ref_key (Optional[str]): Graph provenance source ref to stamp
              atomically as part of this write. Backends that support graph-provenance
              provenance fold it into the same statement; others may ignore it. (default None)
            - pipeline_run_id (Optional[str]): Run id recorded with the provenance stamp so
              the write is rollbackable by run. Ignored when source_ref_key is None. (default None)
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

    async def remove_belongs_to_set_tags(
        self,
        tags: List[str],
        node_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Remove the given tag names from every node's `belongs_to_set` property
        array. Keeps the property consistent with the additive
        `belongs_to_set` edges after a NodeSet or its containing dataset is
        deleted.

        When `node_ids` is provided, the detag only applies to nodes whose
        id appears in the list — used to reconcile shared nodes that lose
        membership in one dataset without disturbing unrelated nodes that
        legitimately still carry the tag.

        Default no-op; only Neo4j overrides this today. Other
        list-property-storing adapters are free to implement it later.
        """
        return None

    async def attach_node_source_refs(
        self,
        node_ids: list[str],
        source_ref_keys: list[str],
        pipeline_run_id: str | None = None,
    ) -> None:
        """
        Attach source refs to existing graph nodes.

        Implementations append the supplied source_ref_keys, derive and append
        source_dataset_ids from those refs, and when pipeline_run_id is
        provided, append source_run_ids and source_run_refs as rollback indexes.

        Parameters:
        -----------

            - node_ids (list[str]): Unique identifiers of the nodes to update.
            - source_ref_keys (list[str]): Source refs to append to each node.
            - pipeline_run_id (str | None): Pipeline run that attached the refs.

        Default implementation raises UnsupportedProvenanceCapability.
        """
        raise UnsupportedProvenanceCapability()

    async def attach_edge_source_refs(
        self,
        edges: list[EdgeIdentity],
        source_ref_keys: list[str],
        pipeline_run_id: str | None = None,
    ) -> None:
        """
        Attach source refs to existing graph edges.

        Implementations append the supplied source_ref_keys, derive and append
        source_dataset_ids from those refs, and when pipeline_run_id is
        provided, append source_run_ids and source_run_refs as rollback indexes.

        Parameters:
        -----------

            - edges (list[EdgeIdentity]): Edge identities to update.
            - source_ref_keys (list[str]): Source refs to append to each edge.
            - pipeline_run_id (str | None): Pipeline run that attached the refs.

        Default implementation raises UnsupportedProvenanceCapability.
        """
        raise UnsupportedProvenanceCapability()

    async def remove_node_source_refs(
        self,
        node_ids: list[str],
        source_ref_keys: list[str],
    ) -> None:
        """
        Remove source refs from graph nodes.

        Implementations also keep source_dataset_ids, source_run_ids, and
        source_run_refs consistent with the remaining source_ref_keys.

        Parameters:
        -----------

            - node_ids (list[str]): Unique identifiers of the nodes to update.
            - source_ref_keys (list[str]): Source refs to remove from each node.

        Default implementation raises UnsupportedProvenanceCapability.
        """
        raise UnsupportedProvenanceCapability()

    async def remove_edge_source_refs(
        self,
        edges: list[EdgeIdentity],
        source_ref_keys: list[str],
    ) -> None:
        """
        Remove source refs from graph edges.

        Implementations also keep source_dataset_ids, source_run_ids, and
        source_run_refs consistent with the remaining source_ref_keys.

        Parameters:
        -----------

            - edges (list[EdgeIdentity]): Edge identities to update.
            - source_ref_keys (list[str]): Source refs to remove from each edge.

        Default implementation raises UnsupportedProvenanceCapability.
        """
        raise UnsupportedProvenanceCapability()

    async def delete_edge_triples(
        self,
        edges: list[EdgeIdentity],
    ) -> None:
        """
        Delete graph edges by source id, target id, and relationship name.

        Parameters:
        -----------

            - edges (list[EdgeIdentity]): Edge identities to delete.

        Default implementation raises UnsupportedProvenanceCapability.
        """
        raise UnsupportedProvenanceCapability()

    async def get_node_delete_data(
        self,
        node_ids: list[str],
    ) -> dict[str, NodeDeleteData]:
        """
        Return node properties needed by graph-provenance delete and vector cleanup.

        Returned data includes node identity, indexed fields, node properties,
        and all four provenance fields.

        Parameters:
        -----------

            - node_ids (list[str]): Unique identifiers of the nodes to inspect.

        Returns:
        --------

            - dict[str, NodeDeleteData]: Delete data keyed by node id.
        """
        raise UnsupportedProvenanceCapability()

    async def get_edge_delete_data(
        self,
        edges: list[EdgeIdentity],
    ) -> dict[EdgeIdentity, EdgeDeleteData]:
        """
        Return edge properties needed by graph-provenance delete and rollback.

        Returned data includes edge identity, edge text, edge properties, and
        all four provenance fields.

        Parameters:
        -----------

            - edges (list[EdgeIdentity]): Edge identities to inspect.

        Returns:
        --------

            - dict[EdgeIdentity, EdgeDeleteData]: Delete data keyed by edge identity.
        """
        raise UnsupportedProvenanceCapability()

    async def find_nodes_by_source_ref(
        self,
        source_ref_key: str,
    ) -> list[str]:
        """
        Find graph node ids that currently contain a source ref.

        Parameters:
        -----------

            - source_ref_key (str): Source ref key to match.
        """
        raise UnsupportedProvenanceCapability()

    async def find_edges_by_source_ref(
        self,
        source_ref_key: str,
    ) -> list[EdgeIdentity]:
        """
        Find graph edges that currently contain a source ref.

        Parameters:
        -----------

            - source_ref_key (str): Source ref key to match.
        """
        raise UnsupportedProvenanceCapability()

    async def find_node_source_refs_by_dataset(
        self,
        dataset_id: str,
    ) -> dict[str, list[str]]:
        """
        Find node source refs owned by a dataset.

        Parameters:
        -----------

            - dataset_id (str): Dataset id to match.

        Returns:
        --------

            - dict[str, list[str]]: Matching source refs keyed by node id.
        """
        raise UnsupportedProvenanceCapability()

    async def find_edge_source_refs_by_dataset(
        self,
        dataset_id: str,
    ) -> dict[EdgeIdentity, list[str]]:
        """
        Find edge source refs owned by a dataset.

        Parameters:
        -----------

            - dataset_id (str): Dataset id to match.

        Returns:
        --------

            - dict[EdgeIdentity, list[str]]: Matching source refs keyed by edge identity.
        """
        raise UnsupportedProvenanceCapability()

    async def find_node_source_refs_by_pipeline_run(
        self,
        pipeline_run_id: str,
    ) -> dict[str, list[str]]:
        """
        Find node source refs attached by a pipeline run.

        Parameters:
        -----------

            - pipeline_run_id (str): Pipeline run id to match.

        Returns:
        --------

            - dict[str, list[str]]: Matching source refs keyed by node id.
        """
        raise UnsupportedProvenanceCapability()

    async def find_edge_source_refs_by_pipeline_run(
        self,
        pipeline_run_id: str,
    ) -> dict[EdgeIdentity, list[str]]:
        """
        Find edge source refs attached by a pipeline run.

        Parameters:
        -----------

            - pipeline_run_id (str): Pipeline run id to match.

        Returns:
        --------

            - dict[EdgeIdentity, list[str]]: Matching source refs keyed by edge identity.
        """
        raise UnsupportedProvenanceCapability()

    async def set_graph_metadata(
        self,
        metadata: dict[str, str],
    ) -> None:
        """
        Store graph-level metadata used to identify provenance schema support.

        Parameters:
        -----------

            - metadata (dict[str, str]): Metadata keys and values to persist.
        """
        raise UnsupportedProvenanceCapability()

    async def get_graph_metadata(self) -> dict[str, str]:
        """Return graph-level metadata used to identify provenance schema support."""
        raise UnsupportedProvenanceCapability()

    async def get_edges_created_since(
        self,
        since: Optional[datetime],
        limit: int,
    ) -> Tuple[List[Tuple[str, str, str, datetime]], Dict[str, Dict[str, Any]]]:
        """Return edges created after ``since`` (oldest first), with endpoint nodes.

        Graph-provenance equivalent of the relational incremental edge fetch used to
        sync recent graph knowledge into the session cache. On graphs that store
        provenance in the graph itself the relational Edge/Node tables are empty,
        so callers read new edges here instead.

        Parameters:
        -----------

            - since (Optional[datetime]): Return only edges created strictly after
              this timestamp; None returns from the beginning.
            - limit (int): Maximum number of edges to return.

        Returns:
        --------

            - Tuple of (edges, node_map):
                - edges: list of (source_id, target_id, relationship_name, created_at),
                  ordered by created_at ascending.
                - node_map: {node_id: properties} for every endpoint node.
        """
        raise UnsupportedProvenanceCapability()

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
    async def add_edges(
        self,
        edges: Union[List[EdgeData], List[Tuple[str, str, str, Optional[Dict[str, Any]]]]],
        source_ref_key: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> None:
        """
        Add multiple edges to the graph in a single operation.

        Parameters:
        -----------

            - edges (Union[List[EdgeData], List[Tuple[str, str, str, Optional[Dict[str, Any]]]]]): A list of EdgeData objects or tuples representing edges to be added.
            - source_ref_key (Optional[str]): Graph provenance source ref to stamp
              atomically as part of this write. Backends that support graph-provenance
              provenance fold it into the same statement; others may ignore it. (default None)
            - pipeline_run_id (Optional[str]): Run id recorded with the provenance stamp so
              the write is rollbackable by run. Ignored when source_ref_key is None. (default None)
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
        self, node_type: Type[Any], node_name: List[str], node_name_filter_operator: str = "OR"
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

    @abstractmethod
    async def get_neighborhood(
        self,
        node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[List[Node], List[EdgeData]]:
        """
        Get the k-hop neighborhood subgraph around a set of seed nodes.

        Returns all nodes and edges within `depth` hops of any seed node,
        in the same format as get_graph_data().
        Optional edge_type filtering to constrain traversal paths.

        Parameters:
        -----------

            - node_ids (List[str]): Seed node identifiers to start traversal from.
            - depth (int): Number of hops to traverse from each seed node. (default 1)
            - edge_types (Optional[List[str]]): If provided, only traverse edges of these
              relationship types. (default None)
        """
        raise NotImplementedError

    @abstractmethod
    async def get_filtered_graph_data(
        self, attribute_filters: List[Dict[str, List[Union[str, int]]]]
    ) -> Tuple[List[Node], List[EdgeData]]:
        """
        Retrieve nodes and edges filtered by the provided attribute criteria.

        Parameters:
        -----------

            - attribute_filters: A list of dictionaries where keys are attribute names and values
              are lists of attribute values to filter by.
        """
        raise NotImplementedError

    async def get_node_feedback_weights(self, node_ids: List[str]) -> Dict[str, float]:
        """
        Retrieve node feedback weights for multiple node ids.
        Returns only found node ids.
        """
        raise NotImplementedError("get_node_feedback_weights is not implemented for this adapter")

    async def set_node_feedback_weights(
        self, node_feedback_weights: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Persist node feedback weights for multiple node ids.
        Returns per-id update success.
        """
        raise NotImplementedError("set_node_feedback_weights is not implemented for this adapter")

    async def get_node_truth_state(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve node truth alignment state for multiple node ids.
        Returns only found node ids.
        """
        raise NotImplementedError("get_node_truth_state is not implemented for this adapter")

    async def set_node_truth_state(
        self, node_truth_state: Dict[str, Dict[str, Any]]
    ) -> Dict[str, bool]:
        """
        Persist node truth alignment state for multiple node ids.
        Returns per-id update success.
        """
        raise NotImplementedError("set_node_truth_state is not implemented for this adapter")

    async def get_edge_feedback_weights(self, edge_object_ids: List[str]) -> Dict[str, float]:
        """
        Retrieve edge feedback weights for multiple edge_object_ids.
        Returns only found edge ids.
        """
        raise NotImplementedError("get_edge_feedback_weights is not implemented for this adapter")

    async def set_edge_feedback_weights(
        self, edge_feedback_weights: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Persist edge feedback weights for multiple edge_object_ids.
        Returns per-id update success.
        """
        raise NotImplementedError("set_edge_feedback_weights is not implemented for this adapter")

    async def get_triplets_batch(self, offset: int, limit: int) -> List[Dict[str, Any]]:
        """Retrieve a batch of triplets (source, edge, target).

        Optional extension — implemented by PostgresAdapter, Neo4jAdapter,
        and LadybugAdapter but not NeptuneGraphDB.

        Parameters
        ----------

            - offset: Number of triplets to skip.
            - limit: Maximum number of triplets to return.
        """
        raise NotImplementedError("get_triplets_batch is not implemented for this adapter")

    async def get_node_frequency_weights(self, node_ids: List[str]) -> Dict[str, float]:
        """
        Retrieve node frequency weights for multiple node ids.
        Returns only found node ids.
        """
        raise NotImplementedError("get_node_frequency_weights is not implemented for this adapter")

    async def set_node_frequency_weights(
        self, node_frequency_weights: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Persist node frequency weights for multiple node ids.
        Returns per-id update success.
        """
        raise NotImplementedError("set_node_frequency_weights is not implemented for this adapter")

    async def get_edge_frequency_weights(self, edge_object_ids: List[str]) -> Dict[str, float]:
        """
        Retrieve edge frequency weights for multiple edge_object_ids.
        Returns only found edge ids.
        """
        raise NotImplementedError("get_edge_frequency_weights is not implemented for this adapter")

    async def set_edge_frequency_weights(
        self, edge_frequency_weights: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Persist edge frequency weights for multiple edge_object_ids.
        Returns per-id update success.
        """
        raise NotImplementedError("set_edge_frequency_weights is not implemented for this adapter")
