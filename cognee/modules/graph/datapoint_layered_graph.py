"""
Data models for layered knowledge graphs.

This module provides data models to represent layered knowledge graphs, including nodes, edges,
and layers. These models are designed to be compatible with Cognee's database adapters.
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from datetime import datetime
from pydantic import Field

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import JSONEncoder


class GraphNode(DataPoint):
    """
    Represents a node in a layered knowledge graph.

    Attributes:
        id: Unique identifier for the node
        name: Name of the node
        node_type: Type of the node
        description: Description of the node
        properties: Additional properties of the node
        layer_id: ID of the layer this node belongs to
        metadata: Metadata for the node
    """

    name: str
    node_type: str
    description: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    layer_id: Optional[UUID] = None
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "GraphNode", "index_fields": ["name"]}
    )

    @classmethod
    def create(
        cls,
        name: str,
        node_type: str,
        description: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a new node with a generated UUID.

        Note: This is a convenience method for in-memory operations only.
        For database persistence, use the appropriate adapter class like
        LayeredGraphDBAdapter or NetworkXAdapter.

        Args:
            name: Name of the node
            node_type: Type of the node
            description: Description of the node
            properties: Additional properties of the node

        Returns:
            A new GraphNode instance
        """
        return cls(
            id=uuid.uuid4(),
            name=name,
            node_type=node_type,
            description=description,
            properties=properties or {},
        )


class GraphEdge(DataPoint):
    """
    Represents an edge in a layered knowledge graph.

    Attributes:
        id: Unique identifier for the edge
        source_node_id: ID of the source node
        target_node_id: ID of the target node
        relationship_name: Name of the relationship
        properties: Additional properties of the edge
        layer_id: ID of the layer this edge belongs to
        metadata: Metadata for the edge
    """

    source_node_id: UUID
    target_node_id: UUID
    relationship_name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    layer_id: Optional[UUID] = None
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "GraphEdge", "index_fields": ["relationship_name"]}
    )

    @classmethod
    def create(
        cls,
        source_node_id: UUID,
        target_node_id: UUID,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a new edge with a generated UUID.

        Note: This is a convenience method for in-memory operations only.
        For database persistence, use the appropriate adapter class like
        LayeredGraphDBAdapter or NetworkXAdapter.

        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            relationship_name: Name of the relationship
            properties: Additional properties of the edge

        Returns:
            A new GraphEdge instance
        """
        return cls(
            id=uuid.uuid4(),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_name=relationship_name,
            properties=properties or {},
        )


class GraphLayer(DataPoint):
    """
    Represents a layer in a layered knowledge graph.

    Attributes:
        id: Unique identifier for the layer
        name: Name of the layer
        description: Description of the layer
        layer_type: Type of the layer
        parent_layers: List of parent layer IDs
        properties: Additional properties of the layer
        metadata: Metadata for the layer
    """

    name: str
    description: str
    layer_type: str = "default"
    parent_layers: List[UUID] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "GraphLayer", "index_fields": ["name"]}
    )

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        layer_type: str = "default",
        parent_layers: Optional[List[UUID]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a new layer with a generated UUID.

        Note: This is a convenience method for in-memory operations only.
        For database persistence, use the appropriate adapter class like
        LayeredGraphDBAdapter or NetworkXAdapter.

        Args:
            name: Name of the layer
            description: Description of the layer
            layer_type: Type of the layer
            parent_layers: List of parent layer IDs
            properties: Additional properties of the layer

        Returns:
            A new GraphLayer instance
        """
        return cls(
            id=uuid.uuid4(),
            name=name,
            description=description,
            layer_type=layer_type,
            parent_layers=parent_layers or [],
            properties=properties or {},
        )


class LayeredKnowledgeGraphDP(DataPoint):
    """
    Represents a complete layered knowledge graph.

    A layered knowledge graph is a knowledge graph where nodes and edges
    are organized into layers, which can have hierarchical relationships.

    Attributes:
        id: Unique identifier for the graph
        name: Name of the graph
        description: Description of the graph
        layers: Dictionary mapping layer IDs to Layer objects
        nodes: Dictionary mapping node IDs to Node objects
        edges: Dictionary mapping edge IDs to Edge objects
        node_layer_map: Dictionary mapping node IDs to their layer IDs
        edge_layer_map: Dictionary mapping edge IDs to their layer IDs
        metadata: Metadata for the graph
    """

    name: str
    description: str
    layers: Dict[UUID, GraphLayer] = Field(default_factory=dict)
    nodes: Dict[UUID, GraphNode] = Field(default_factory=dict)
    edges: Dict[UUID, GraphEdge] = Field(default_factory=dict)
    node_layer_map: Dict[UUID, UUID] = Field(default_factory=dict)
    edge_layer_map: Dict[UUID, UUID] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "LayeredKnowledgeGraph", "index_fields": ["name"]}
    )
    _adapter = None

    @classmethod
    def create_empty(cls, name: str, description: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Create a new empty LayeredKnowledgeGraph with a generated UUID.

        Note: This is a convenience method for in-memory operations only.
        For database persistence, use the appropriate adapter class like
        LayeredGraphDBAdapter or NetworkXAdapter.

        Args:
            name: Name of the graph
            description: Description of the graph
            metadata: Additional metadata for the graph

        Returns:
            A new empty LayeredKnowledgeGraphDP instance
        """
        default_metadata = {"type": "LayeredKnowledgeGraph", "index_fields": ["name"]}
        if metadata:
            default_metadata.update(metadata)

        return cls(id=uuid.uuid4(), name=name, description=description, metadata=default_metadata)

    def add_layer(self, layer: GraphLayer) -> None:
        """
        Add a layer to the graph.

        This performs in-memory operations first. If an adapter is set,
        the layer will also be persisted to the database.

        Args:
            layer: The layer to add

        Raises:
            InvalidValueError: If the layer already exists
        """
        if layer.id in self.layers:
            raise InvalidValueError(f"Layer with ID {layer.id} already exists in the graph")

        # Add to in-memory representation
        self.layers[layer.id] = layer

        # If adapter is set, persist to database
        if self._adapter:
            import asyncio

            try:
                # Add the layer node
                asyncio.create_task(self._adapter._graph_db.add_node(layer))

                # Add the CONTAINS_LAYER relationship
                asyncio.create_task(
                    self._adapter._graph_db.add_edge(
                        str(self.id),
                        str(layer.id),
                        "CONTAINS_LAYER",
                        {"graph_id": str(self.id), "layer_id": str(layer.id)},
                    )
                )

                # Add parent layer relationships
                for parent_id in layer.parent_layers:
                    asyncio.create_task(
                        self._adapter._graph_db.add_edge(
                            str(layer.id),
                            str(parent_id),
                            "EXTENDS_LAYER",
                            {"child_layer_id": str(layer.id), "parent_layer_id": str(parent_id)},
                        )
                    )
            except Exception as e:
                # Log error but don't disrupt flow - keep in-memory operation working
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error persisting layer to database: {str(e)}")

    def add_node(self, node: GraphNode, layer_id: UUID) -> None:
        """
        Add a node to the graph and associate it with a layer.

        This performs in-memory operations first. If an adapter is set,
        the node will also be persisted to the database.

        Args:
            node: The node to add
            layer_id: The ID of the layer to associate the node with

        Raises:
            InvalidValueError: If the node already exists or the layer doesn't exist
        """
        # Validate node and layer existence
        if node.id in self.nodes:
            raise InvalidValueError(f"Node with ID {node.id} already exists in the graph")

        if layer_id not in self.layers:
            raise InvalidValueError(f"Layer with ID {layer_id} doesn't exist in the graph")

        # Add node to in-memory collections
        self.nodes[node.id] = node
        self.node_layer_map[node.id] = layer_id

        # Ensure the node knows which layer it belongs to
        node.layer_id = layer_id

        # If adapter is set, persist to database
        if self._adapter:
            import asyncio

            try:
                # Add the node
                asyncio.create_task(self._adapter._graph_db.add_node(node))

                # Add the IN_LAYER relationship
                asyncio.create_task(
                    self._adapter._graph_db.add_edge(
                        str(node.id),
                        str(layer_id),
                        "IN_LAYER",
                        {"node_id": str(node.id), "layer_id": str(layer_id)},
                    )
                )
            except Exception as e:
                # Log error but don't disrupt flow - keep in-memory operation working
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error persisting node to database: {str(e)}")

    def add_edge(self, edge: GraphEdge, layer_id: UUID) -> None:
        """
        Add an edge to the graph and associate it with a layer.

        This performs in-memory operations first. If an adapter is set,
        the edge will also be persisted to the database.

        Args:
            edge: The edge to add
            layer_id: The ID of the layer to associate the edge with

        Raises:
            InvalidValueError: If the edge already exists, the layer doesn't exist,
                               or the source or target nodes don't exist
        """
        # Validate edge, node, and layer existence
        if edge.id in self.edges:
            raise InvalidValueError(f"Edge with ID {edge.id} already exists in the graph")

        if layer_id not in self.layers:
            raise InvalidValueError(f"Layer with ID {layer_id} doesn't exist in the graph")

        if edge.source_node_id not in self.nodes:
            raise InvalidValueError(
                f"Source node with ID {edge.source_node_id} doesn't exist in the graph"
            )

        if edge.target_node_id not in self.nodes:
            raise InvalidValueError(
                f"Target node with ID {edge.target_node_id} doesn't exist in the graph"
            )

        # Add edge to in-memory collections
        self.edges[edge.id] = edge
        self.edge_layer_map[edge.id] = layer_id

        # Ensure the edge knows which layer it belongs to
        edge.layer_id = layer_id

        # If adapter is set, persist to database
        if self._adapter:
            import asyncio

            try:
                # Add the edge node
                asyncio.create_task(self._adapter._graph_db.add_node(edge))

                # Add the relationship between source and target
                asyncio.create_task(
                    self._adapter._graph_db.add_edge(
                        str(edge.source_node_id),
                        str(edge.target_node_id),
                        edge.relationship_name,
                        {"edge_id": str(edge.id), **edge.properties},
                    )
                )

                # Add the IN_LAYER relationship
                asyncio.create_task(
                    self._adapter._graph_db.add_edge(
                        str(edge.id),
                        str(layer_id),
                        "IN_LAYER",
                        {"edge_id": str(edge.id), "layer_id": str(layer_id)},
                    )
                )
            except Exception as e:
                # Log error but don't disrupt flow - keep in-memory operation working
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error persisting edge to database: {str(e)}")

    def get_nodes_in_layer(self, layer_id: UUID) -> List[GraphNode]:
        """
        Get all nodes in a specific layer.

        If an adapter is set, this will attempt to retrieve from the database first.
        Otherwise, it will return the in-memory nodes.

        Args:
            layer_id: The ID of the layer

        Returns:
            List of nodes in the layer
        """
        # If adapter is set, try to get from database
        if self._adapter:
            import asyncio

            try:
                # Query nodes in the layer using the adapter
                # This is a simplified example - actual implementation may vary
                # based on your database schema and query capabilities
                query = f"""
                MATCH (n:GraphNode)-[:IN_LAYER]->(l:GraphLayer)
                WHERE l.id = '{str(layer_id)}'
                RETURN n
                """

                # Execute query and parse results
                future = asyncio.ensure_future(self._adapter._graph_db.query(query))
                # This is a blocking operation in a primarily non-blocking context
                # Consider implementing a fully async approach in production
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context but need to wait
                    import threading
                    import time

                    event = threading.Event()

                    def check_future():
                        if future.done():
                            event.set()
                            return
                        loop.call_later(0.1, check_future)

                    loop.call_soon(check_future)
                    # Wait with timeout to avoid hanging
                    event.wait(timeout=5.0)
                    if future.done():
                        results = future.result()
                    else:
                        # Fallback to in-memory if query is taking too long
                        return [node for node in self.nodes.values() if node.layer_id == layer_id]
                else:
                    # If we're not in an async context, we can just run the loop
                    results = loop.run_until_complete(future)

                # Convert results to GraphNode objects
                nodes = []
                for result in results:
                    node_data = result.get("n", {})
                    if node_data:
                        node = GraphNode(**node_data)
                        nodes.append(node)

                return nodes
            except Exception as e:
                # Log error and fall back to in-memory
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error retrieving nodes from database: {str(e)}")
                # Fall back to in-memory lookup

        # Return in-memory nodes
        return [node for node in self.nodes.values() if node.layer_id == layer_id]

    def get_edges_in_layer(self, layer_id: UUID) -> List[GraphEdge]:
        """
        Get all edges in a specific layer.

        If an adapter is set, this will attempt to retrieve from the database first.
        Otherwise, it will return the in-memory edges.

        Args:
            layer_id: The ID of the layer

        Returns:
            List of edges in the layer
        """
        # If adapter is set, try to get from database
        if self._adapter:
            import asyncio

            try:
                # Query edges in the layer using the adapter
                # This is a simplified example - actual implementation may vary
                # based on your database schema and query capabilities
                query = f"""
                MATCH (e:GraphEdge)-[:IN_LAYER]->(l:GraphLayer)
                WHERE l.id = '{str(layer_id)}'
                RETURN e
                """

                # Execute query and parse results
                future = asyncio.ensure_future(self._adapter._graph_db.query(query))
                # This is a blocking operation in a primarily non-blocking context
                # Consider implementing a fully async approach in production
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context but need to wait
                    import threading
                    import time

                    event = threading.Event()

                    def check_future():
                        if future.done():
                            event.set()
                            return
                        loop.call_later(0.1, check_future)

                    loop.call_soon(check_future)
                    # Wait with timeout to avoid hanging
                    event.wait(timeout=5.0)
                    if future.done():
                        results = future.result()
                    else:
                        # Fallback to in-memory if query is taking too long
                        return [edge for edge in self.edges.values() if edge.layer_id == layer_id]
                else:
                    # If we're not in an async context, we can just run the loop
                    results = loop.run_until_complete(future)

                # Convert results to GraphEdge objects
                edges = []
                for result in results:
                    edge_data = result.get("e", {})
                    if edge_data:
                        edge = GraphEdge(**edge_data)
                        edges.append(edge)

                return edges
            except Exception as e:
                # Log error and fall back to in-memory
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error retrieving edges from database: {str(e)}")
                # Fall back to in-memory lookup

        # Return in-memory edges
        return [edge for edge in self.edges.values() if edge.layer_id == layer_id]

    def set_adapter(self, adapter):
        """
        Set the graph database adapter for this graph.

        Args:
            adapter: An instance of LayeredGraphDBAdapter
        """
        self._adapter = adapter
        return self

    async def persist(self):
        """
        Persist the entire graph to the database using the adapter.

        This method requires that an adapter has been set using set_adapter().

        Returns:
            The ID of the stored graph

        Raises:
            ValueError: If no adapter has been set
        """
        if not self._adapter:
            raise ValueError("No adapter set. Use set_adapter() before calling persist().")

        try:
            return await self._adapter.store_graph(self)
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error persisting graph to database: {str(e)}")
            raise

    @classmethod
    async def from_database(cls, graph_id: UUID, adapter):
        """
        Retrieve a graph from the database using the adapter.

        Args:
            graph_id: The ID of the graph to retrieve
            adapter: An instance of LayeredGraphDBAdapter

        Returns:
            A LayeredKnowledgeGraphDP instance populated from the database
        """
        import logging

        logger = logging.getLogger(__name__)

        logger.debug(f"Attempting to retrieve graph with ID {graph_id} from database")

        # Ensure adapter is initialized
        await adapter._ensure_graph_db()
        logger.debug(f"Adapter initialized: {type(adapter._graph_db).__name__}")

        # Get the graph node - handle different adapter types
        graph_data = None

        if hasattr(adapter._graph_db, "extract_node"):
            logger.debug(f"Using extract_node method to get graph")
            graph_data = await adapter._graph_db.extract_node(str(graph_id))
            logger.debug(f"extract_node result: {graph_data}")
        else:
            # For NetworkXAdapter which doesn't have extract_node
            logger.debug(f"NetworkXAdapter detected")

            if not hasattr(adapter._graph_db, "graph") or not adapter._graph_db.graph:
                logger.debug(f"Loading graph from file")
                await adapter._graph_db.load_graph_from_file()

            logger.debug(
                f"Graph loaded in memory: {getattr(adapter._graph_db, 'graph') is not None}"
            )

            if hasattr(adapter._graph_db, "graph") and adapter._graph_db.graph:
                logger.debug(
                    f"Graph has node {str(graph_id)}: {adapter._graph_db.graph.has_node(str(graph_id))}"
                )

                # List all nodes for debugging
                logger.debug("Nodes in memory:")
                node_count = 0
                for node_id in adapter._graph_db.graph.nodes():
                    logger.debug(f"  Node: {node_id}")
                    node_count += 1
                    if node_count > 20:
                        logger.debug(
                            f"  ... and {len(adapter._graph_db.graph.nodes()) - 20} more nodes"
                        )
                        break

                if adapter._graph_db.graph.has_node(str(graph_id)):
                    graph_data = adapter._graph_db.graph.nodes[str(graph_id)]
                    logger.debug(f"Found graph data: {graph_data}")
                else:
                    logger.debug(f"Graph node not found with ID {str(graph_id)}")
            else:
                logger.debug("No graph found in memory")

        if not graph_data:
            # Direct search in the graph as a fallback
            if hasattr(adapter._graph_db, "graph") and adapter._graph_db.graph:
                for node_id, node_data in adapter._graph_db.graph.nodes(data=True):
                    if node_data.get("id") == str(graph_id):
                        logger.debug(f"Found graph node by id attribute: {node_id}")
                        graph_data = node_data
                        break
                    elif node_id == str(graph_id):
                        logger.debug(f"Found graph node by node_id: {node_id}")
                        graph_data = node_data
                        break

            if not graph_data:
                logger.debug(f"No graph data found for ID {graph_id}")
                raise ValueError(f"Graph with ID {graph_id} not found in the database")

        # Create a new graph instance
        logger.debug(f"Creating graph instance from data: {graph_data}")

        try:
            id_value = graph_data.get("id", str(graph_id))
            if not isinstance(id_value, str):
                id_value = str(id_value)

            graph = cls(
                id=UUID(id_value),
                name=graph_data.get("name", ""),
                description=graph_data.get("description", ""),
                metadata=graph_data.get("metadata", {}),
            )

            # Set the adapter
            graph.set_adapter(adapter)

            # Get all layers, nodes, and edges using adapter-specific methods
            try:
                # For adapters with query method
                if hasattr(adapter._graph_db, "query"):
                    logger.debug("Using query method to get graph components")
                    # Get all layers for this graph
                    query = f"""
                    MATCH (g)-[:CONTAINS_LAYER]->(l:GraphLayer)
                    WHERE g.id = '{str(graph_id)}'
                    RETURN l
                    """

                    layers_data = await adapter._graph_db.query(query)
                    logger.debug(f"Found {len(layers_data)} layers via query")

                    # Add layers to the graph
                    for layer_data in layers_data:
                        layer_node = layer_data.get("l", {})
                        if layer_node:
                            layer = GraphLayer(
                                id=UUID(layer_node.get("id")),
                                name=layer_node.get("name"),
                                description=layer_node.get("description"),
                                layer_type=layer_node.get("layer_type", "default"),
                                parent_layers=[
                                    UUID(parent_id)
                                    for parent_id in layer_node.get("parent_layers", [])
                                ],
                                properties=layer_node.get("properties", {}),
                                metadata=layer_node.get("metadata", {}),
                            )
                            graph.layers[layer.id] = layer

                    # Get all nodes for this graph
                    query = f"""
                    MATCH (n:GraphNode)-[:IN_LAYER]->(l:GraphLayer)<-[:CONTAINS_LAYER]-(g)
                    WHERE g.id = '{str(graph_id)}'
                    RETURN n, l.id as layer_id
                    """

                    nodes_data = await adapter._graph_db.query(query)
                    logger.debug(f"Found {len(nodes_data)} nodes via query")

                    # Add nodes to the graph
                    for node_data in nodes_data:
                        node_node = node_data.get("n", {})
                        layer_id = node_data.get("layer_id")
                        if node_node and layer_id:
                            node = GraphNode(
                                id=UUID(node_node.get("id")),
                                name=node_node.get("name"),
                                node_type=node_node.get("node_type"),
                                description=node_node.get("description"),
                                properties=node_node.get("properties", {}),
                                layer_id=UUID(layer_id),
                                metadata=node_node.get("metadata", {}),
                            )
                            graph.nodes[node.id] = node
                            graph.node_layer_map[node.id] = UUID(layer_id)

                    # Get all edges for this graph
                    query = f"""
                    MATCH (e:GraphEdge)-[:IN_LAYER]->(l:GraphLayer)<-[:CONTAINS_LAYER]-(g)
                    WHERE g.id = '{str(graph_id)}'
                    RETURN e, l.id as layer_id
                    """

                    edges_data = await adapter._graph_db.query(query)
                    logger.debug(f"Found {len(edges_data)} edges via query")

                    # Add edges to the graph
                    for edge_data in edges_data:
                        edge_node = edge_data.get("e", {})
                        layer_id = edge_data.get("layer_id")
                        if edge_node and layer_id:
                            edge = GraphEdge(
                                id=UUID(edge_node.get("id")),
                                source_node_id=UUID(edge_node.get("source_node_id")),
                                target_node_id=UUID(edge_node.get("target_node_id")),
                                relationship_name=edge_node.get("relationship_name"),
                                properties=edge_node.get("properties", {}),
                                layer_id=UUID(layer_id),
                                metadata=edge_node.get("metadata", {}),
                            )
                            graph.edges[edge.id] = edge
                            graph.edge_layer_map[edge.id] = UUID(layer_id)

                # For NetworkXAdapter which uses direct graph access
                elif hasattr(adapter._graph_db, "graph"):
                    logger.debug("Using direct graph access to get graph components")
                    nx_graph = adapter._graph_db.graph

                    # Find all layer nodes
                    layer_count = 0
                    for node_id, node_data in nx_graph.nodes(data=True):
                        if node_data.get("metadata", {}).get("type") == "GraphLayer":
                            logger.debug(f"Found potential layer: {node_id}")

                            # Check if this layer is part of our graph
                            # First, try to find direct edge from graph to layer
                            layer_found = False
                            if nx_graph.has_edge(str(graph_id), node_id):
                                edges = nx_graph.get_edge_data(str(graph_id), node_id)
                                for key, edge_data in edges.items():
                                    if key == "CONTAINS_LAYER":
                                        logger.debug(
                                            f"Found layer {node_id} belonging to graph via edge {key}"
                                        )
                                        layer_found = True
                                        break

                            # If not found, try looking for all edges
                            if not layer_found:
                                for source, target, key, edge_data in nx_graph.edges(
                                    str(graph_id), data=True, keys=True
                                ):
                                    if target == node_id and key == "CONTAINS_LAYER":
                                        logger.debug(f"Found layer {node_id} belonging to graph")
                                        layer_found = True
                                        break

                            if layer_found:
                                # This layer belongs to our graph
                                try:
                                    parent_layers = []
                                    if "parent_layers" in node_data:
                                        raw_parent_layers = node_data["parent_layers"]
                                        if isinstance(raw_parent_layers, str):
                                            import json

                                            parent_layers = [
                                                UUID(p) for p in json.loads(raw_parent_layers)
                                            ]
                                        elif isinstance(raw_parent_layers, list):
                                            parent_layers = [UUID(p) for p in raw_parent_layers]

                                    layer = GraphLayer(
                                        id=UUID(node_id),
                                        name=node_data.get("name", ""),
                                        description=node_data.get("description", ""),
                                        layer_type=node_data.get("layer_type", "default"),
                                        parent_layers=parent_layers,
                                        properties=node_data.get("properties", {}),
                                        metadata=node_data.get("metadata", {}),
                                    )
                                    graph.layers[layer.id] = layer
                                    layer_count += 1
                                except Exception as e:
                                    logger.error(
                                        f"Error creating layer from node {node_id}: {str(e)}"
                                    )

                    logger.debug(f"Found {layer_count} layers via direct access")

                    # Find all node nodes and their layer relationships
                    node_count = 0
                    for node_id, node_data in nx_graph.nodes(data=True):
                        if node_data.get("metadata", {}).get("type") == "GraphNode":
                            # Find the layer this node belongs to
                            for _, target, key, edge_data in nx_graph.edges(
                                node_id, data=True, keys=True
                            ):
                                if key == "IN_LAYER" and UUID(target) in graph.layers:
                                    # This node belongs to a layer in our graph
                                    try:
                                        node = GraphNode(
                                            id=UUID(node_id),
                                            name=node_data.get("name", ""),
                                            node_type=node_data.get("node_type", ""),
                                            description=node_data.get("description", ""),
                                            properties=node_data.get("properties", {}),
                                            layer_id=UUID(target),
                                            metadata=node_data.get("metadata", {}),
                                        )
                                        graph.nodes[node.id] = node
                                        graph.node_layer_map[node.id] = UUID(target)
                                        node_count += 1
                                        logger.debug(f"Found node {node_id} in layer {target}")
                                    except Exception as e:
                                        logger.error(
                                            f"Error creating node from node {node_id}: {str(e)}"
                                        )

                    logger.debug(f"Found {node_count} nodes via direct access")

                    # Find all edge nodes and their layer relationships
                    edge_count = 0
                    for node_id, node_data in nx_graph.nodes(data=True):
                        if node_data.get("metadata", {}).get("type") == "GraphEdge":
                            # Find the layer this edge belongs to
                            for _, target, key, edge_data in nx_graph.edges(
                                node_id, data=True, keys=True
                            ):
                                if key == "IN_LAYER" and UUID(target) in graph.layers:
                                    # This edge belongs to a layer in our graph
                                    try:
                                        edge = GraphEdge(
                                            id=UUID(node_id),
                                            source_node_id=UUID(node_data.get("source_node_id")),
                                            target_node_id=UUID(node_data.get("target_node_id")),
                                            relationship_name=node_data.get(
                                                "relationship_name", ""
                                            ),
                                            properties=node_data.get("properties", {}),
                                            layer_id=UUID(target),
                                            metadata=node_data.get("metadata", {}),
                                        )
                                        graph.edges[edge.id] = edge
                                        graph.edge_layer_map[edge.id] = UUID(target)
                                        edge_count += 1
                                        logger.debug(f"Found edge {node_id} in layer {target}")
                                    except Exception as e:
                                        logger.error(
                                            f"Error creating edge from node {node_id}: {str(e)}"
                                        )

                    logger.debug(f"Found {edge_count} edges via direct access")

                else:
                    logger.warning(
                        f"Adapter {type(adapter._graph_db).__name__} doesn't support query or direct graph access"
                    )

            except Exception as e:
                logger.error(f"Error retrieving graph components from database: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())

            logger.debug(
                f"Final graph contains {len(graph.layers)} layers, {len(graph.nodes)} nodes, and {len(graph.edges)} edges"
            )
            return graph

        except Exception as e:
            logger.error(f"Error creating graph instance: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            raise

    def __str__(self):
        """String representation of the graph."""
        return f"LayeredKnowledgeGraph(id={self.id}, name={self.name}, layers={len(self.layers)}, nodes={len(self.nodes)}, edges={len(self.edges)})"

    def __repr__(self):
        """Detailed representation of the graph."""
        return self.__str__()
