"""
Simplified implementation of layered knowledge graphs.

This module provides data models to represent layered knowledge graphs, including nodes, edges,
and layers. These models are designed to work directly with database adapters like NetworkX.
"""

import uuid
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import UUID
from pydantic import Field, field_validator, model_validator
import logging
from datetime import datetime
import networkx as nx
import json

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

logger = logging.getLogger(__name__)


def safe_uuid(value) -> Optional[UUID]:
    """Safely convert a value to UUID, returning None if conversion fails"""
    if value is None:
        return None

    if isinstance(value, UUID):
        return value

    try:
        if isinstance(value, str):
            # Strip any whitespace or quotes
            value = value.strip().strip("\"'")
            return UUID(value)
        elif isinstance(value, bytes):
            return UUID(value.decode("utf-8"))
        elif isinstance(value, (int, float)):
            # Convert numeric types to string first
            return UUID(str(int(value)))
        else:
            # Try string conversion for other types
            return UUID(str(value))
    except (ValueError, AttributeError, TypeError) as e:
        logger.debug(f"Failed to convert {value} to UUID: {str(e)}")
        return None


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

    id: UUID
    name: str
    node_type: str
    description: Optional[str] = ""
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
        description: str = "",
        properties: Dict = None,
        metadata: Dict = None,
    ) -> "GraphNode":
        """
        Create a new node (in-memory only, not persisted to any database)

        For database operations, use the appropriate adapter class like
        LayeredGraphDBAdapter or NetworkXAdapter
        """
        return cls(
            id=uuid.uuid4(),
            name=name,
            node_type=node_type,
            description=description,
            properties=properties or {},
            metadata=metadata or {},
        )


class GraphEdge(DataPoint):
    """
    Represents an edge in a layered knowledge graph.

    Attributes:
        id: Unique identifier for the edge
        source_id: ID of the source node
        target_id: ID of the target node
        edge_type: Type of the edge
        properties: Additional properties of the edge
        layer_id: ID of the layer this edge belongs to
        metadata: Metadata for the edge
    """

    source_id: UUID
    target_id: UUID
    edge_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    layer_id: Optional[UUID] = None
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "GraphEdge", "index_fields": ["edge_type"]}
    )

    @classmethod
    def create(
        cls,
        source_id: UUID,
        target_id: UUID,
        edge_type: str,
        properties: Dict = None,
        metadata: Dict = None,
    ) -> "GraphEdge":
        """
        Create a new edge (in-memory only, not persisted to any database)

        For database operations, use the appropriate adapter class like
        LayeredGraphDBAdapter or NetworkXAdapter
        """
        return cls(
            id=uuid.uuid4(),
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            properties=properties or {},
            metadata=metadata or {},
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


class LayeredKnowledgeGraph(DataPoint):
    """A graph with multiple layers, each containing nodes and edges."""

    id: UUID = Field(default_factory=uuid.uuid4)
    name: str = ""
    description: str = ""
    adapter: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._layers_cache = {}  # Cache for layers
        self._nodes_cache = {}  # Cache for nodes
        self._edges_cache = {}  # Cache for edges

    def set_adapter(self, adapter: Any) -> None:
        """Set the adapter for this graph"""
        self.adapter = adapter

    async def initialize(self):
        """Initialize the graph in the database if it doesn't exist"""
        if not self.adapter:
            raise ValueError("No adapter set. The adapter must be provided during initialization.")

        # Check if graph exists using has_node
        exists = False
        try:
            exists = await self.adapter._graph_db.has_node(str(self.id))
        except Exception as e:
            import logging

            logging.warning(f"Error checking if graph exists: {e}")

        if not exists:
            # Create graph node in the database with type information in metadata
            # Try both methods to ensure compatibility
            try:
                # Use LayeredGraphDP if available in the enhanced adapter
                from cognee.modules.graph.enhanced_layered_graph_adapter import LayeredGraphDP

                graph_dp = LayeredGraphDP(
                    id=self.id,
                    name=self.name,
                    description=self.description,
                    type="LayeredKnowledgeGraph",
                )
                await self.adapter._graph_db.add_node(graph_dp)
            except (ImportError, Exception):
                # Fallback to standard DataPoint
                from cognee.infrastructure.engine import DataPoint

                graph_dp = DataPoint(
                    id=self.id,
                    name=self.name,
                    description=self.description,
                    metadata={"type": "LayeredKnowledgeGraph", "index_fields": ["name"]},
                )
                await self.adapter._graph_db.add_node(graph_dp)

        return self

    @classmethod
    async def create(cls, name: str, description: str, adapter: Any):
        """
        Create a new layered knowledge graph.

        Args:
            name: Name of the graph
            description: Description of the graph
            adapter: The adapter to use for database operations

        Returns:
            A new LayeredKnowledgeGraph instance
        """
        graph = cls(name=name, description=description, adapter=adapter)
        await graph.initialize()
        return graph

    @classmethod
    async def load(cls, graph_id: UUID, adapter: Any):
        """
        Load a layered knowledge graph from the database.

        Args:
            graph_id: The ID of the graph to load
            adapter: The adapter to use for database operations

        Returns:
            A LayeredKnowledgeGraph instance loaded from the database
        """
        # Ensure the adapter is initialized
        if not adapter._graph_db_initialized:
            await adapter._ensure_graph_db()

        # Check if graph exists
        exists = await adapter._graph_db.has_node(str(graph_id))
        if not exists:
            raise ValueError(f"Graph with ID {graph_id} not found")

        # Get graph data
        graph_data = {}
        if hasattr(adapter._graph_db, "graph") and adapter._graph_db.graph.has_node(str(graph_id)):
            graph_data = adapter._graph_db.graph.nodes[str(graph_id)]

        # Create graph instance
        graph = cls(
            graph_id=graph_id,
            name=graph_data.get("name", ""),
            description=graph_data.get("description", ""),
            adapter=adapter,
        )

        # Optionally preload some data to cache
        await graph._preload_data()

        return graph

    async def _preload_data(self):
        """Preload essential data to local caches for better performance"""
        if not self.adapter:
            return

        # Load layers
        layers = await self._get_layers_from_db()
        for layer in layers:
            self._layers_cache[layer.id] = layer

        # Load node and edge metadata (not full data to avoid memory issues)
        # This is optional and can be adjusted based on graph size

    async def _get_layers_from_db(self) -> List[GraphLayer]:
        """Get all layers in this graph from the database"""
        layers = []

        # If it's a NetworkX adapter
        if hasattr(self.adapter._graph_db, "graph"):
            nx_graph = self.adapter._graph_db.graph

            # Find all outgoing CONTAINS_LAYER edges from the graph node
            if nx_graph.has_node(str(self.id)):
                for _, target, key in nx_graph.edges(str(self.id), keys=True):
                    if key == "CONTAINS_LAYER":
                        # Found a layer node
                        layer_data = nx_graph.nodes[target]
                        logger.debug(f"Found layer node in database: {target} - {layer_data}")

                        # Parse parent layers from the node data
                        parent_layers = []
                        if "parent_layers" in layer_data:
                            # Handle different formats of parent_layers
                            try:
                                if isinstance(layer_data["parent_layers"], str):
                                    # Try to parse as JSON
                                    try:
                                        import json

                                        parent_layers = [
                                            safe_uuid(p)
                                            for p in json.loads(layer_data["parent_layers"])
                                        ]
                                    except json.JSONDecodeError:
                                        # Treat as a comma-separated string
                                        parent_layers = [
                                            safe_uuid(p.strip())
                                            for p in layer_data["parent_layers"].split(",")
                                        ]
                                elif isinstance(layer_data["parent_layers"], list):
                                    parent_layers = [
                                        safe_uuid(p) for p in layer_data["parent_layers"]
                                    ]

                                # Filter out None values
                                parent_layers = [p for p in parent_layers if p is not None]
                                logger.debug(f"Parsed parent layers: {parent_layers}")
                            except Exception as e:
                                logger.error(f"Error parsing parent layers: {e}")
                                parent_layers = []

                        # As a fallback, check for EXTENDS_LAYER edges
                        if not parent_layers:
                            logger.debug(f"Looking for EXTENDS_LAYER edges from {target}")
                            try:
                                for _, parent_target, edge_key in nx_graph.edges(target, keys=True):
                                    if edge_key == "EXTENDS_LAYER":
                                        parent_uuid = safe_uuid(parent_target)
                                        if parent_uuid:
                                            parent_layers.append(parent_uuid)
                                            logger.debug(
                                                f"Found parent layer via edge: {parent_target}"
                                            )
                            except Exception as e:
                                logger.error(f"Error finding parent layers via edges: {e}")

                        # Create layer
                        try:
                            layer = GraphLayer(
                                id=UUID(target),
                                name=layer_data.get("name", ""),
                                description=layer_data.get("description", ""),
                                layer_type=layer_data.get("layer_type", "default"),
                                parent_layers=parent_layers,
                                properties=layer_data.get("properties", {}),
                                metadata=layer_data.get("metadata", {}),
                            )
                            layers.append(layer)
                            logger.debug(
                                f"Created layer object: {layer.id} with {len(parent_layers)} parent layers"
                            )
                        except Exception as e:
                            logger.error(f"Error creating layer object: {e}")

        return layers

    async def add_layer(
        self,
        name: str,
        description: str,
        layer_type: str = "default",
        parent_layers: List[UUID] = None,
    ) -> GraphLayer:
        """
        Add a new layer to the graph.

        Args:
            name: Name of the layer
            description: Description of the layer
            layer_type: Type of the layer
            parent_layers: List of parent layer IDs

        Returns:
            The newly created layer
        """
        if not self.adapter:
            raise ValueError("No adapter set")

        # Convert parent_layers to list of strings for better compatibility with adapters
        str_parent_layers = [str(p) for p in (parent_layers or [])]
        logger.info(f"Adding layer with parent layers: {str_parent_layers}")

        # Create layer
        layer = GraphLayer(
            id=uuid.uuid4(),
            name=name,
            description=description,
            layer_type=layer_type,
            parent_layers=parent_layers or [],
        )

        # Create a node for the database
        # Check if we're using NetworkXAdapter
        if hasattr(self.adapter._graph_db, "graph") and isinstance(
            self.adapter._graph_db.graph, nx.MultiDiGraph
        ):
            logger.debug("Using NetworkXAdapter to add layer node")
            # NetworkXAdapter expects node_id and attributes as separate arguments
            self.adapter._graph_db.graph.add_node(
                str(layer.id),
                id=str(layer.id),
                name=layer.name,
                description=layer.description,
                layer_type=layer.layer_type,
                parent_layers=str_parent_layers,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            # Save the graph
            await self.adapter._graph_db.save_graph_to_file(self.adapter._graph_db.filename)
        else:
            # For other adapters, try to use the add_node method
            try:
                # Create a simple adapter-compatible object
                class SimpleNode:
                    def __init__(self, id, **data):
                        self.id = id
                        self.data = data

                    def model_dump(self):
                        return {**self.data, "id": self.id}

                node_data = {
                    "name": layer.name,
                    "description": layer.description,
                    "layer_type": layer.layer_type,
                    "parent_layers": str_parent_layers,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }

                node = SimpleNode(str(layer.id), **node_data)
                await self.adapter._graph_db.add_node(node)
            except Exception as e:
                logger.error(f"Error adding node: {e}")
                # Last resort - just try to add the node directly
                await self.adapter._graph_db.add_node(layer)

        # Add CONTAINS_LAYER relationship
        await self.adapter._graph_db.add_edge(
            str(self.id),
            str(layer.id),
            "CONTAINS_LAYER",
            {"graph_id": str(self.id), "layer_id": str(layer.id)},
        )

        # Add parent layer relationships
        for parent_id in parent_layers or []:
            str_parent_id = str(parent_id)
            logger.info(f"Adding EXTENDS_LAYER relationship from {layer.id} to {str_parent_id}")

            await self.adapter._graph_db.add_edge(
                str(layer.id),
                str_parent_id,
                "EXTENDS_LAYER",
                {"child_layer_id": str(layer.id), "parent_layer_id": str_parent_id},
            )

        # Update cache
        self._layers_cache[layer.id] = layer

        return layer

    async def add_node(
        self,
        name: str,
        node_type: str,
        properties: Dict = None,
        metadata: Dict = None,
        layer_id: UUID = None,
    ) -> GraphNode:
        """
        Add a node to the graph, optionally in a layer.

        Args:
            name: Name of the node
            node_type: Type of the node
            properties: Node properties
            metadata: Node metadata
            layer_id: Layer to add the node to (optional)

        Returns:
            The newly created node
        """
        if not self.adapter:
            raise ValueError("No adapter set")

        # Create node
        node = GraphNode(
            id=uuid.uuid4(),
            name=name,
            node_type=node_type,
            properties=properties or {},
            metadata=metadata or {},
        )

        # Add to database
        if hasattr(self.adapter._graph_db, "graph") and isinstance(
            self.adapter._graph_db.graph, nx.MultiDiGraph
        ):
            logger.debug("Using NetworkXAdapter to add node")
            # NetworkXAdapter expects node_id and attributes as separate arguments
            self.adapter._graph_db.graph.add_node(
                str(node.id),
                id=str(node.id),
                name=node.name,
                node_type=node.node_type,
                properties=json.dumps(node.properties) if node.properties else "{}",
                metadata=json.dumps(node.metadata) if node.metadata else "{}",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            # Save the graph
            await self.adapter._graph_db.save_graph_to_file(self.adapter._graph_db.filename)
        else:
            # For other adapters, try to use the add_node method
            try:
                # Create a simple adapter-compatible object
                class SimpleNode:
                    def __init__(self, id, **data):
                        self.id = id
                        self.data = data

                    def model_dump(self):
                        return {**self.data, "id": self.id}

                node_data = {
                    "name": node.name,
                    "node_type": node.node_type,
                    "properties": node.properties,
                    "metadata": node.metadata,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }

                simple_node = SimpleNode(str(node.id), **node_data)
                await self.adapter._graph_db.add_node(simple_node)
            except Exception as e:
                logger.error(f"Error adding node: {e}")
                await self.adapter._graph_db.add_node(node)

        # Connect to layer if provided
        if layer_id:
            if layer_id not in self._layers_cache:
                # Try to find the layer
                layers = await self.get_layers()
                layer_found = False
                for layer in layers:
                    if layer.id == layer_id:
                        layer_found = True
                        self._layers_cache[layer.id] = layer
                        break

                if not layer_found:
                    raise ValueError(f"Layer with ID {layer_id} not found")

            # Add CONTAINS_NODE relationship
            await self.adapter._graph_db.add_edge(
                str(layer_id),
                str(node.id),
                "CONTAINS_NODE",
                {"layer_id": str(layer_id), "node_id": str(node.id)},
            )

        return node

    async def add_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        edge_type: str,
        properties: Dict = None,
        metadata: Dict = None,
        layer_id: UUID = None,
    ) -> GraphEdge:
        """
        Add an edge between two nodes to the graph, optionally in a layer.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of the edge
            properties: Edge properties
            metadata: Edge metadata
            layer_id: Layer to add the edge to (optional)

        Returns:
            The newly created edge
        """
        if not self.adapter:
            raise ValueError("No adapter set")

        # Check if nodes exist
        if not await self.adapter._graph_db.has_node(str(source_id)):
            raise ValueError(f"Source node with ID {source_id} not found")

        if not await self.adapter._graph_db.has_node(str(target_id)):
            raise ValueError(f"Target node with ID {target_id} not found")

        # Create edge
        edge = GraphEdge(
            id=uuid.uuid4(),
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            properties=properties or {},
            metadata=metadata or {},
        )

        # Add edge to graph
        edge_data = {
            "edge_id": str(edge.id),  # Use edge_id instead of id to avoid duplicate
            "source_id": str(edge.source_id),
            "target_id": str(edge.target_id),
            "edge_type": edge.edge_type,
            "properties": json.dumps(edge.properties) if edge.properties else "{}",
            "metadata": json.dumps(edge.metadata) if edge.metadata else "{}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        await self.adapter._graph_db.add_edge(
            str(edge.source_id), str(edge.target_id), edge.edge_type, edge_data
        )

        # Connect to layer if provided
        if layer_id:
            if layer_id not in self._layers_cache:
                # Try to find the layer
                layers = await self.get_layers()
                layer_found = False
                for layer in layers:
                    if layer.id == layer_id:
                        layer_found = True
                        self._layers_cache[layer.id] = layer
                        break

                if not layer_found:
                    raise ValueError(f"Layer with ID {layer_id} not found")

            # Create an edge node for the CONTAINS_EDGE relationship
            edge_node_id = f"edge_{edge.id}"

            if hasattr(self.adapter._graph_db, "graph") and isinstance(
                self.adapter._graph_db.graph, nx.MultiDiGraph
            ):
                # Add an edge node to store edge data and connect it to the layer
                # Remove 'id' from edge_data to avoid duplicate
                edge_node_data = {
                    "node_id": edge_node_id,  # Use node_id instead of id
                    "is_edge_node": True,
                    **edge_data,
                }

                self.adapter._graph_db.graph.add_node(edge_node_id, **edge_node_data)
                await self.adapter._graph_db.save_graph_to_file(self.adapter._graph_db.filename)

                # Add CONTAINS_EDGE relationship from layer to edge node
                await self.adapter._graph_db.add_edge(
                    str(layer_id),
                    edge_node_id,
                    "CONTAINS_EDGE",
                    {"layer_id": str(layer_id), "edge_id": str(edge.id)},
                )
            else:
                # For other adapters, try to use a direct relationship
                await self.adapter._graph_db.add_edge(
                    str(layer_id),
                    str(edge.id),
                    "CONTAINS_EDGE",
                    {"layer_id": str(layer_id), "edge_id": str(edge.id)},
                )

        return edge

    async def _layer_exists_in_db(self, layer_id: UUID) -> bool:
        """Check if a layer exists in the database"""
        if hasattr(self.adapter._graph_db, "has_node"):
            exists = await self.adapter._graph_db.has_node(str(layer_id))

            # If it exists, also check if it's connected to this graph
            if exists and hasattr(self.adapter._graph_db, "has_edge"):
                connected = await self.adapter._graph_db.has_edge(
                    str(self.id), str(layer_id), "CONTAINS_LAYER"
                )
                return connected

            return exists

        # Fallback for NetworkX
        if hasattr(self.adapter._graph_db, "graph"):
            nx_graph = self.adapter._graph_db.graph
            has_node = nx_graph.has_node(str(layer_id))

            if has_node:
                # Check if it's connected to this graph
                return nx_graph.has_edge(str(self.id), str(layer_id), key="CONTAINS_LAYER")

        return False

    async def get_layers(self) -> List[GraphLayer]:
        """
        Get all layers in the graph.

        Returns:
            List of layers in the graph
        """
        # Try cache first
        if self._layers_cache:
            return list(self._layers_cache.values())

        # Get from database
        layers = await self._get_layers_from_db()

        # Update cache
        for layer in layers:
            self._layers_cache[layer.id] = layer

        return layers

    async def get_nodes_in_layer(self, layer_id: UUID) -> List[GraphNode]:
        """
        Get all nodes in a layer

        Args:
            layer_id: ID of the layer

        Returns:
            List of nodes in the layer
        """
        if not self.adapter:
            raise ValueError("No adapter set")

        # If using NetworkX adapter
        if hasattr(self.adapter._graph_db, "graph") and isinstance(
            self.adapter._graph_db.graph, nx.MultiDiGraph
        ):
            nx_graph = self.adapter._graph_db.graph
            nodes = []

            # Find all outgoing CONTAINS_NODE edges from layer
            if nx_graph.has_node(str(layer_id)):
                for source, target, key in nx_graph.out_edges(str(layer_id), keys=True):
                    if source == str(layer_id) and key == "CONTAINS_NODE":
                        # Get the node data
                        if nx_graph.has_node(target):
                            node_data = nx_graph.nodes[target]

                            # Parse metadata if it's a string
                            metadata = node_data.get("metadata", {})
                            if isinstance(metadata, str):
                                try:
                                    metadata = json.loads(metadata)
                                except:
                                    metadata = {}

                            # Parse properties if it's a string
                            properties = node_data.get("properties", {})
                            if isinstance(properties, str):
                                try:
                                    properties = json.loads(properties)
                                except:
                                    properties = {}

                            # Check if it's a node (not an edge node)
                            if node_data.get("is_edge_node", False) is not True:
                                # Create GraphNode
                                try:
                                    node = GraphNode(
                                        id=UUID(target),
                                        name=node_data.get("name", ""),
                                        node_type=node_data.get("node_type", ""),
                                        description=node_data.get("description", ""),
                                        properties=properties,
                                        metadata=metadata,
                                    )
                                    nodes.append(node)
                                except Exception as e:
                                    logger.error(f"Error creating node {target}: {e}")

            return nodes
        else:
            # For other adapters, use the generic approach
            # ... (other implementation)
            return []

    async def get_edges_in_layer(self, layer_id: UUID) -> List[GraphEdge]:
        """
        Get all edges in a layer

        Args:
            layer_id: ID of the layer

        Returns:
            List of edges in the layer
        """
        if not self.adapter:
            raise ValueError("No adapter set")

        # If using NetworkX adapter
        if hasattr(self.adapter._graph_db, "graph") and isinstance(
            self.adapter._graph_db.graph, nx.MultiDiGraph
        ):
            nx_graph = self.adapter._graph_db.graph
            edges = []

            # Find all outgoing CONTAINS_EDGE edges from layer
            if nx_graph.has_node(str(layer_id)):
                for source, target, key in nx_graph.out_edges(str(layer_id), keys=True):
                    if source == str(layer_id) and key == "CONTAINS_EDGE":
                        # Get the edge node data
                        if nx_graph.has_node(target) and target.startswith("edge_"):
                            edge_node_data = nx_graph.nodes[target]

                            # Extract the edge ID from the edge node data
                            edge_id = edge_node_data.get("edge_id")
                            if not edge_id:
                                continue

                            source_id = edge_node_data.get("source_id")
                            target_id = edge_node_data.get("target_id")

                            if not (source_id and target_id):
                                continue

                            # Parse metadata if it's a string
                            metadata = edge_node_data.get("metadata", {})
                            if isinstance(metadata, str):
                                try:
                                    metadata = json.loads(metadata)
                                except:
                                    metadata = {}

                            # Parse properties if it's a string
                            properties = edge_node_data.get("properties", {})
                            if isinstance(properties, str):
                                try:
                                    properties = json.loads(properties)
                                except:
                                    properties = {}

                            # Create GraphEdge
                            try:
                                edge = GraphEdge(
                                    id=UUID(edge_id),
                                    source_id=UUID(source_id),
                                    target_id=UUID(target_id),
                                    edge_type=edge_node_data.get("edge_type", ""),
                                    properties=properties,
                                    metadata=metadata,
                                )
                                edges.append(edge)
                            except Exception as e:
                                logger.error(f"Error creating edge {edge_id}: {e}")

            return edges
        else:
            # For other adapters, use the generic approach
            # ... (other implementation)
            return []

    async def get_layer_graph(self, layer_id: UUID) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Get all nodes and edges in a specific layer.

        Args:
            layer_id: The ID of the layer

        Returns:
            Tuple of (nodes, edges) in the layer
        """
        nodes = await self.get_nodes_in_layer(layer_id)
        edges = await self.get_edges_in_layer(layer_id)
        return nodes, edges

    async def get_cumulative_layer_graph(
        self, layer_id: UUID
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Get all nodes and edges in this layer and all of its parent layers

        Args:
            layer_id: The ID of the layer to get the graph for

        Returns:
            A tuple of (nodes, edges) in the cumulative graph
        """
        # Get all layers
        layers = await self.get_layers()

        # Map layers by their IDs for faster lookup
        layers_by_id = {layer.id: layer for layer in layers}

        logger.info(f"Looking up layer with ID: {layer_id}")
        logger.info(f"Found {len(layers)} total layers")
        for l in layers:
            logger.debug(f"Layer {l.id} ({l.name}): parent_layers={l.parent_layers}")

        if layer_id not in layers_by_id:
            logger.warning(f"Layer with ID {layer_id} not found in graph with ID {self.id}")
            return [], []

        target_layer = layers_by_id[layer_id]

        # Set to track layers we've already included
        included_layers = set()

        # Process function to add all parent layers recursively
        def include_layer_and_parents(current_layer_id: UUID):
            if current_layer_id in included_layers:
                return  # Already processed

            if current_layer_id not in layers_by_id:
                logger.warning(f"Referenced layer {current_layer_id} not found in graph")
                return

            current_layer = layers_by_id[current_layer_id]
            included_layers.add(current_layer_id)

            # Process all parent layers
            logger.debug(
                f"Processing layer {current_layer_id} with {len(current_layer.parent_layers)} parents"
            )
            for parent_id in current_layer.parent_layers:
                logger.debug(f"Adding parent layer: {parent_id}")
                include_layer_and_parents(parent_id)

        # Start with the target layer
        include_layer_and_parents(layer_id)

        logger.info(f"Included layers in cumulative view: {included_layers}")

        # Get all nodes and edges in the included layers
        all_nodes = []
        all_edges = []

        for lid in included_layers:
            # Get nodes and edges for this layer
            try:
                nodes = await self.get_nodes_in_layer(lid)
                logger.debug(f"Found {len(nodes)} nodes in layer {lid}")
                all_nodes.extend(nodes)

                edges = await self.get_edges_in_layer(lid)
                logger.debug(f"Found {len(edges)} edges in layer {lid}")
                all_edges.extend(edges)
            except Exception as e:
                logger.error(f"Error retrieving nodes/edges for layer {lid}: {str(e)}")

        logger.info(f"Cumulative graph has {len(all_nodes)} nodes and {len(all_edges)} edges")

        # Remove duplicates (just in case)
        unique_nodes = {node.id: node for node in all_nodes}
        unique_edges = {edge.id: edge for edge in all_edges}

        return list(unique_nodes.values()), list(unique_edges.values())

    def __str__(self):
        """String representation of the graph."""
        layer_count = len(self._layers_cache)
        return f"LayeredKnowledgeGraph(id={self.id}, name={self.name}, layers={layer_count})"

    def __repr__(self):
        """Detailed representation of the graph."""
        return self.__str__()

    @classmethod
    def create_empty(cls, name: str, description: str = "") -> "LayeredKnowledgeGraph":
        """
        Create a new empty layered knowledge graph (in-memory only)

        For database operations, use the appropriate adapter class like
        LayeredGraphDBAdapter or NetworkXAdapter

        Args:
            name: Name of the graph
            description: Description of the graph

        Returns:
            A new LayeredKnowledgeGraph instance
        """
        return cls(id=uuid.uuid4(), name=name, description=description)
