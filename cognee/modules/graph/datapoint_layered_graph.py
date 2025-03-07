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

class UUIDEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling UUID objects."""
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)

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
    metadata: Dict[str, Any] = Field(default_factory=lambda: {"type": "GraphNode", "index_fields": ["name"]})
    
    @classmethod
    def create(cls, name: str, node_type: str, description: str, properties: Optional[Dict[str, Any]] = None):
        """Create a new node with a generated UUID."""
        return cls(
            id=uuid.uuid4(),
            name=name,
            node_type=node_type,
            description=description,
            properties=properties or {}
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
    metadata: Dict[str, Any] = Field(default_factory=lambda: {"type": "GraphEdge", "index_fields": ["relationship_name"]})
    
    @classmethod
    def create(cls, source_node_id: UUID, target_node_id: UUID, relationship_name: str, properties: Optional[Dict[str, Any]] = None):
        """Create a new edge with a generated UUID."""
        return cls(
            id=uuid.uuid4(),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_name=relationship_name,
            properties=properties or {}
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
    metadata: Dict[str, Any] = Field(default_factory=lambda: {"type": "GraphLayer", "index_fields": ["name"]})
    
    @classmethod
    def create(cls, name: str, description: str, layer_type: str = "default", parent_layers: Optional[List[UUID]] = None, properties: Optional[Dict[str, Any]] = None):
        """Create a new layer with a generated UUID."""
        return cls(
            id=uuid.uuid4(),
            name=name,
            description=description,
            layer_type=layer_type,
            parent_layers=parent_layers or [],
            properties=properties or {}
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
    metadata: Dict[str, Any] = Field(default_factory=lambda: {"type": "LayeredKnowledgeGraph", "index_fields": ["name"]})
    
    @classmethod
    def create_empty(cls, name: str, description: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a new empty LayeredKnowledgeGraph with a generated UUID."""
        default_metadata = {"type": "LayeredKnowledgeGraph", "index_fields": ["name"]}
        if metadata:
            default_metadata.update(metadata)
        
        return cls(
            id=uuid.uuid4(),
            name=name,
            description=description,
            metadata=default_metadata
        )
    
    def add_layer(self, layer: GraphLayer) -> None:
        """Add a layer to the graph."""
        if layer.id in self.layers:
            raise InvalidValueError(f"Layer with ID {layer.id} already exists in the graph")
        self.layers[layer.id] = layer
    
    def add_node(self, node: GraphNode, layer_id: UUID) -> None:
        """
        Add a node to the graph and associate it with a layer.
        
        Args:
            node: The node to add
            layer_id: The ID of the layer to associate the node with
        
        Raises:
            InvalidValueError: If the node already exists or the layer doesn't exist
        """
        if node.id in self.nodes:
            raise InvalidValueError(f"Node with ID {node.id} already exists in the graph")
        
        if layer_id not in self.layers:
            raise InvalidValueError(f"Layer with ID {layer_id} doesn't exist in the graph")
        
        self.nodes[node.id] = node
        self.node_layer_map[node.id] = layer_id
        
        # Ensure the node knows which layer it belongs to
        node.layer_id = layer_id
    
    def add_edge(self, edge: GraphEdge, layer_id: UUID) -> None:
        """
        Add an edge to the graph and associate it with a layer.
        
        Args:
            edge: The edge to add
            layer_id: The ID of the layer to associate the edge with
        
        Raises:
            InvalidValueError: If the edge already exists, the layer doesn't exist,
                               or the source or target nodes don't exist
        """
        if edge.id in self.edges:
            raise InvalidValueError(f"Edge with ID {edge.id} already exists in the graph")
        
        if layer_id not in self.layers:
            raise InvalidValueError(f"Layer with ID {layer_id} doesn't exist in the graph")
        
        if edge.source_node_id not in self.nodes:
            raise InvalidValueError(f"Source node with ID {edge.source_node_id} doesn't exist in the graph")
        
        if edge.target_node_id not in self.nodes:
            raise InvalidValueError(f"Target node with ID {edge.target_node_id} doesn't exist in the graph")
        
        self.edges[edge.id] = edge
        self.edge_layer_map[edge.id] = layer_id
        
        # Ensure the edge knows which layer it belongs to
        edge.layer_id = layer_id
    
    def get_nodes_in_layer(self, layer_id: UUID) -> List[GraphNode]:
        """
        Get all nodes in a specific layer.
        
        Args:
            layer_id: The ID of the layer
        
        Returns:
            List of nodes in the layer
        """
        return [node for node in self.nodes.values() if node.layer_id == layer_id]
    
    def get_edges_in_layer(self, layer_id: UUID) -> List[GraphEdge]:
        """
        Get all edges in a specific layer.
        
        Args:
            layer_id: The ID of the layer
        
        Returns:
            List of edges in the layer
        """
        return [edge for edge in self.edges.values() if edge.layer_id == layer_id]
    
    def __str__(self):
        """String representation of the graph."""
        return f"LayeredKnowledgeGraph(id={self.id}, name={self.name}, layers={len(self.layers)}, nodes={len(self.nodes)}, edges={len(self.edges)})"
    
    def __repr__(self):
        """Detailed representation of the graph."""
        return self.__str__()