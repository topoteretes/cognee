"""
Layered Knowledge Graph Builder for Cognee.

This module provides utilities for building and managing layered knowledge graphs.
"""

import uuid
from typing import Dict, List, Optional, Any, Union

from cognee.shared.data_models import (
    LayeredKnowledgeGraph,
    KnowledgeGraph,
    Layer,
    Node,
    Edge
)


class LayeredGraphBuilder:
    """
    Utility class for building layered knowledge graphs in Cognee.
    
    This class provides methods for creating and managing layered knowledge graphs,
    including adding layers, nodes, and edges to specific layers, and building
    hierarchical relationships between layers.
    """
    
    def __init__(self, name: str = "Layered Knowledge Graph", description: str = ""):
        """
        Initialize a new layered graph builder.
        
        Args:
            name: Name of the layered graph
            description: Description of the layered graph
        """
        # Initialize empty base graph
        self.base_graph = KnowledgeGraph(nodes=[], edges=[])
        
        # Initialize layered graph
        self.layered_graph = LayeredKnowledgeGraph(
            base_graph=self.base_graph,
            layers=[],
            name=name,
            description=description
        )
        
        # Keep track of layers
        self.layers: Dict[str, Layer] = {}
        
        # Keep track of node IDs by layer
        self.layer_nodes: Dict[str, List[str]] = {}
        
        # Keep track of edge IDs by layer
        self.layer_edges: Dict[str, List[tuple]] = {}
    
    def create_layer(
        self, 
        name: str, 
        description: str, 
        layer_type: str = "default", 
        parent_layers: List[str] = None,
        layer_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new layer in the layered graph.
        
        Args:
            name: Name of the layer
            description: Description of the layer
            layer_type: Type of the layer (e.g., "base", "enrichment", "inference")
            parent_layers: List of parent layer IDs
            layer_id: Specific ID for the layer (generated if not provided)
            properties: Additional layer properties
            
        Returns:
            ID of the created layer
        """
        # Generate ID if not provided
        if layer_id is None:
            layer_id = str(uuid.uuid4())
            
        # Initialize empty parents list if not provided
        if parent_layers is None:
            parent_layers = []
            
        # Initialize empty properties dict if not provided
        if properties is None:
            properties = {}
            
        # Verify parent layers exist
        for parent_id in parent_layers:
            if parent_id not in self.layers:
                raise ValueError(f"Parent layer with ID {parent_id} does not exist")
        
        # Create layer
        layer = Layer(
            id=layer_id,
            name=name,
            description=description,
            layer_type=layer_type,
            parent_layers=parent_layers,
            properties=properties
        )
        
        # Add layer to layered graph
        self.layered_graph.add_layer(layer)
        
        # Keep track of layer
        self.layers[layer_id] = layer
        self.layer_nodes[layer_id] = []
        self.layer_edges[layer_id] = []
        
        return layer_id
    
    def add_node_to_layer(
        self,
        layer_id: str,
        node_id: str,
        name: str,
        node_type: str,
        description: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Node:
        """
        Add a node to a specific layer.
        
        Args:
            layer_id: ID of the layer to add the node to
            node_id: ID of the node
            name: Name of the node
            node_type: Type of the node
            description: Description of the node
            properties: Additional node properties
            
        Returns:
            Created node
        """
        # Verify layer exists
        if layer_id not in self.layers:
            raise ValueError(f"Layer with ID {layer_id} does not exist")
            
        # Create node
        node = Node(
            id=node_id,
            name=name,
            type=node_type,
            description=description,
            layer_id=layer_id,
            properties=properties
        )
        
        # Add node to layer
        self.layered_graph.add_node_to_layer(node, layer_id)
        
        # Keep track of node
        self.layer_nodes[layer_id].append(node_id)
        
        return node
    
    def add_edge_to_layer(
        self,
        layer_id: str,
        source_node_id: str,
        target_node_id: str,
        relationship_name: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Edge:
        """
        Add an edge to a specific layer.
        
        Args:
            layer_id: ID of the layer to add the edge to
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            relationship_name: Name of the relationship
            properties: Additional edge properties
            
        Returns:
            Created edge
        """
        # Verify layer exists
        if layer_id not in self.layers:
            raise ValueError(f"Layer with ID {layer_id} does not exist")
            
        # Create edge
        edge = Edge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_name=relationship_name,
            layer_id=layer_id,
            properties=properties
        )
        
        # Add edge to layer
        self.layered_graph.add_edge_to_layer(edge, layer_id)
        
        # Keep track of edge
        self.layer_edges[layer_id].append((source_node_id, target_node_id, relationship_name))
        
        return edge
    
    def add_subgraph_to_layer(
        self,
        layer_id: str,
        subgraph: KnowledgeGraph,
        id_prefix: str = ""
    ) -> Dict[str, str]:
        """
        Add an entire subgraph to a layer, with optional ID prefixing to avoid conflicts.
        
        Args:
            layer_id: ID of the layer to add the subgraph to
            subgraph: Knowledge graph to add
            id_prefix: Prefix to add to node IDs to avoid conflicts
            
        Returns:
            Dictionary mapping original node IDs to new node IDs
        """
        # Verify layer exists
        if layer_id not in self.layers:
            raise ValueError(f"Layer with ID {layer_id} does not exist")
            
        # Map to track original to new node IDs
        id_mapping = {}
        
        # Add nodes
        for node in subgraph.nodes:
            new_id = f"{id_prefix}{node.id}" if id_prefix else node.id
            
            # Create a copy of the node with the new ID and layer
            new_node = Node(
                id=new_id,
                name=node.name,
                type=node.type,
                description=node.description,
                layer_id=layer_id,
                properties=node.properties
            )
            
            # Add to layered graph
            self.layered_graph.add_node_to_layer(new_node, layer_id)
            
            # Keep track of mapping and node
            id_mapping[node.id] = new_id
            self.layer_nodes[layer_id].append(new_id)
        
        # Add edges
        for edge in subgraph.edges:
            # Map source and target IDs
            new_source_id = id_mapping.get(edge.source_node_id, edge.source_node_id)
            new_target_id = id_mapping.get(edge.target_node_id, edge.target_node_id)
            
            # Create a copy of the edge with new IDs and layer
            new_edge = Edge(
                source_node_id=new_source_id,
                target_node_id=new_target_id,
                relationship_name=edge.relationship_name,
                layer_id=layer_id,
                properties=edge.properties
            )
            
            # Add to layered graph
            self.layered_graph.add_edge_to_layer(new_edge, layer_id)
            
            # Keep track of edge
            self.layer_edges[layer_id].append((new_source_id, new_target_id, edge.relationship_name))
        
        return id_mapping
    
    def build(self) -> LayeredKnowledgeGraph:
        """
        Build and return the layered knowledge graph.
        
        Returns:
            Complete layered knowledge graph
        """
        return self.layered_graph


async def convert_to_layered_graph(
    knowledge_graph: KnowledgeGraph,
    layer_name: str = "Base Layer",
    layer_description: str = "Original knowledge graph",
    graph_name: str = "Layered Knowledge Graph",
    graph_description: str = "Layered knowledge graph converted from standard knowledge graph"
) -> LayeredKnowledgeGraph:
    """
    Convert a standard knowledge graph to a layered knowledge graph with one layer.
    
    Args:
        knowledge_graph: Standard knowledge graph to convert
        layer_name: Name for the base layer
        layer_description: Description for the base layer
        graph_name: Name for the layered graph
        graph_description: Description for the layered graph
        
    Returns:
        Layered knowledge graph with one layer containing all nodes and edges
    """
    builder = LayeredGraphBuilder(name=graph_name, description=graph_description)
    
    # Create base layer
    layer_id = builder.create_layer(
        name=layer_name,
        description=layer_description,
        layer_type="base"
    )
    
    # Add all nodes and edges to the layer
    builder.add_subgraph_to_layer(layer_id, knowledge_graph)
    
    return builder.build() 