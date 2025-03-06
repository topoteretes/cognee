"""
DataPoint-based implementation of layered knowledge graphs.

This module provides classes for creating and managing layered knowledge graphs
that integrate with Cognee's DataPoint infrastructure.
"""

from typing import Dict, List, Optional, Set, Union, Any
from uuid import UUID, uuid4
from datetime import datetime
import json

from pydantic import Field, field_validator, model_validator

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.shared.data_models import Node, Edge, KnowledgeGraph, Layer

class GraphNode(DataPoint):
    """
    A node in a knowledge graph that extends DataPoint.
    
    Attributes:
        id: Unique identifier for the node
        name: Human-readable name of the node
        node_type: Type of the node (e.g., "Person", "Organization", "Concept")
        description: Detailed description of the node
        properties: Additional properties of the node
        layer_id: ID of the layer this node belongs to
    """
    name: str
    node_type: str
    description: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    layer_id: Optional[UUID] = None
    
    @classmethod
    def create(cls, name: str, node_type: str, description: str, 
               properties: Optional[Dict[str, Any]] = None, 
               layer_id: Optional[UUID] = None) -> 'GraphNode':
        """
        Create a new GraphNode with a generated UUID.
        
        Args:
            name: Human-readable name of the node
            node_type: Type of the node
            description: Detailed description of the node
            properties: Additional properties of the node
            layer_id: ID of the layer this node belongs to
            
        Returns:
            A new GraphNode instance
        """
        node_id = uuid4()
        return cls(
            id=node_id,
            name=name,
            node_type=node_type,
            description=description,
            properties=properties or {},
            layer_id=layer_id,
            metadata={"type": "GraphNode", "index_fields": ["name"], "created_at": datetime.now().isoformat()}
        )
    
    @classmethod
    def from_basic_node(cls, node: Node, layer_id: Optional[UUID] = None) -> 'GraphNode':
        """
        Convert a basic Node to a GraphNode.
        
        Args:
            node: Basic Node instance
            layer_id: ID of the layer this node belongs to
            
        Returns:
            A new GraphNode instance
        """
        node_id = UUID(node.id) if isinstance(node.id, str) else node.id
        return cls(
            id=node_id,
            name=node.name,
            node_type=node.type,
            description=node.description,
            properties=getattr(node, 'properties', {}),
            layer_id=UUID(node.layer_id) if node.layer_id and isinstance(node.layer_id, str) else node.layer_id,
            metadata={"type": "GraphNode", "index_fields": ["name"], "created_at": datetime.now().isoformat()}
        )
    
    def to_basic_node(self) -> Node:
        """
        Convert this GraphNode to a basic Node.
        
        Returns:
            A basic Node instance
        """
        return Node(
            id=str(self.id),
            name=self.name,
            type=self.node_type,
            description=self.description,
            properties=self.properties,
            layer_id=str(self.layer_id) if self.layer_id else None
        )


class GraphEdge(DataPoint):
    """
    An edge in a knowledge graph that extends DataPoint.
    
    Attributes:
        id: Unique identifier for the edge
        source_node_id: ID of the source node
        target_node_id: ID of the target node
        relationship_name: Name of the relationship
        properties: Additional properties of the edge
        layer_id: ID of the layer this edge belongs to
    """
    source_node_id: UUID
    target_node_id: UUID
    relationship_name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    layer_id: Optional[UUID] = None
    
    @classmethod
    def create(cls, source_node_id: UUID, target_node_id: UUID, 
               relationship_name: str, properties: Optional[Dict[str, Any]] = None,
               layer_id: Optional[UUID] = None) -> 'GraphEdge':
        """
        Create a new GraphEdge with a generated UUID.
        
        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            relationship_name: Name of the relationship
            properties: Additional properties of the edge
            layer_id: ID of the layer this edge belongs to
            
        Returns:
            A new GraphEdge instance
        """
        edge_id = uuid4()
        return cls(
            id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_name=relationship_name,
            properties=properties or {},
            layer_id=layer_id,
            metadata={"type": "GraphEdge", "index_fields": ["relationship_name"], "created_at": datetime.now().isoformat()}
        )
    
    @classmethod
    def from_basic_edge(cls, edge: Edge, layer_id: Optional[UUID] = None) -> 'GraphEdge':
        """
        Convert a basic Edge to a GraphEdge.
        
        Args:
            edge: Basic Edge instance
            layer_id: ID of the layer this edge belongs to
            
        Returns:
            A new GraphEdge instance
        """
        edge_id = uuid4()
        source_id = UUID(edge.source_node_id) if isinstance(edge.source_node_id, str) else edge.source_node_id
        target_id = UUID(edge.target_node_id) if isinstance(edge.target_node_id, str) else edge.target_node_id
        
        return cls(
            id=edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            relationship_name=edge.relationship_name,
            properties=getattr(edge, 'properties', {}),
            layer_id=UUID(edge.layer_id) if edge.layer_id and isinstance(edge.layer_id, str) else edge.layer_id,
            metadata={"type": "GraphEdge", "index_fields": ["relationship_name"], "created_at": datetime.now().isoformat()}
        )
    
    def to_basic_edge(self) -> Edge:
        """
        Convert this GraphEdge to a basic Edge.
        
        Returns:
            A basic Edge instance
        """
        return Edge(
            id=str(self.id),
            source_node_id=str(self.source_node_id),
            target_node_id=str(self.target_node_id),
            relationship_name=self.relationship_name,
            properties=self.properties,
            layer_id=str(self.layer_id) if self.layer_id else None
        )


class GraphLayer(DataPoint):
    """
    A layer in a layered knowledge graph that extends DataPoint.
    
    Attributes:
        id: Unique identifier for the layer
        name: Human-readable name of the layer
        description: Detailed description of the layer
        layer_type: Type of the layer (e.g., "base", "enrichment", "inference")
        parent_layers: IDs of parent layers this layer builds upon
        properties: Additional properties of the layer
    """
    name: str
    description: str
    layer_type: str = "default"
    parent_layers: List[UUID] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def create(cls, name: str, description: str, layer_type: str = "default",
               parent_layers: Optional[List[UUID]] = None, 
               properties: Optional[Dict[str, Any]] = None) -> 'GraphLayer':
        """
        Create a new GraphLayer with a generated UUID.
        
        Args:
            name: Human-readable name of the layer
            description: Detailed description of the layer
            layer_type: Type of the layer
            parent_layers: IDs of parent layers this layer builds upon
            properties: Additional properties of the layer
            
        Returns:
            A new GraphLayer instance
        """
        layer_id = uuid4()
        return cls(
            id=layer_id,
            name=name,
            description=description,
            layer_type=layer_type,
            parent_layers=parent_layers or [],
            properties=properties or {},
            metadata={"type": "GraphLayer", "index_fields": ["name"], "created_at": datetime.now().isoformat()}
        )
    
    @classmethod
    def from_basic_layer(cls, layer: Layer) -> 'GraphLayer':
        """
        Convert a basic Layer to a GraphLayer.
        
        Args:
            layer: Basic Layer instance
            
        Returns:
            A new GraphLayer instance
        """
        layer_id = UUID(layer.id) if isinstance(layer.id, str) else layer.id
        parent_layers = [UUID(pl) if isinstance(pl, str) else pl for pl in layer.parent_layers]
        
        return cls(
            id=layer_id,
            name=layer.name,
            description=layer.description,
            layer_type=layer.layer_type,
            parent_layers=parent_layers,
            properties=getattr(layer, 'properties', {}),
            metadata={"type": "GraphLayer", "index_fields": ["name"], "created_at": datetime.now().isoformat()}
        )
    
    def to_basic_layer(self) -> Layer:
        """
        Convert this GraphLayer to a basic Layer.
        
        Returns:
            A basic Layer instance
        """
        return Layer(
            id=str(self.id),
            name=self.name,
            description=self.description,
            layer_type=self.layer_type,
            parent_layers=[str(pl) for pl in self.parent_layers],
            properties=self.properties
        )


class LayeredKnowledgeGraphDP(DataPoint):
    """
    A layered knowledge graph that extends DataPoint.
    
    This class manages multiple layers of a knowledge graph, where each layer
    can build upon previous layers. It provides methods for adding and retrieving
    nodes and edges from specific layers, as well as extracting subgraphs.
    
    Attributes:
        id: Unique identifier for the graph
        name: Human-readable name of the graph
        description: Detailed description of the graph
        layers: List of layers in the graph
        nodes: Dictionary mapping node IDs to GraphNode instances
        edges: Dictionary mapping edge IDs to GraphEdge instances
        node_layer_map: Dictionary mapping node IDs to layer IDs
        edge_layer_map: Dictionary mapping edge IDs to layer IDs
    """
    name: str
    description: str
    layers: Dict[UUID, GraphLayer] = Field(default_factory=dict)
    nodes: Dict[UUID, GraphNode] = Field(default_factory=dict)
    edges: Dict[UUID, GraphEdge] = Field(default_factory=dict)
    node_layer_map: Dict[UUID, UUID] = Field(default_factory=dict)
    edge_layer_map: Dict[UUID, UUID] = Field(default_factory=dict)
    
    @classmethod
    def create_empty(cls, name: str, description: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> 'LayeredKnowledgeGraphDP':
        """
        Create a new empty layered knowledge graph.
        
        Args:
            name: Human-readable name of the graph
            description: Detailed description of the graph
            metadata: Additional metadata for the graph
            
        Returns:
            A new LayeredKnowledgeGraphDP instance
        """
        graph_id = uuid4()
        base_metadata = {
            "type": "LayeredKnowledgeGraphDP", 
            "index_fields": ["name"], 
            "created_at": datetime.now().isoformat()
        }
        
        # Merge with provided metadata if any
        if metadata:
            for key, value in metadata.items():
                if key not in ["type", "index_fields"]:
                    base_metadata[key] = value
        
        return cls(
            id=graph_id,
            name=name,
            description=description,
            layers={},
            nodes={},
            edges={},
            node_layer_map={},
            edge_layer_map={},
            metadata=base_metadata
        )
    
    @classmethod
    def from_basic_graph(cls, graph: KnowledgeGraph, layer_name: str = "Base Layer",
                        layer_description: str = "Converted from basic knowledge graph") -> 'LayeredKnowledgeGraphDP':
        """
        Convert a basic KnowledgeGraph to a LayeredKnowledgeGraphDP.
        
        Args:
            graph: Basic KnowledgeGraph instance
            layer_name: Name for the base layer
            layer_description: Description for the base layer
            
        Returns:
            A new LayeredKnowledgeGraphDP instance
        """
        layered_graph = cls.create_empty(
            name=getattr(graph, 'name', "Converted Graph"),
            description=getattr(graph, 'description', "Converted from basic knowledge graph")
        )
        
        # Create a base layer
        base_layer = GraphLayer.create(
            name=layer_name,
            description=layer_description,
            layer_type="base"
        )
        layered_graph.add_layer(base_layer)
        
        # Add nodes and edges to the base layer
        for node in graph.nodes:
            graph_node = GraphNode.from_basic_node(node, base_layer.id)
            layered_graph.add_node(graph_node, base_layer.id)
        
        for edge in graph.edges:
            # Convert string IDs to UUIDs if needed
            source_id = UUID(edge.source_node_id) if isinstance(edge.source_node_id, str) else edge.source_node_id
            target_id = UUID(edge.target_node_id) if isinstance(edge.target_node_id, str) else edge.target_node_id
            
            # Find the corresponding GraphNode objects
            source_node = next((n for n in layered_graph.nodes.values() if n.id == source_id), None)
            target_node = next((n for n in layered_graph.nodes.values() if n.id == target_id), None)
            
            if source_node and target_node:
                graph_edge = GraphEdge.create(
                    source_node_id=source_node.id,
                    target_node_id=target_node.id,
                    relationship_name=edge.relationship_name,
                    properties=getattr(edge, 'properties', {}),
                    layer_id=base_layer.id
                )
                layered_graph.add_edge(graph_edge, base_layer.id)
        
        return layered_graph
    
    def add_layer(self, layer: GraphLayer) -> None:
        """
        Add a layer to the layered graph.
        
        Args:
            layer: The layer to add
            
        Raises:
            ValueError: If a parent layer does not exist
        """
        # Check if parent layers exist
        for parent_id in layer.parent_layers:
            if parent_id not in self.layers:
                raise ValueError(f"Parent layer with ID {parent_id} does not exist")
        
        self.layers[layer.id] = layer
    
    def add_node(self, node: GraphNode, layer_id: UUID) -> None:
        """
        Add a node to a specific layer.
        
        Args:
            node: The node to add
            layer_id: ID of the layer to add the node to
            
        Raises:
            ValueError: If the layer does not exist
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer with ID {layer_id} does not exist")
        
        # Update the node's layer_id
        node.layer_id = layer_id
        
        # Add to the graph
        self.nodes[node.id] = node
        self.node_layer_map[node.id] = layer_id
    
    def add_edge(self, edge: GraphEdge, layer_id: UUID) -> None:
        """
        Add an edge to a specific layer.
        
        Args:
            edge: The edge to add
            layer_id: ID of the layer to add the edge to
            
        Raises:
            ValueError: If the layer does not exist or if source/target nodes don't exist
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer with ID {layer_id} does not exist")
        
        # Check if source and target nodes exist
        if edge.source_node_id not in self.nodes:
            raise ValueError(f"Source node with ID {edge.source_node_id} does not exist")
        if edge.target_node_id not in self.nodes:
            raise ValueError(f"Target node with ID {edge.target_node_id} does not exist")
        
        # Update the edge's layer_id
        edge.layer_id = layer_id
        
        # Add to the graph
        self.edges[edge.id] = edge
        self.edge_layer_map[edge.id] = layer_id
    
    def get_layer(self, layer_id: UUID) -> GraphLayer:
        """
        Get a layer by its ID.
        
        Args:
            layer_id: ID of the layer to get
            
        Returns:
            The requested layer
            
        Raises:
            ValueError: If the layer does not exist
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer with ID {layer_id} does not exist")
        return self.layers[layer_id]
    
    def get_layer_nodes(self, layer_id: UUID) -> List[GraphNode]:
        """
        Get all nodes in a specific layer.
        
        Args:
            layer_id: ID of the layer to get nodes from
            
        Returns:
            List of nodes in the layer
            
        Raises:
            ValueError: If the layer does not exist
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer with ID {layer_id} does not exist")
        
        return [node for node in self.nodes.values() if node.layer_id == layer_id]
    
    def get_layer_edges(self, layer_id: UUID) -> List[GraphEdge]:
        """
        Get all edges in a specific layer.
        
        Args:
            layer_id: ID of the layer to get edges from
            
        Returns:
            List of edges in the layer
            
        Raises:
            ValueError: If the layer does not exist
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer with ID {layer_id} does not exist")
        
        return [edge for edge in self.edges.values() if edge.layer_id == layer_id]
    
    def collect_parent_layers(self, layer_id: UUID) -> List[UUID]:
        """
        Collect all parent layers of a given layer.
        
        Args:
            layer_id: ID of the layer to collect parents for
            
        Returns:
            List of parent layer IDs
            
        Raises:
            ValueError: If the layer does not exist
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer with ID {layer_id} does not exist")
        
        layer = self.layers[layer_id]
        parent_ids = []
        
        # Recursively collect parent layers
        for parent_id in layer.parent_layers:
            parent_ids.append(parent_id)
            parent_ids.extend(self.collect_parent_layers(parent_id))
        
        # Remove duplicates while preserving order
        unique_parents = []
        for pid in parent_ids:
            if pid not in unique_parents:
                unique_parents.append(pid)
        
        return unique_parents
    
    def _get_layer(self, layer_id: UUID) -> GraphLayer:
        """
        Get a layer by its ID (internal method).
        
        Args:
            layer_id: ID of the layer to get
            
        Returns:
            The requested layer
            
        Raises:
            ValueError: If the layer does not exist
        """
        if layer_id not in self.layers:
            raise ValueError(f"Layer with ID {layer_id} does not exist")
        return self.layers[layer_id]
    
    def get_layer_graph(self, layer_id: UUID) -> KnowledgeGraph:
        """
        Get a knowledge graph containing only nodes and edges from the specified layer.
        
        Args:
            layer_id: ID of the layer to extract
            
        Returns:
            Knowledge graph with nodes and edges from the layer
            
        Raises:
            ValueError: If the layer does not exist
        """
        layer = self._get_layer(layer_id)
        
        # Get nodes and edges from this layer
        layer_nodes = self.get_layer_nodes(layer_id)
        layer_edges = self.get_layer_edges(layer_id)
        
        # Convert to basic nodes and edges
        nodes = [Node(
            id=str(node.id),
            name=node.name,
            type=node.node_type,
            description=node.description,
            properties=node.properties,
            layer_id=str(node.layer_id) if node.layer_id else None
        ) for node in layer_nodes]
        
        edges = [Edge(
            id=str(edge.id),
            source_node_id=str(edge.source_node_id),
            target_node_id=str(edge.target_node_id),
            relationship_name=edge.relationship_name,
            properties=edge.properties,
            layer_id=str(edge.layer_id) if edge.layer_id else None
        ) for edge in layer_edges]
        
        # Create and return the knowledge graph
        return KnowledgeGraph(
            name=f"Layer Graph for {layer.name}",
            description=f"Graph containing only nodes and edges from {layer.name}",
            nodes=nodes,
            edges=edges
        )
    
    def get_cumulative_layer_graph(self, layer_id: UUID) -> KnowledgeGraph:
        """
        Get a knowledge graph containing nodes and edges from the specified layer and all its parent layers.
        
        Args:
            layer_id: ID of the layer to extract (and its parents)
            
        Returns:
            Knowledge graph with nodes and edges from the layer and all parent layers
            
        Raises:
            ValueError: If the layer does not exist
        """
        # Ensure the layer exists
        layer = self._get_layer(layer_id)
        
        # Get all parent layers to include
        layer_ids = self.collect_parent_layers(layer_id)
        layer_ids.append(layer_id)
        
        # Collect all nodes and edges from layers
        all_nodes = []
        all_edges = []
        node_ids = set()
        edge_ids = set()
        
        for lid in layer_ids:
            # Add nodes from this layer
            for node in self.get_layer_nodes(lid):
                if node.id not in node_ids:
                    all_nodes.append(Node(
                        id=str(node.id),  # Convert UUID to string
                        name=node.name,
                        type=node.node_type,
                        description=node.description,
                        properties=node.properties,
                        layer_id=str(node.layer_id) if node.layer_id else None
                    ))
                    node_ids.add(node.id)
            
            # Add edges from this layer
            for edge in self.get_layer_edges(lid):
                if edge.id not in edge_ids:
                    all_edges.append(Edge(
                        id=str(edge.id),  # Convert UUID to string
                        source_node_id=str(edge.source_node_id),  # Convert UUID to string
                        target_node_id=str(edge.target_node_id),  # Convert UUID to string
                        relationship_name=edge.relationship_name,
                        properties=edge.properties,
                        layer_id=str(edge.layer_id) if edge.layer_id else None
                    ))
                    edge_ids.add(edge.id)
        
        # Create and return the knowledge graph
        return KnowledgeGraph(
            name=f"Cumulative Graph for {layer.name}",
            description=f"Cumulative graph including {layer.name} and all parent layers",
            nodes=all_nodes,
            edges=all_edges
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the layered knowledge graph to a dictionary.
        
        Returns:
            Dictionary representation of the graph
        """
        # Helper function to ensure metadata has required fields
        def ensure_metadata(data):
            if isinstance(data, dict) and 'metadata' in data:
                if 'type' not in data['metadata']:
                    data['metadata']['type'] = data.get('type', 'Unknown')
                if 'index_fields' not in data['metadata']:
                    data['metadata']['index_fields'] = []
            return data
        
        # Convert layers to dictionaries with proper metadata
        layers_dict = {}
        for lid, layer in self.layers.items():
            layer_dict = layer.model_dump()
            layer_dict = ensure_metadata(layer_dict)
            layers_dict[str(lid)] = layer_dict
        
        # Convert nodes to dictionaries with proper metadata
        nodes_dict = {}
        for nid, node in self.nodes.items():
            node_dict = node.model_dump()
            node_dict = ensure_metadata(node_dict)
            nodes_dict[str(nid)] = node_dict
        
        # Convert edges to dictionaries with proper metadata
        edges_dict = {}
        for eid, edge in self.edges.items():
            edge_dict = edge.model_dump()
            edge_dict = ensure_metadata(edge_dict)
            edges_dict[str(eid)] = edge_dict
        
        # Ensure our own metadata has required fields
        metadata = self.metadata.copy() if self.metadata else {}
        if 'type' not in metadata:
            metadata['type'] = self.__class__.__name__
        if 'index_fields' not in metadata:
            metadata['index_fields'] = ['name']
        
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "layers": layers_dict,
            "nodes": nodes_dict,
            "edges": edges_dict,
            "node_layer_map": {str(nid): str(lid) for nid, lid in self.node_layer_map.items()},
            "edge_layer_map": {str(eid): str(lid) for eid, lid in self.edge_layer_map.items()},
            "metadata": metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayeredKnowledgeGraphDP':
        """
        Create a layered knowledge graph from a dictionary.
        
        Args:
            data: Dictionary representation of the graph
            
        Returns:
            A new LayeredKnowledgeGraphDP instance
        """
        # Ensure metadata has required fields
        metadata = data.get("metadata", {}).copy()
        if 'type' not in metadata:
            metadata['type'] = cls.__name__
        if 'index_fields' not in metadata:
            metadata['index_fields'] = ['name']
        
        graph = cls(
            id=UUID(data["id"]),
            name=data["name"],
            description=data["description"],
            layers={},
            nodes={},
            edges={},
            node_layer_map={},
            edge_layer_map={},
            metadata=metadata
        )
        
        # Helper function to ensure metadata in model data
        def prepare_model_data(model_data):
            if 'metadata' not in model_data:
                model_data['metadata'] = {'type': model_data.get('type', 'Unknown'), 'index_fields': []}
            elif 'type' not in model_data['metadata']:
                model_data['metadata']['type'] = model_data.get('type', 'Unknown')
            elif 'index_fields' not in model_data['metadata']:
                model_data['metadata']['index_fields'] = []
            return model_data
        
        # Load layers
        for lid_str, layer_data in data["layers"].items():
            lid = UUID(lid_str)
            layer_data = prepare_model_data(layer_data)
            layer = GraphLayer.model_validate(layer_data)
            graph.layers[lid] = layer
        
        # Load nodes
        for nid_str, node_data in data["nodes"].items():
            nid = UUID(nid_str)
            node_data = prepare_model_data(node_data)
            node = GraphNode.model_validate(node_data)
            graph.nodes[nid] = node
        
        # Load edges
        for eid_str, edge_data in data["edges"].items():
            eid = UUID(eid_str)
            edge_data = prepare_model_data(edge_data)
            edge = GraphEdge.model_validate(edge_data)
            graph.edges[eid] = edge
        
        # Load maps
        for nid_str, lid_str in data["node_layer_map"].items():
            graph.node_layer_map[UUID(nid_str)] = UUID(lid_str)
        
        for eid_str, lid_str in data["edge_layer_map"].items():
            graph.edge_layer_map[UUID(eid_str)] = UUID(lid_str)
        
        return graph
    
    def to_json(self) -> str:
        """
        Convert the layered knowledge graph to a JSON string.
        
        Returns:
            JSON string representation of the graph
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LayeredKnowledgeGraphDP':
        """
        Create a layered knowledge graph from a JSON string.
        
        Args:
            json_str: JSON string representation of the graph
            
        Returns:
            A new LayeredKnowledgeGraphDP instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def add_node_to_layer(self, node: GraphNode, layer_id: UUID) -> None:
        """
        Add a node to a specific layer.
        
        Args:
            node: The node to add
            layer_id: ID of the layer to add the node to
            
        Raises:
            ValueError: If the layer does not exist
        """
        self.add_node(node, layer_id)
    
    def add_edge_to_layer(self, edge: GraphEdge, layer_id: UUID) -> None:
        """
        Add an edge to a specific layer.
        
        Args:
            edge: The edge to add
            layer_id: ID of the layer to add the edge to
            
        Raises:
            ValueError: If the layer does not exist or if source/target nodes don't exist
        """
        self.add_edge(edge, layer_id)