"""
Layered Knowledge Graph Builder for Cognee.

This module provides utilities for building and managing layered knowledge graphs.
"""

import uuid
import asyncio
import os
from typing import Dict, List, Optional, Any, Union

from cognee.shared.data_models import (
    KnowledgeGraph,
    Layer,
    Node,
    Edge
)
from cognee.modules.graph.simplified_layered_graph import LayeredKnowledgeGraph
from cognee.infrastructure.databases.graph.networkx.adapter import NetworkXAdapter
import networkx as nx

import logging
logger = logging.getLogger(__name__)


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
            id=uuid.uuid4(),
            name=name,
            description=description
        )
        
        # Set up an adapter for the layered graph
        db_directory = os.path.join(os.getcwd(), ".cognee_system", "databases")
        os.makedirs(db_directory, exist_ok=True)
        db_file = os.path.join(db_directory, "temp_layered_graph.pkl")
        
        # Create a NetworkXAdapter with an initialized graph
        nx_adapter = NetworkXAdapter(filename=db_file)
        if not hasattr(nx_adapter, 'graph') or nx_adapter.graph is None:
            nx_adapter.graph = nx.MultiDiGraph()
            
        # Create a wrapper adapter that provides the expected _graph_db attribute
        class AdapterWrapper:
            def __init__(self, adapter):
                self._graph_db = adapter
                self._graph_db_initialized = True
        
        # Set the adapter on the layered graph
        self.adapter = AdapterWrapper(nx_adapter)
        self.layered_graph.set_adapter(self.adapter)
        
        # Initialize the graph in the database
        try:
            # If we're in an async context, await the initialization
            import inspect
            if inspect.iscoroutinefunction(inspect.currentframe().f_back.f_code):
                asyncio.create_task(self.layered_graph.initialize())
            else:
                # For synchronous context, use nest_asyncio
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.layered_graph.initialize())
        except Exception as e:
            logger.warning(f"Error initializing layered graph: {e}")
        
        # Keep track of layers for builder's internal state
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
        
        # Create layer for local tracking
        layer = Layer(
            id=layer_id,
            name=name,
            description=description,
            layer_type=layer_type,
            parent_layers=parent_layers,
            properties=properties
        )
        
        # Convert parent_layers strings to UUIDs for the new API
        parent_layer_uuids = [uuid.UUID(p) for p in parent_layers] if parent_layers else None
        
        # Use the new add_layer API that accepts individual parameters
        try:
            # Import nest_asyncio for handling async in sync context
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            
            # Call add_layer and wait for the result
            layer_obj = loop.run_until_complete(
                self.layered_graph.add_layer(
                    name=name,
                    description=description,
                    layer_type=layer_type,
                    parent_layers=parent_layer_uuids
                )
            )
            
            # Store the layer in local tracking dictionaries
            self.layers[layer_id] = layer
            
            # Initialize empty layer node and edge lists
            self.layer_nodes[layer_id] = []
            self.layer_edges[layer_id] = []
            
        except Exception as e:
            logger.error(f"Error adding layer: {e}")
            # Fall back to local tracking only
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
        
        # Add node to layer using the async add_node method
        try:
            node_properties = {}
            if properties:
                node_properties = properties
            if description:
                node_properties["description"] = description
                
            # Create a Pydantic-compatible properties dictionary
            # Import nest_asyncio for handling async in sync context
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            
            # Call add_node and wait for the result
            graph_node = loop.run_until_complete(
                self.layered_graph.add_node(
                    name=name,
                    node_type=node_type,
                    properties=node_properties,
                    metadata={"id": node_id},
                    layer_id=uuid.UUID(layer_id)
                )
            )
        except Exception as e:
            logger.error(f"Error adding node to layer: {e}")
            # Fall back to local tracking only
            pass
        
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
        
        # Add edge to layer using the async add_edge method
        try:
            # Import nest_asyncio for handling async in sync context
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            
            # Call add_edge and wait for the result
            graph_edge = loop.run_until_complete(
                self.layered_graph.add_edge(
                    source_id=uuid.UUID(source_node_id),
                    target_id=uuid.UUID(target_node_id),
                    edge_type=relationship_name,
                    properties=properties or {},
                    layer_id=uuid.UUID(layer_id)
                )
            )
        except Exception as e:
            logger.error(f"Error adding edge to layer: {e}")
            # Fall back to local tracking only
            pass
        
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
        
        # Import nest_asyncio for handling async in sync context
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        
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
            
            # Add to layered graph using async method
            try:
                node_properties = node.properties or {}
                if node.description:
                    node_properties["description"] = node.description
                
                # Call add_node and wait for the result
                graph_node = loop.run_until_complete(
                    self.layered_graph.add_node(
                        name=node.name,
                        node_type=node.type,
                        properties=node_properties,
                        metadata={"id": new_id},
                        layer_id=uuid.UUID(layer_id)
                    )
                )
            except Exception as e:
                logger.error(f"Error adding node to layer: {e}")
                # Fall back to local tracking only
                pass
            
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
            
            # Add to layered graph using async method
            try:
                # Call add_edge and wait for the result
                graph_edge = loop.run_until_complete(
                    self.layered_graph.add_edge(
                        source_id=uuid.UUID(new_source_id),
                        target_id=uuid.UUID(new_target_id),
                        edge_type=edge.relationship_name,
                        properties=edge.properties or {},
                        layer_id=uuid.UUID(layer_id)
                    )
                )
            except Exception as e:
                logger.error(f"Error adding edge to layer: {e}")
                # Fall back to local tracking only
                pass
            
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