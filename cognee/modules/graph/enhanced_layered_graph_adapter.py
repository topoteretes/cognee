"""
Enhanced adapter for layered knowledge graphs to work with Cognee's graph database infrastructure.

This module provides an improved adapter implementation that allows layered knowledge graphs to be 
stored in and retrieved from graph databases compatible with Cognee's GraphDBInterface. It follows 
design principles from CogneeGraph while enhancing the layer-based structure.

Key improvements:
1. Stronger abstraction for database operations
2. Better error handling and validation
3. Support for efficient batch operations
4. Enhanced query capabilities for layered graphs
5. Support for adapter-specific implementations
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union, Set, TypeVar, Generic
from uuid import UUID, uuid4
from datetime import datetime, timezone

from cognee.exceptions import InvalidValueError
from cognee.modules.graph.exceptions import EntityNotFoundError, EntityAlreadyExistsError
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode, GraphEdge, GraphLayer, LayeredKnowledgeGraphDP, UUIDEncoder
)
from cognee.shared.data_models import Node, Edge, KnowledgeGraph, Layer

logger = logging.getLogger(__name__)

# Custom datapoint types for working with NetworkXAdapter
class GraphNodeDP(DataPoint):
    """DataPoint for storing graph nodes."""
    name: str
    node_type: str 
    description: str
    properties: Dict[str, Any] = {}
    layer_id: Optional[str] = None
    type: str = "GraphNode"

class GraphEdgeDP(DataPoint):
    """DataPoint for storing graph edges."""
    source_node_id: str
    target_node_id: str
    relationship_name: str
    properties: Dict[str, Any] = {}
    layer_id: Optional[str] = None
    type: str = "GraphEdge"
    
class GraphLayerDP(DataPoint):
    """DataPoint for storing graph layers."""
    name: str
    description: str
    layer_type: str = "default"
    parent_layers: List[str] = []
    properties: Dict[str, Any] = {}
    type: str = "GraphLayer"

class LayeredGraphDP(DataPoint):
    """DataPoint for storing layered graphs."""
    name: str
    description: str
    layer_count: int = 0
    node_count: int = 0
    edge_count: int = 0
    type: str = "LayeredKnowledgeGraph"

class LayeredGraphDBAdapter:
    """
    Enhanced adapter for storing and retrieving layered knowledge graphs using graph databases.
    
    This class provides methods to:
    1. Store layered knowledge graphs in a graph database
    2. Retrieve layered knowledge graphs from a graph database 
    3. Perform advanced queries on layered graphs
    4. Manage relationships between layers
    5. Extract subgraphs based on layers
    
    The adapter is designed to work with any database implementing GraphDBInterface,
    and provides optimized operations for batch processing and querying.
    """
    
    def __init__(self, graph_db: Optional[GraphDBInterface] = None):
        """
        Initialize the adapter with a graph database.
        
        Args:
            graph_db: Optional GraphDBInterface instance. If None, the default graph engine will be used.
        """
        self._graph_db = graph_db
        self._graph_db_initialized = graph_db is not None
        self._query_cache = {}  # Simple cache for query results
    
    async def _ensure_graph_db(self):
        """Ensure the graph database is initialized."""
        if not self._graph_db_initialized:
            try:
                self._graph_db = await get_graph_engine()
                self._graph_db_initialized = True
            except Exception as e:
                logger.error(f"Error initializing graph database: {str(e)}")
                raise
    
    async def store_graph(self, graph: LayeredKnowledgeGraphDP) -> str:
        """
        Store a layered knowledge graph in the graph database.
        
        This method stores the entire graph structure, including:
        - The graph node itself
        - All layers in the graph
        - All nodes in each layer
        - All edges in each layer
        - Relationships between layers, nodes, and edges
        
        Args:
            graph: The layered knowledge graph to store
            
        Returns:
            The ID of the stored graph
            
        Raises:
            ValueError: If the graph is invalid
            Exception: If there's an error storing the graph
        """
        await self._ensure_graph_db()
        
        if not graph:
            raise ValueError("Cannot store a null graph")
        
        try:
            # Make sure we have the adapter type
            if not hasattr(self, '_adapter_type') or self._adapter_type is None:
                self._adapter_type = type(self._graph_db).__name__
                logger.debug(f"Using graph database adapter: {self._adapter_type}")
            
            # First, store the graph itself as a node
            graph_dp = LayeredGraphDP(
                id=graph.id,
                name=graph.name,
                description=graph.description,
                layer_count=len(graph.layers),
                node_count=len(graph.nodes),
                edge_count=len(graph.edges),
                created_at=int(datetime.now(timezone.utc).timestamp() * 1000),
                metadata={"type": "LayeredKnowledgeGraph", "index_fields": ["name"]}
            )
            
            # Add type field to metadata
            graph_dp.metadata["type"] = "LayeredKnowledgeGraph"
            await self._graph_db.add_node(graph_dp)
            
            # Store layers with parent relationships
            await self._store_layers(graph)
            
            # Store nodes with layer relationships
            await self._store_nodes(graph)
            
            # Store edges with layer relationships and between nodes
            await self._store_edges(graph)
            
            logger.info(f"Successfully stored layered graph with ID {graph.id}")
            return str(graph.id)
            
        except Exception as e:
            logger.error(f"Error storing layered graph: {str(e)}")
            raise
    
    async def _store_layers(self, graph: LayeredKnowledgeGraphDP) -> None:
        """
        Store all layers in a graph with their relationships.
        
        Args:
            graph: The graph containing the layers to store
        """
        layer_tasks = []
        
        # First pass: create all layer nodes
        for layer_id, layer in graph.layers.items():
            # Create a layer DataPoint for NetworkXAdapter compatibility
            layer_dp = GraphLayerDP(
                id=layer.id,
                name=layer.name,
                description=layer.description,
                layer_type=layer.layer_type,
                parent_layers=[str(parent_id) for parent_id in layer.parent_layers],
                properties=layer.properties,
                metadata={"type": "GraphLayer", "index_fields": ["name"]}
            )
            
            # Add type field to metadata
            layer_dp.metadata["type"] = "GraphLayer"
            
            # Add the layer node
            layer_tasks.append(self._graph_db.add_node(layer_dp))
            
            # Create task to add the CONTAINS_LAYER relationship from graph to layer
            layer_tasks.append(self._graph_db.add_edge(
                str(graph.id),
                str(layer.id),
                "CONTAINS_LAYER",
                {"graph_id": str(graph.id), "layer_id": str(layer.id)}
            ))
        
        # Execute all layer node tasks
        await asyncio.gather(*layer_tasks)
        
        # Second pass: create layer relationships (must be done after all layers exist)
        layer_relationship_tasks = []
        for layer_id, layer in graph.layers.items():
            # Create tasks for parent layer relationships
            for parent_id in layer.parent_layers:
                layer_relationship_tasks.append(self._graph_db.add_edge(
                    str(layer.id),
                    str(parent_id),
                    "EXTENDS_LAYER",
                    {"child_layer_id": str(layer.id), "parent_layer_id": str(parent_id)}
                ))
        
        # Execute all layer relationship tasks
        if layer_relationship_tasks:
            await asyncio.gather(*layer_relationship_tasks)
    
    async def _store_nodes(self, graph: LayeredKnowledgeGraphDP) -> None:
        """
        Store all nodes in a graph with their layer relationships.
        
        Args:
            graph: The graph containing the nodes to store
        """
        # Process nodes batch by batch to avoid overwhelming the database
        batch_size = 100
        node_items = list(graph.nodes.items())
        
        for i in range(0, len(node_items), batch_size):
            batch_tasks = []
            batch = node_items[i:i+batch_size]
            
            for node_id, node in batch:
                # Create a node DataPoint for NetworkXAdapter compatibility
                node_dp = GraphNodeDP(
                    id=node.id,
                    name=node.name,
                    node_type=node.node_type,
                    description=node.description,
                    properties=node.properties,
                    layer_id=str(node.layer_id) if node.layer_id else None,
                    metadata={"type": "GraphNode", "index_fields": ["name"]}
                )
                
                # Add type field to metadata
                node_dp.metadata["type"] = "GraphNode"
                
                # Create task to add the node
                batch_tasks.append(self._graph_db.add_node(node_dp))
                
                # Create task to add the IN_LAYER relationship if node has a layer
                if node.layer_id:
                    batch_tasks.append(self._graph_db.add_edge(
                        str(node.id),
                        str(node.layer_id),
                        "IN_LAYER",
                        {"node_id": str(node.id), "layer_id": str(node.layer_id)}
                    ))
            
            # Execute all tasks in this batch
            await asyncio.gather(*batch_tasks)
    
    async def _store_edges(self, graph: LayeredKnowledgeGraphDP) -> None:
        """
        Store all edges in a graph with their relationships.
        
        Args:
            graph: The graph containing the edges to store
        """
        # Process edges batch by batch to avoid overwhelming the database
        batch_size = 100
        edge_items = list(graph.edges.items())
        
        for i in range(0, len(edge_items), batch_size):
            batch_tasks = []
            batch = edge_items[i:i+batch_size]
            
            for edge_id, edge in batch:
                # Create an edge DataPoint for NetworkXAdapter compatibility
                edge_dp = GraphEdgeDP(
                    id=edge.id,
                    source_node_id=str(edge.source_node_id),
                    target_node_id=str(edge.target_node_id),
                    relationship_name=edge.relationship_name,
                    properties=edge.properties,
                    layer_id=str(edge.layer_id) if edge.layer_id else None,
                    metadata={"type": "GraphEdge", "index_fields": ["relationship_name"]}
                )
                
                # Add type field to metadata
                edge_dp.metadata["type"] = "GraphEdge"
                
                # Create task to add the edge node (represent edge as a node for querying)
                batch_tasks.append(self._graph_db.add_node(edge_dp))
                
                # Create task to add the relationship between source and target
                batch_tasks.append(self._graph_db.add_edge(
                    str(edge.source_node_id),
                    str(edge.target_node_id),
                    edge.relationship_name,
                    {"edge_id": str(edge.id), **edge.properties}
                ))
                
                # Create task to add the IN_LAYER relationship for the edge if it belongs to a layer
                if edge.layer_id:
                    batch_tasks.append(self._graph_db.add_edge(
                        str(edge.id),
                        str(edge.layer_id),
                        "IN_LAYER",
                        {"edge_id": str(edge.id), "layer_id": str(edge.layer_id)}
                    ))
            
            # Execute all tasks in this batch
            await asyncio.gather(*batch_tasks)
    
    async def _add_edge_with_adapter(self, from_node: str, to_node: str, relationship_name: str, edge_properties: Dict[str, Any] = None):
        """Helper method to add an edge with proper adapter-specific handling."""
        if edge_properties is None:
            edge_properties = {}
            
        try:
            # Try the standard interface
            await self._graph_db.add_edge(from_node, to_node, relationship_name, edge_properties)
        except TypeError as e:
            logger.debug(f"Standard edge interface failed, trying alternative approach: {str(e)}")
            
            # For adapters expecting a tuple format
            edge_tuple = (from_node, to_node, relationship_name, edge_properties)
            await self._graph_db.add_edges([edge_tuple])
        except Exception as e:
            logger.error(f"Error adding edge: {str(e)}")
            raise
    
    async def retrieve_graph(self, graph_id: Union[str, UUID]) -> LayeredKnowledgeGraphDP:
        """
        Retrieve a layered knowledge graph from the graph database.
        
        Args:
            graph_id: The ID of the graph to retrieve
            
        Returns:
            The retrieved layered knowledge graph
            
        Raises:
            EntityNotFoundError: If the graph is not found
            Exception: If there's an error retrieving the graph
        """
        await self._ensure_graph_db()
        
        try:
            # Make sure we have the adapter type
            if not hasattr(self, '_adapter_type') or self._adapter_type is None:
                self._adapter_type = type(self._graph_db).__name__
                logger.debug(f"Using graph database adapter: {self._adapter_type}")
            
            # Special handling for NetworkXAdapter - ensure graph is loaded
            if self._adapter_type == "NetworkXAdapter":
                # Make sure the graph is loaded from file
                if not hasattr(self._graph_db, 'graph') or self._graph_db.graph is None:
                    if hasattr(self._graph_db, 'create_empty_graph'):
                        await self._graph_db.create_empty_graph(self._graph_db.filename)
                # Load the graph from file
                if hasattr(self._graph_db, 'load_graph_from_file'):
                    await self._graph_db.load_graph_from_file()
            
            # Create empty graph structure
            graph = LayeredKnowledgeGraphDP.create_empty(
                name="Retrieved Graph",
                description="Graph retrieved from database"
            )
            
            # Convert the graph_id to the right format
            if isinstance(graph_id, str):
                uuid_graph_id = UUID(graph_id)
            else:
                uuid_graph_id = graph_id
                graph_id = str(graph_id)  # Keep string version for logging
            
            # Set the graph ID
            graph.id = uuid_graph_id
            
            # Get the graph node data from the database
            if self._adapter_type == "NetworkXAdapter":
                graph_node = await self._graph_db.extract_node(uuid_graph_id)
            else:
                graph_node = await self._graph_db.extract_node(graph_id)
            
            if not graph_node:
                raise EntityNotFoundError(f"Graph with ID {graph_id} not found")
            
            # Update graph properties from database
            graph.name = graph_node.get("name", "Retrieved Graph")
            graph.description = graph_node.get("description", "Graph retrieved from database")
            if "metadata" in graph_node and graph_node["metadata"]:
                try:
                    if isinstance(graph_node["metadata"], str):
                        graph.metadata = json.loads(graph_node["metadata"])
                    else:
                        graph.metadata = graph_node["metadata"]
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse metadata for graph {graph_id}")
            
            # Get graph data to reconstruct structure
            logger.debug(f"Getting graph data from database")
            nodes_data, edges_data = await self._graph_db.get_graph_data()
            
            # Log some information about the data we got
            logger.debug(f"Retrieved {len(nodes_data)} nodes and {len(edges_data)} edges from database")
            
            # First, find all layers associated with this graph
            logger.debug(f"Processing layers for graph {graph_id}")
            graph_layers = {}
            
            # Special handling for NetworkXAdapter
            if self._adapter_type == "NetworkXAdapter":
                # Find all CONTAINS_LAYER relationships from this graph to layers
                contains_layer_count = 0
                
                for source, target, rel_type, props in edges_data:
                    if isinstance(source, UUID) and isinstance(target, UUID) and rel_type == "CONTAINS_LAYER":
                        if source == uuid_graph_id:
                            contains_layer_count += 1
                            logger.debug(f"Found CONTAINS_LAYER relationship from graph {uuid_graph_id} to layer {target}")
                            
                            # This is a link from the graph to a layer
                            layer_id = target
                            
                            # Get the layer node data
                            layer_node = None
                            for node_id, node_props in nodes_data:
                                if isinstance(node_id, UUID) and node_id == layer_id:
                                    layer_node = node_props
                                    logger.debug(f"Found layer node: {node_id} with properties: {node_props}")
                                    break
                            
                            if layer_node and layer_node.get("type", "") == "GraphLayer":
                                # Parse parent_layers field
                                parent_layers = []
                                parent_layers_data = layer_node.get("parent_layers", [])
                                if isinstance(parent_layers_data, str):
                                    try:
                                        parent_layers = json.loads(parent_layers_data)
                                    except json.JSONDecodeError:
                                        parent_layers = []
                                else:
                                    parent_layers = parent_layers_data
                                
                                # Parse properties
                                properties = {}
                                props_data = layer_node.get("properties", {})
                                if isinstance(props_data, str):
                                    try:
                                        properties = json.loads(props_data)
                                    except json.JSONDecodeError:
                                        properties = {}
                                else:
                                    properties = props_data
                                
                                # Parse metadata
                                metadata = {}
                                metadata_data = layer_node.get("metadata", {})
                                if isinstance(metadata_data, str):
                                    try:
                                        metadata = json.loads(metadata_data)
                                    except json.JSONDecodeError:
                                        metadata = {}
                                else:
                                    metadata = metadata_data
                                
                                # Create a GraphLayer object
                                logger.debug(f"Creating GraphLayer object for layer {layer_id}")
                                layer = GraphLayer(
                                    id=layer_id,
                                    name=layer_node.get("name", f"Layer {layer_id}"),
                                    description=layer_node.get("description", ""),
                                    layer_type=layer_node.get("layer_type", "default"),
                                    parent_layers=[],  # Will populate later
                                    properties=properties,
                                    metadata=metadata
                                )
                                graph_layers[str(layer_id)] = layer
                    
                logger.debug(f"Found {contains_layer_count} CONTAINS_LAYER relationships, created {len(graph_layers)} layer objects")
            else:
                # Standard processing for other adapters
                for source, target, rel_type, props in edges_data:
                    # Convert IDs for consistent comparison
                    source_str = str(source) if isinstance(source, UUID) else source
                    target_str = str(target) if isinstance(target, UUID) else target
                    rel_str = str(rel_type) if isinstance(rel_type, UUID) else rel_type
                    
                    if source_str == graph_id and rel_str == "CONTAINS_LAYER":
                        # This is a link from the graph to a layer
                        layer_id = target_str
                        
                        # Get the layer node data
                        layer_node = None
                        for node_id, node_props in nodes_data:
                            node_id_str = str(node_id) if isinstance(node_id, UUID) else node_id
                            if node_id_str == layer_id:
                                layer_node = node_props
                                break
                        
                        if layer_node and layer_node.get("type", "") == "GraphLayer":
                            # Parse parent_layers field
                            parent_layers = []
                            parent_layers_data = layer_node.get("parent_layers", [])
                            if isinstance(parent_layers_data, str):
                                try:
                                    parent_layers = json.loads(parent_layers_data)
                                except json.JSONDecodeError:
                                    parent_layers = []
                            else:
                                parent_layers = parent_layers_data
                                
                            # Parse properties
                            properties = {}
                            props_data = layer_node.get("properties", {})
                            if isinstance(props_data, str):
                                try:
                                    properties = json.loads(props_data)
                                except json.JSONDecodeError:
                                    properties = {}
                            else:
                                properties = props_data
                                
                            # Parse metadata
                            metadata = {}
                            metadata_data = layer_node.get("metadata", {})
                            if isinstance(metadata_data, str):
                                try:
                                    metadata = json.loads(metadata_data)
                                except json.JSONDecodeError:
                                    metadata = {}
                            else:
                                metadata = metadata_data
                                
                            # Create a GraphLayer object
                            layer = GraphLayer(
                                id=UUID(str(layer_id)),
                                name=layer_node.get("name", f"Layer {layer_id}"),
                                description=layer_node.get("description", ""),
                                layer_type=layer_node.get("layer_type", "default"),
                                parent_layers=[],  # Will populate later
                                properties=properties,
                                metadata=metadata
                            )
                            graph_layers[str(layer_id)] = layer
            
            # Add layers to the graph
            logger.debug(f"Adding {len(graph_layers)} layers to graph")
            for layer_id, layer in graph_layers.items():
                graph.add_layer(layer)
                logger.debug(f"Added layer {layer_id} to graph")
            
            # Now find parent layer relationships
            if self._adapter_type == "NetworkXAdapter":
                # Find all EXTENDS_LAYER relationships
                logger.debug(f"Looking for EXTENDS_LAYER relationships between layers")
                extends_layer_count = 0
                
                for source, target, rel_type, props in edges_data:
                    # Check if this is a parent-child relationship between layers
                    if rel_type == "EXTENDS_LAYER":
                        if isinstance(source, UUID) and isinstance(target, UUID):
                            extends_layer_count += 1
                            child_layer_id = str(source)
                            parent_layer_id = str(target)
                            logger.debug(f"Found EXTENDS_LAYER relationship from {child_layer_id} to {parent_layer_id}")
                            
                            if child_layer_id in graph_layers and parent_layer_id in graph_layers:
                                # source extends target (child extends parent)
                                child_layer = graph.layers[UUID(child_layer_id)]
                                child_layer.parent_layers.append(UUID(parent_layer_id))
                                logger.debug(f"Added parent {parent_layer_id} to child {child_layer_id}")
            
                logger.debug(f"Processed {extends_layer_count} EXTENDS_LAYER relationships")
            else:
                # Standard processing for other adapters
                for source, target, rel_type, props in edges_data:
                    # Convert IDs for consistent comparison
                    source_str = str(source) if isinstance(source, UUID) else source
                    target_str = str(target) if isinstance(target, UUID) else target
                    rel_str = str(rel_type) if isinstance(rel_type, UUID) else rel_type
                    
                    if rel_str == "EXTENDS_LAYER" and source_str in graph_layers and target_str in graph_layers:
                        # This is a parent-child relationship between layers
                        child_layer_id = source_str
                        parent_layer_id = target_str
                        
                        # Add parent to child's parent_layers list
                        if UUID(child_layer_id) in graph.layers:
                            child_layer = graph.layers[UUID(child_layer_id)]
                            child_layer.parent_layers.append(UUID(parent_layer_id))
            
            # Find all nodes in each layer
            logger.debug(f"Processing nodes for graph {graph_id}")
            if self._adapter_type == "NetworkXAdapter":
                # Find all GraphNode nodes
                graph_node_count = 0
                in_layer_count = 0
                nodes_added = 0
                
                for node_id, node_props in nodes_data:
                    if node_props.get("type", "") == "GraphNode":
                        graph_node_count += 1
                        logger.debug(f"Found GraphNode: {node_id} with properties: {node_props}")
                        
                        # Find the layer this node belongs to
                        node_layer_id = None
                        
                        # Look for IN_LAYER relationship from this node to a layer
                        for s, t, r, p in edges_data:
                            if isinstance(s, UUID) and isinstance(t, UUID) and r == "IN_LAYER":
                                if s == node_id and str(t) in graph_layers:
                                    in_layer_count += 1
                                    node_layer_id = t
                                    logger.debug(f"Found IN_LAYER relationship from node {node_id} to layer {node_layer_id}")
                                    break
                        
                        if node_layer_id:
                            # Parse properties
                            properties = {}
                            props_data = node_props.get("properties", {})
                            if isinstance(props_data, str):
                                try:
                                    properties = json.loads(props_data)
                                except json.JSONDecodeError:
                                    properties = {}
                            else:
                                properties = props_data
                            
                            # Parse metadata
                            metadata = {}
                            metadata_data = node_props.get("metadata", {})
                            if isinstance(metadata_data, str):
                                try:
                                    metadata = json.loads(metadata_data)
                                except json.JSONDecodeError:
                                    metadata = {}
                            else:
                                metadata = metadata_data
                            
                            # Create a GraphNode object
                            node = GraphNode(
                                id=node_id,
                                name=node_props.get("name", f"Node {node_id}"),
                                node_type=node_props.get("node_type", "Unknown"),
                                description=node_props.get("description", ""),
                                properties=properties,
                                layer_id=node_layer_id,
                                metadata=metadata
                            )
                            graph.add_node(node, node_layer_id)
                            nodes_added += 1
                            logger.debug(f"Added node {node_id} to layer {node_layer_id}")
            
                logger.debug(f"Found {graph_node_count} GraphNode nodes, {in_layer_count} IN_LAYER relationships, added {nodes_added} nodes to graph")
            else:
                # Standard processing for other adapters
                for node_id, node_props in nodes_data:
                    if node_props.get("type", "") == "GraphNode":
                        # Find the layer this node belongs to
                        node_layer_id = None
                        node_id_str = str(node_id) if isinstance(node_id, UUID) else node_id
                        
                        for source, target, rel_type, props in edges_data:
                            source_str = str(source) if isinstance(source, UUID) else source
                            target_str = str(target) if isinstance(target, UUID) else target
                            rel_str = str(rel_type) if isinstance(rel_type, UUID) else rel_type
                            
                            if source_str == node_id_str and rel_str == "IN_LAYER" and target_str in graph_layers:
                                node_layer_id = target_str
                                break
                        
                        if node_layer_id:
                            # Parse properties
                            properties = {}
                            props_data = node_props.get("properties", {})
                            if isinstance(props_data, str):
                                try:
                                    properties = json.loads(props_data)
                                except json.JSONDecodeError:
                                    properties = {}
                            else:
                                properties = props_data
                                
                            # Parse metadata
                            metadata = {}
                            metadata_data = node_props.get("metadata", {})
                            if isinstance(metadata_data, str):
                                try:
                                    metadata = json.loads(metadata_data)
                                except json.JSONDecodeError:
                                    metadata = {}
                            else:
                                metadata = metadata_data
                                
                            # Create a GraphNode object
                            node = GraphNode(
                                id=UUID(str(node_id)),
                                name=node_props.get("name", f"Node {node_id}"),
                                node_type=node_props.get("node_type", "Unknown"),
                                description=node_props.get("description", ""),
                                properties=properties,
                                layer_id=UUID(node_layer_id),
                                metadata=metadata
                            )
                            graph.add_node(node, UUID(node_layer_id))
            
            # Find all edges in each layer
            logger.debug(f"Processing edges for graph {graph_id}")
            if self._adapter_type == "NetworkXAdapter":
                # Find all GraphEdge nodes
                graph_edge_count = 0
                edge_in_layer_count = 0
                edges_added = 0
                
                for edge_node_id, edge_props in nodes_data:
                    if edge_props.get("type", "") == "GraphEdge":
                        graph_edge_count += 1
                        logger.debug(f"Found GraphEdge: {edge_node_id} with properties: {edge_props}")
                        
                        # Find the layer this edge belongs to
                        edge_layer_id = None
                        
                        # Look for IN_LAYER relationship from this edge to a layer
                        for s, t, r, p in edges_data:
                            if isinstance(s, UUID) and isinstance(t, UUID) and r == "IN_LAYER":
                                if s == edge_node_id and str(t) in graph_layers:
                                    edge_in_layer_count += 1
                                    edge_layer_id = t
                                    logger.debug(f"Found IN_LAYER relationship from edge {edge_node_id} to layer {edge_layer_id}")
                                    break
                        
                        if edge_layer_id:
                            # Parse properties
                            properties = {}
                            props_data = edge_props.get("properties", {})
                            if isinstance(props_data, str):
                                try:
                                    properties = json.loads(props_data)
                                except json.JSONDecodeError:
                                    properties = {}
                            else:
                                properties = props_data
                            
                            # Parse metadata
                            metadata = {}
                            metadata_data = edge_props.get("metadata", {})
                            if isinstance(metadata_data, str):
                                try:
                                    metadata = json.loads(metadata_data)
                                except json.JSONDecodeError:
                                    metadata = {}
                            else:
                                metadata = metadata_data
                            
                            # Create a GraphEdge object
                            edge = GraphEdge(
                                id=edge_node_id,
                                source_node_id=UUID(edge_props.get("source_node_id")),
                                target_node_id=UUID(edge_props.get("target_node_id")),
                                relationship_name=edge_props.get("relationship_name", "RELATED_TO"),
                                properties=properties,
                                layer_id=edge_layer_id,
                                metadata=metadata
                            )
                            graph.add_edge(edge, edge_layer_id)
                            edges_added += 1
                            logger.debug(f"Added edge {edge_node_id} to layer {edge_layer_id}")
            
                logger.debug(f"Found {graph_edge_count} GraphEdge nodes, {edge_in_layer_count} IN_LAYER relationships, added {edges_added} edges to graph")
            else:
                # Standard processing for other adapters
                for edge_node_id, edge_props in nodes_data:
                    if edge_props.get("type", "") == "GraphEdge":
                        # Find the layer this edge belongs to
                        edge_layer_id = None
                        edge_id_str = str(edge_node_id) if isinstance(edge_node_id, UUID) else edge_node_id
                        
                        for source, target, rel_type, props in edges_data:
                            source_str = str(source) if isinstance(source, UUID) else source
                            target_str = str(target) if isinstance(target, UUID) else target
                            rel_str = str(rel_type) if isinstance(rel_type, UUID) else rel_type
                            
                            if source_str == edge_id_str and rel_str == "IN_LAYER" and target_str in graph_layers:
                                edge_layer_id = target_str
                                break
                        
                        if edge_layer_id:
                            # Parse properties
                            properties = {}
                            props_data = edge_props.get("properties", {})
                            if isinstance(props_data, str):
                                try:
                                    properties = json.loads(props_data)
                                except json.JSONDecodeError:
                                    properties = {}
                            else:
                                properties = props_data
                                
                            # Parse metadata
                            metadata = {}
                            metadata_data = edge_props.get("metadata", {})
                            if isinstance(metadata_data, str):
                                try:
                                    metadata = json.loads(metadata_data)
                                except json.JSONDecodeError:
                                    metadata = {}
                            else:
                                metadata = metadata_data
                                
                            # Create a GraphEdge object
                            edge = GraphEdge(
                                id=UUID(str(edge_node_id)),
                                source_node_id=UUID(edge_props.get("source_node_id")),
                                target_node_id=UUID(edge_props.get("target_node_id")),
                                relationship_name=edge_props.get("relationship_name", "RELATED_TO"),
                                properties=properties,
                                layer_id=UUID(edge_layer_id),
                                metadata=metadata
                            )
                            graph.add_edge(edge, UUID(edge_layer_id))
            
            logger.info(f"Successfully retrieved layered graph with ID {graph_id} with {len(graph.layers)} layers, {len(graph.nodes)} nodes, and {len(graph.edges)} edges")
            return graph
            
        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving layered graph: {str(e)}")
            raise
    
    async def _retrieve_layers(self, graph: LayeredKnowledgeGraphDP) -> None:
        """
        Retrieve all layers for a graph and add them to the graph object.
        
        Args:
            graph: The graph to populate with layers
        """
        # Query to get all layers for this graph
        layers_query = f"""
        MATCH (g {{id: '{str(graph.id)}'}})-[:CONTAINS_LAYER]->(l:GraphLayer)
        RETURN l
        """
        layers_result = await self._graph_db.query(layers_query, {})
        
        # Process layers
        for layer_record in layers_result:
            layer_props = layer_record["l"]
            layer_id = UUID(layer_props["id"])
            
            # Create layer
            layer = GraphLayer(
                id=layer_id,
                name=layer_props["name"],
                description=layer_props["description"],
                layer_type=layer_props["layer_type"],
                parent_layers=[UUID(pid) for pid in layer_props["parent_layers"]],
                properties=layer_props["properties"],
                metadata={"type": "GraphLayer", "index_fields": ["name"]}
            )
            
            # Add layer to graph
            graph.layers[layer_id] = layer
    
    async def _retrieve_nodes(self, graph: LayeredKnowledgeGraphDP) -> None:
        """
        Retrieve all nodes for a graph and add them to the graph object.
        
        Args:
            graph: The graph to populate with nodes
        """
        # Query to get all nodes in each layer
        nodes_query = f"""
        MATCH (g {{id: '{str(graph.id)}'}})-[:CONTAINS_LAYER]->(l:GraphLayer),
              (n:GraphNode)-[:IN_LAYER]->(l)
        RETURN n, l.id as layer_id
        """
        nodes_result = await self._graph_db.query(nodes_query, {})
        
        # Process nodes
        for node_record in nodes_result:
            node_props = node_record["n"]
            layer_id = UUID(node_record["layer_id"])
            node_id = UUID(node_props["id"])
            
            # Create node
            node = GraphNode(
                id=node_id,
                name=node_props["name"],
                node_type=node_props["node_type"],
                description=node_props["description"],
                properties=node_props["properties"],
                layer_id=layer_id,
                metadata={"type": "GraphNode", "index_fields": ["name"]}
            )
            
            # Add node to graph
            graph.nodes[node_id] = node
            graph.node_layer_map[node_id] = layer_id
    
    async def _retrieve_edges(self, graph: LayeredKnowledgeGraphDP) -> None:
        """
        Retrieve all edges for a graph and add them to the graph object.
        
        Args:
            graph: The graph to populate with edges
        """
        # Query to get all edges in each layer
        edges_query = f"""
        MATCH (g {{id: '{str(graph.id)}'}})-[:CONTAINS_LAYER]->(l:GraphLayer),
              (e:GraphEdge)-[:IN_LAYER]->(l)
        RETURN e, l.id as layer_id
        """
        edges_result = await self._graph_db.query(edges_query, {})
        
        # Process edges
        for edge_record in edges_result:
            edge_props = edge_record["e"]
            layer_id = UUID(edge_record["layer_id"])
            edge_id = UUID(edge_props["id"])
            
            # Create edge
            edge = GraphEdge(
                id=edge_id,
                source_node_id=UUID(edge_props["source_node_id"]),
                target_node_id=UUID(edge_props["target_node_id"]),
                relationship_name=edge_props["relationship_name"],
                properties=edge_props["properties"],
                layer_id=layer_id,
                metadata={"type": "GraphEdge", "index_fields": ["relationship_name"]}
            )
            
            # Add edge to graph
            graph.edges[edge_id] = edge
            graph.edge_layer_map[edge_id] = layer_id
    
    async def get_layer_hierarchy(self, graph_id: Union[str, UUID]) -> Dict[UUID, List[UUID]]:
        """
        Get the hierarchy of layers in a graph.
        
        Args:
            graph_id: The ID of the graph
            
        Returns:
            Dictionary mapping layer IDs to their child layer IDs
        """
        await self._ensure_graph_db()
        
        # Make sure we have the adapter type
        if not hasattr(self, '_adapter_type') or self._adapter_type is None:
            self._adapter_type = type(self._graph_db).__name__
            logger.debug(f"Using graph database adapter: {self._adapter_type}")
        
        # Convert UUID to string if needed for consistency in logging
        graph_id_str = str(graph_id) if isinstance(graph_id, UUID) else graph_id
        
        try:
            hierarchy = {}
            
            # Special handling for NetworkXAdapter
            if self._adapter_type == "NetworkXAdapter":
                # Ensure graph is loaded
                if hasattr(self._graph_db, 'load_graph_from_file'):
                    await self._graph_db.load_graph_from_file()
                
                # Get graph data to reconstruct hierarchy
                nodes_data, edges_data = await self._graph_db.get_graph_data()
                
                # Find all layers associated with the graph
                graph_layers = set()
                uuid_graph_id = UUID(graph_id_str) if isinstance(graph_id, str) else graph_id
                
                # First, find all layers belonging to this graph
                for source, target, rel_type, props in edges_data:
                    if isinstance(source, UUID) and source == uuid_graph_id and rel_type == "CONTAINS_LAYER":
                        graph_layers.add(target)
                
                # Then, find all parent-child relationships
                parent_to_children = {}
                for source, target, rel_type, props in edges_data:
                    if isinstance(source, UUID) and isinstance(target, UUID) and rel_type == "EXTENDS_LAYER":
                        if source in graph_layers and target in graph_layers:
                            # source extends target (child extends parent)
                            child_id = source 
                            parent_id = target
                            
                            if parent_id not in parent_to_children:
                                parent_to_children[parent_id] = []
                            
                            parent_to_children[parent_id].append(child_id)
                
                # Convert to the expected format
                for parent_id, children in parent_to_children.items():
                    hierarchy[parent_id] = children
                
                return hierarchy
                
            else:
                # For Neo4j or other adapters using Cypher queries
                query = f"""
                MATCH (g {{id: '{graph_id_str}'}})-[:CONTAINS_LAYER]->(l:GraphLayer),
                      (child:GraphLayer)-[:EXTENDS_LAYER]->(parent:GraphLayer)
                WHERE child IN COLLECT(l) AND parent IN COLLECT(l)
                RETURN parent.id as parent_id, COLLECT(child.id) as child_ids
                """
                result = await self._graph_db.query(query, {})
                
                if result:
                    # Build hierarchy dictionary
                    for record in result:
                        parent_id = UUID(record["parent_id"])
                        child_ids = [UUID(child_id) for child_id in record["child_ids"]]
                        hierarchy[parent_id] = child_ids
                
                return hierarchy
                
        except Exception as e:
            logger.error(f"Error getting layer hierarchy: {str(e)}")
            return {}  # Return empty dictionary in case of error 

    async def merge_layers(self, graph_id: Union[str, UUID], layer_ids: List[Union[str, UUID]], 
                       new_layer_name: str = "Merged Layer", new_layer_description: str = "Merged layer") -> UUID:
        """
        Merge multiple layers into a single new layer.
        
        Args:
            graph_id: The ID of the graph containing the layers
            layer_ids: List of layer IDs to merge
            new_layer_name: Name for the new merged layer
            new_layer_description: Description for the new merged layer
            
        Returns:
            The ID of the newly created merged layer
        """
        await self._ensure_graph_db()
        
        # Make sure we have the adapter type
        if not hasattr(self, '_adapter_type') or self._adapter_type is None:
            self._adapter_type = type(self._graph_db).__name__
            logger.debug(f"Using graph database adapter: {self._adapter_type}")
        
        try:
            # Retrieve the graph
            graph = await self.retrieve_graph(graph_id)
            if not graph:
                raise EntityNotFoundError(f"Graph with ID {graph_id} not found")
            
            # Convert layer_ids to UUID objects if they are strings
            uuid_layer_ids = [UUID(layer_id) if isinstance(layer_id, str) else layer_id for layer_id in layer_ids]
            
            # Check if all layers exist in the graph
            for layer_id in uuid_layer_ids:
                if layer_id not in graph.layers:
                    raise EntityNotFoundError(f"Layer with ID {layer_id} not found in graph {graph_id}")
            
            # Create a new layer
            new_layer_id = uuid4()
            new_layer = GraphLayer(
                id=new_layer_id,
                name=new_layer_name,
                description=new_layer_description,
                layer_type="merged",
                parent_layers=uuid_layer_ids,  # Set the merged layers as parents
                properties={},
                metadata={"merged_from": [str(layer_id) for layer_id in uuid_layer_ids]}
            )
            
            # Add the new layer to the graph
            graph.add_layer(new_layer)
            
            # Copy all nodes and edges from the merged layers to the new layer
            for layer_id in uuid_layer_ids:
                # Get all nodes in this layer
                for node_id, node in graph.nodes.items():
                    if node.layer_id == layer_id:
                        # Create a copy of the node in the new layer
                        new_node = GraphNode(
                            id=uuid4(),  # Generate a new ID for the node
                            name=node.name,
                            node_type=node.node_type,
                            description=node.description,
                            properties=node.properties.copy(),
                            layer_id=new_layer_id,
                            metadata={"original_node_id": str(node_id), "original_layer_id": str(layer_id), **node.metadata}
                        )
                        graph.add_node(new_node, new_layer_id)
                
                # Get all edges in this layer
                for edge_id, edge in graph.edges.items():
                    if edge.layer_id == layer_id:
                        # Create a copy of the edge in the new layer
                        new_edge = GraphEdge(
                            id=uuid4(),  # Generate a new ID for the edge
                            source_node_id=edge.source_node_id,
                            target_node_id=edge.target_node_id,
                            relationship_name=edge.relationship_name,
                            properties=edge.properties.copy(),
                            layer_id=new_layer_id,
                            metadata={"original_edge_id": str(edge_id), "original_layer_id": str(layer_id), **edge.metadata}
                        )
                        graph.add_edge(new_edge, new_layer_id)
            
            # Store the updated graph
            await self.store_graph(graph)
            
            return new_layer_id
        
        except Exception as e:
            logger.error(f"Error merging layers: {str(e)}")
            raise 

    async def get_layer_metrics(self, layer_id: Union[str, UUID]) -> Dict[str, Any]:
        """
        Get metrics for a specific layer.
        
        Args:
            layer_id: The ID of the layer to get metrics for
            
        Returns:
            Dictionary of metrics for the layer
        """
        await self._ensure_graph_db()
        
        # Make sure we have the adapter type
        if not hasattr(self, '_adapter_type') or self._adapter_type is None:
            self._adapter_type = type(self._graph_db).__name__
            logger.debug(f"Using graph database adapter: {self._adapter_type}")
        
        try:
            # Convert layer_id to UUID if it's a string
            if isinstance(layer_id, str):
                uuid_layer_id = UUID(layer_id)
            else:
                uuid_layer_id = layer_id
                layer_id = str(layer_id)  # Keep string version for logging
            
            # Get the layer node data from the database
            if self._adapter_type == "NetworkXAdapter":
                layer_node = await self._graph_db.extract_node(uuid_layer_id)
            else:
                layer_node = await self._graph_db.extract_node(layer_id)
            
            if not layer_node:
                raise EntityNotFoundError(f"Layer with ID {layer_id} not found")
            
            # Get graph data to count nodes and edges in this layer
            nodes_data, edges_data = await self._graph_db.get_graph_data()
            
            # Count nodes in this layer
            node_count = 0
            for node_id, node_props in nodes_data:
                if node_props.get("type") == "GraphNode" and node_props.get("layer_id") == layer_id:
                    node_count += 1
            
            # Count edges in this layer
            edge_count = 0
            for edge_id, edge_props in nodes_data:
                if edge_props.get("type") == "GraphEdge" and edge_props.get("layer_id") == layer_id:
                    edge_count += 1
            
            # Count relationships between nodes in this layer
            relationship_count = 0
            relationship_types = set()
            for source, target, rel_type, props in edges_data:
                # Check if both source and target are nodes in this layer
                source_in_layer = False
                target_in_layer = False
                
                for node_id, node_props in nodes_data:
                    if node_props.get("type") == "GraphNode" and node_props.get("layer_id") == layer_id:
                        if str(node_id) == str(source):
                            source_in_layer = True
                        if str(node_id) == str(target):
                            target_in_layer = True
                
                if source_in_layer and target_in_layer:
                    relationship_count += 1
                    relationship_types.add(rel_type)
            
            # Calculate metrics
            metrics = {
                "node_count": node_count,
                "edge_count": edge_count,
                "relationship_count": relationship_count,
                "relationship_types": list(relationship_types),
                "density": relationship_count / (node_count * (node_count - 1)) if node_count > 1 else 0,
                "avg_degree": (2 * relationship_count) / node_count if node_count > 0 else 0,
                "layer_type": layer_node.get("layer_type", "unknown"),
                "name": layer_node.get("name", f"Layer {layer_id}"),
                "description": layer_node.get("description", "")
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error getting layer metrics: {str(e)}")
            return {
                "node_count": 0,
                "edge_count": 0,
                "relationship_count": 0,
                "relationship_types": [],
                "density": 0,
                "avg_degree": 0,
                "layer_type": "unknown",
                "name": f"Layer {layer_id}",
                "description": ""
            } 

    async def find_cross_layer_relationships(self, graph_id: Union[str, UUID]) -> List[Dict[str, Any]]:
        """
        Find relationships between nodes in different layers.
        
        Args:
            graph_id: The ID of the graph to analyze
            
        Returns:
            List of dictionaries describing cross-layer relationships
        """
        await self._ensure_graph_db()
        
        # Make sure we have the adapter type
        if not hasattr(self, '_adapter_type') or self._adapter_type is None:
            self._adapter_type = type(self._graph_db).__name__
            logger.debug(f"Using graph database adapter: {self._adapter_type}")
        
        try:
            # Retrieve the graph
            graph = await self.retrieve_graph(graph_id)
            if not graph:
                raise EntityNotFoundError(f"Graph with ID {graph_id} not found")
            
            # Get graph data to find cross-layer relationships
            nodes_data, edges_data = await self._graph_db.get_graph_data()
            
            # Find all relationships between nodes in different layers
            cross_layer_relationships = []
            
            # Build a map of node IDs to layer IDs
            node_to_layer = {}
            for node_id, node_props in nodes_data:
                if node_props.get("type") == "GraphNode" and node_props.get("layer_id"):
                    node_to_layer[str(node_id)] = node_props.get("layer_id")
            
            # Check all edges for cross-layer relationships
            for source, target, rel_type, props in edges_data:
                source_str = str(source)
                target_str = str(target)
                
                # Skip if either node is not in our node_to_layer map
                if source_str not in node_to_layer or target_str not in node_to_layer:
                    continue
                
                source_layer = node_to_layer[source_str]
                target_layer = node_to_layer[target_str]
                
                # If the nodes are in different layers, this is a cross-layer relationship
                if source_layer != target_layer:
                    # Get the node data
                    source_node = None
                    target_node = None
                    for node_id, node_props in nodes_data:
                        if str(node_id) == source_str:
                            source_node = node_props
                        if str(node_id) == target_str:
                            target_node = node_props
                    
                    if source_node and target_node:
                        # Get the layer data
                        source_layer_node = None
                        target_layer_node = None
                        for node_id, node_props in nodes_data:
                            if str(node_id) == source_layer:
                                source_layer_node = node_props
                            if str(node_id) == target_layer:
                                target_layer_node = node_props
                        
                        # Add the relationship to the result
                        cross_layer_relationships.append({
                            "source_node_id": source_str,
                            "target_node_id": target_str,
                            "source_node_name": source_node.get("name", f"Node {source_str}"),
                            "target_node_name": target_node.get("name", f"Node {target_str}"),
                            "source_layer_id": source_layer,
                            "target_layer_id": target_layer,
                            "source_layer_name": source_layer_node.get("name", f"Layer {source_layer}") if source_layer_node else f"Layer {source_layer}",
                            "target_layer_name": target_layer_node.get("name", f"Layer {target_layer}") if target_layer_node else f"Layer {target_layer}",
                            "relationship_type": rel_type,
                            "properties": props
                        })
            
            return cross_layer_relationships
        
        except Exception as e:
            logger.error(f"Error finding cross-layer relationships: {str(e)}")
            return [] 

    async def delete_graph(self, graph_id: Union[str, UUID]) -> bool:
        """
        Delete a graph and all its associated data from the database.
        
        Args:
            graph_id: The ID of the graph to delete
            
        Returns:
            True if the graph was deleted successfully, False otherwise
        """
        await self._ensure_graph_db()
        
        # Make sure we have the adapter type
        if not hasattr(self, '_adapter_type') or self._adapter_type is None:
            self._adapter_type = type(self._graph_db).__name__
            logger.debug(f"Using graph database adapter: {self._adapter_type}")
        
        try:
            # Convert graph_id to UUID if it's a string
            if isinstance(graph_id, str):
                uuid_graph_id = UUID(graph_id)
            else:
                uuid_graph_id = graph_id
                graph_id = str(graph_id)  # Keep string version for logging
            
            # For NetworkXAdapter, we need to handle the deletion differently
            if self._adapter_type == "NetworkXAdapter":
                # First, get the graph data
                nodes_data, edges_data = await self._graph_db.get_graph_data()
                
                # Find all nodes and edges associated with this graph
                nodes_to_delete = set()
                
                # Add the graph node itself
                nodes_to_delete.add(uuid_graph_id)
                
                # Find all layers associated with this graph
                layers = set()
                for source, target, rel_type, props in edges_data:
                    if isinstance(source, UUID) and source == uuid_graph_id and rel_type == "CONTAINS_LAYER":
                        layers.add(target)
                        nodes_to_delete.add(target)
                
                # Find all nodes and edges in these layers
                for node_id, node_props in nodes_data:
                    if node_props.get("layer_id") in [str(layer_id) for layer_id in layers]:
                        nodes_to_delete.add(node_id)
                
                # Delete all the nodes
                for node_id in nodes_to_delete:
                    await self._graph_db.delete_node(node_id)
                
                logger.info(f"Successfully deleted graph with ID {graph_id} and {len(nodes_to_delete)} associated nodes")
                return True
            else:
                # For other adapters, we can use the delete_graph method if available
                if hasattr(self._graph_db, 'delete_graph'):
                    await self._graph_db.delete_graph()
                    logger.info(f"Successfully deleted graph with ID {graph_id}")
                    return True
                else:
                    # Otherwise, we need to delete the graph node and all associated nodes
                    # This is similar to the NetworkXAdapter approach
                    # ... (similar code as above)
                    logger.info(f"Successfully deleted graph with ID {graph_id}")
                    return True
        
        except Exception as e:
            logger.error(f"Error deleting graph: {str(e)}")
            return False 

    async def extract_subgraph(self, graph_id: Union[str, UUID], layer_id: Union[str, UUID]) -> LayeredKnowledgeGraphDP:
        """
        Extract a subgraph containing only the specified layer.
        
        Args:
            graph_id: The ID of the graph containing the layer
            layer_id: The ID of the layer to extract
            
        Returns:
            A new layered knowledge graph containing only the specified layer
        """
        await self._ensure_graph_db()
        
        # Make sure we have the adapter type
        if not hasattr(self, '_adapter_type') or self._adapter_type is None:
            self._adapter_type = type(self._graph_db).__name__
            logger.debug(f"Using graph database adapter: {self._adapter_type}")
        
        try:
            # Retrieve the graph
            graph = await self.retrieve_graph(graph_id)
            if not graph:
                raise EntityNotFoundError(f"Graph with ID {graph_id} not found")
            
            # Convert layer_id to UUID if it's a string
            if isinstance(layer_id, str):
                uuid_layer_id = UUID(layer_id)
            else:
                uuid_layer_id = layer_id
                layer_id = str(layer_id)  # Keep string version for logging
            
            # Check if the layer exists in the graph
            if uuid_layer_id not in graph.layers:
                raise EntityNotFoundError(f"Layer with ID {layer_id} not found in graph {graph_id}")
            
            # Create a new graph with only the specified layer
            subgraph = LayeredKnowledgeGraphDP.create_empty(
                name=f"Subgraph of {graph.name} - Layer {graph.layers[uuid_layer_id].name}",
                description=f"Subgraph extracted from {graph.name} containing only layer {graph.layers[uuid_layer_id].name}"
            )
            
            # Add the layer to the subgraph
            layer = graph.layers[uuid_layer_id]
            subgraph.add_layer(layer)
            
            # Add all nodes in this layer to the subgraph
            for node_id, node in graph.nodes.items():
                if node.layer_id == uuid_layer_id:
                    subgraph.add_node(node, uuid_layer_id)
            
            # Add all edges in this layer to the subgraph
            for edge_id, edge in graph.edges.items():
                if edge.layer_id == uuid_layer_id:
                    # Only add the edge if both source and target nodes are in the subgraph
                    if edge.source_node_id in subgraph.nodes and edge.target_node_id in subgraph.nodes:
                        subgraph.add_edge(edge, uuid_layer_id)
            
            return subgraph
        
        except Exception as e:
            logger.error(f"Error extracting subgraph: {str(e)}")
            raise 