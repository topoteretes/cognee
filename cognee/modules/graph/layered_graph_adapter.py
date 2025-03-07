"""
Adapter for layered knowledge graphs to work with Cognee's graph database infrastructure.

This module provides an adapter implementation that allows layered knowledge graphs to be 
stored in and retrieved from graph databases compatible with Cognee's GraphDBInterface.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode, GraphEdge, GraphLayer, LayeredKnowledgeGraphDP, UUIDEncoder
)
from cognee.shared.data_models import Node, Edge, KnowledgeGraph

logger = logging.getLogger(__name__)

class LayeredGraphDBAdapter:
    """
    Adapter for storing and retrieving layered knowledge graphs using Cognee's graph database.
    
    This class provides methods to:
    1. Store layered knowledge graphs in a graph database
    2. Retrieve layered knowledge graphs from a graph database
    3. Query information about layers in the graph database
    """
    
    def __init__(self, graph_db: Optional[GraphDBInterface] = None):
        """
        Initialize the adapter with a graph database.
        
        Args:
            graph_db: Optional GraphDBInterface instance. If None, the default graph engine will be used.
        """
        self._graph_db = graph_db
        self._graph_db_initialized = graph_db is not None
    
    async def _ensure_graph_db(self):
        """Ensure the graph database is initialized."""
        if not self._graph_db_initialized:
            self._graph_db = await get_graph_engine()
            self._graph_db_initialized = True
    
    async def store_graph(self, graph: LayeredKnowledgeGraphDP) -> str:
        """
        Store a layered knowledge graph in the graph database.
        
        Args:
            graph: The layered knowledge graph to store
            
        Returns:
            The ID of the stored graph
        """
        await self._ensure_graph_db()
        
        # First, store the graph itself as a node
        graph_node_properties = {
            "id": str(graph.id),
            "name": graph.name,
            "description": graph.description,
            "type": "LayeredKnowledgeGraph",
            "metadata": json.dumps(graph.metadata, cls=UUIDEncoder)
        }
        
        await self._graph_db.add_node(str(graph.id), graph_node_properties)
        
        # Store layers
        layer_tasks = []
        for layer_id, layer in graph.layers.items():
            layer_node_properties = {
                "id": str(layer.id),
                "name": layer.name,
                "description": layer.description,
                "layer_type": layer.layer_type,
                "parent_layers": json.dumps([str(parent_id) for parent_id in layer.parent_layers], cls=UUIDEncoder),
                "properties": json.dumps(layer.properties, cls=UUIDEncoder),
                "type": "GraphLayer",
                "metadata": json.dumps(layer.metadata, cls=UUIDEncoder)
            }
            
            # Create task to add the layer node
            layer_tasks.append(self._graph_db.add_node(str(layer.id), layer_node_properties))
            
            # Create task to add the CONTAINS_LAYER relationship
            layer_tasks.append(self._graph_db.add_edge(
                str(graph.id),
                str(layer.id),
                "CONTAINS_LAYER",
                {"graph_id": str(graph.id), "layer_id": str(layer.id)}
            ))
            
            # Create tasks for parent layer relationships
            for parent_id in layer.parent_layers:
                layer_tasks.append(self._graph_db.add_edge(
                    str(layer.id),
                    str(parent_id),
                    "EXTENDS_LAYER",
                    {"child_layer_id": str(layer.id), "parent_layer_id": str(parent_id)}
                ))
        
        # Run all layer tasks
        await asyncio.gather(*layer_tasks)
        
        # Store nodes
        node_tasks = []
        for node_id, node in graph.nodes.items():
            node_properties = {
                "id": str(node.id),
                "name": node.name,
                "node_type": node.node_type,
                "description": node.description,
                "properties": json.dumps(node.properties, cls=UUIDEncoder),
                "layer_id": str(node.layer_id) if node.layer_id else None,
                "type": "GraphNode",
                "metadata": json.dumps(node.metadata, cls=UUIDEncoder)
            }
            
            # Create task to add the node
            node_tasks.append(self._graph_db.add_node(str(node.id), node_properties))
            
            # Create task to add the IN_LAYER relationship
            if node.layer_id:
                node_tasks.append(self._graph_db.add_edge(
                    str(node.id),
                    str(node.layer_id),
                    "IN_LAYER",
                    {"node_id": str(node.id), "layer_id": str(node.layer_id)}
                ))
        
        # Run all node tasks
        await asyncio.gather(*node_tasks)
        
        # Store edges
        edge_tasks = []
        for edge_id, edge in graph.edges.items():
            edge_properties = {
                "id": str(edge.id),
                "source_node_id": str(edge.source_node_id),
                "target_node_id": str(edge.target_node_id),
                "relationship_name": edge.relationship_name,
                "properties": json.dumps(edge.properties, cls=UUIDEncoder),
                "layer_id": str(edge.layer_id) if edge.layer_id else None,
                "type": "GraphEdge",
                "metadata": json.dumps(edge.metadata, cls=UUIDEncoder)
            }
            
            # Create task to add the edge node (represent edge as a node for querying)
            edge_tasks.append(self._graph_db.add_node(str(edge.id), edge_properties))
            
            # Create task to add the relationship between source and target
            edge_tasks.append(self._graph_db.add_edge(
                str(edge.source_node_id),
                str(edge.target_node_id),
                edge.relationship_name,
                {"edge_id": str(edge.id), **edge.properties}
            ))
            
            # Create task to add the IN_LAYER relationship for the edge
            if edge.layer_id:
                edge_tasks.append(self._graph_db.add_edge(
                    str(edge.id),
                    str(edge.layer_id),
                    "IN_LAYER",
                    {"edge_id": str(edge.id), "layer_id": str(edge.layer_id)}
                ))
        
        # Run all edge tasks
        await asyncio.gather(*edge_tasks)
        
        return str(graph.id)
    
    async def retrieve_graph(self, graph_id: Union[str, UUID]) -> LayeredKnowledgeGraphDP:
        """
        Retrieve a layered knowledge graph from the graph database.
        
        Args:
            graph_id: The ID of the graph to retrieve
            
        Returns:
            The retrieved layered knowledge graph
        """
        await self._ensure_graph_db()
        
        # Convert UUID to string if needed
        if isinstance(graph_id, UUID):
            graph_id = str(graph_id)
        
        # Retrieve the graph node
        graph_node = await self._graph_db.extract_node(graph_id)
        if not graph_node:
            raise ValueError(f"Graph with ID {graph_id} not found")
        
        # Create the graph object
        graph = LayeredKnowledgeGraphDP(
            id=UUID(graph_node["id"]),
            name=graph_node["name"],
            description=graph_node["description"],
            metadata=json.loads(graph_node["metadata"]),
            layers={},
            nodes={},
            edges={},
            node_layer_map={},
            edge_layer_map={}
        )
        
        # Query to get all layers for this graph
        layers_query = f"""
        MATCH (g {{id: '{graph_id}'}})-[:CONTAINS_LAYER]->(l:GraphLayer)
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
                parent_layers=[UUID(pid) for pid in json.loads(layer_props["parent_layers"])],
                properties=json.loads(layer_props["properties"]),
                metadata=json.loads(layer_props["metadata"])
            )
            
            # Add layer to graph
            graph.layers[layer_id] = layer
        
        # Query to get all nodes in each layer
        nodes_query = f"""
        MATCH (g {{id: '{graph_id}'}})-[:CONTAINS_LAYER]->(l:GraphLayer),
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
                properties=json.loads(node_props["properties"]),
                layer_id=layer_id,
                metadata=json.loads(node_props["metadata"])
            )
            
            # Add node to graph
            graph.nodes[node_id] = node
            graph.node_layer_map[node_id] = layer_id
        
        # Query to get all edges in each layer
        edges_query = f"""
        MATCH (g {{id: '{graph_id}'}})-[:CONTAINS_LAYER]->(l:GraphLayer),
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
                properties=json.loads(edge_props["properties"]),
                layer_id=layer_id,
                metadata=json.loads(edge_props["metadata"])
            )
            
            # Add edge to graph
            graph.edges[edge_id] = edge
            graph.edge_layer_map[edge_id] = layer_id
        
        return graph
    
    async def store_layer(self, graph_id: Union[str, UUID], layer: GraphLayer) -> str:
        """
        Store a single layer in the graph database, associating it with a graph.
        
        Args:
            graph_id: The ID of the graph to associate the layer with
            layer: The layer to store
            
        Returns:
            The ID of the stored layer
        """
        await self._ensure_graph_db()
        
        # Convert UUID to string if needed
        if isinstance(graph_id, UUID):
            graph_id = str(graph_id)
        
        # Create layer node properties
        layer_node_properties = {
            "id": str(layer.id),
            "name": layer.name,
            "description": layer.description,
            "layer_type": layer.layer_type,
            "parent_layers": json.dumps([str(parent_id) for parent_id in layer.parent_layers], cls=UUIDEncoder),
            "properties": json.dumps(layer.properties, cls=UUIDEncoder),
            "type": "GraphLayer",
            "metadata": json.dumps(layer.metadata, cls=UUIDEncoder)
        }
        
        # Add the layer node
        await self._graph_db.add_node(str(layer.id), layer_node_properties)
        
        # Add the CONTAINS_LAYER relationship
        await self._graph_db.add_edge(
            graph_id,
            str(layer.id),
            "CONTAINS_LAYER",
            {"graph_id": graph_id, "layer_id": str(layer.id)}
        )
        
        # Add parent layer relationships
        for parent_id in layer.parent_layers:
            await self._graph_db.add_edge(
                str(layer.id),
                str(parent_id),
                "EXTENDS_LAYER",
                {"child_layer_id": str(layer.id), "parent_layer_id": str(parent_id)}
            )
        
        return str(layer.id)
    
    async def get_layer_ids(self, graph_id: Union[str, UUID]) -> List[str]:
        """
        Get the IDs of all layers in a graph.
        
        Args:
            graph_id: The ID of the graph
            
        Returns:
            List of layer IDs
        """
        await self._ensure_graph_db()
        
        # Convert UUID to string if needed
        if isinstance(graph_id, UUID):
            graph_id = str(graph_id)
        
        # Query to get all layer IDs
        query = f"""
        MATCH (g {{id: '{graph_id}'}})-[:CONTAINS_LAYER]->(l:GraphLayer)
        RETURN l.id as layer_id
        """
        result = await self._graph_db.query(query, {})
        
        return [record["layer_id"] for record in result]
    
    async def get_nodes_in_layer(self, layer_id: Union[str, UUID]) -> List[Dict[str, Any]]:
        """
        Get all nodes in a specific layer.
        
        Args:
            layer_id: The ID of the layer
            
        Returns:
            List of node data
        """
        await self._ensure_graph_db()
        
        # Convert UUID to string if needed
        if isinstance(layer_id, UUID):
            layer_id = str(layer_id)
        
        # Query to get all nodes in the layer
        query = f"""
        MATCH (n:GraphNode)-[:IN_LAYER]->(:GraphLayer {{id: '{layer_id}'}})
        RETURN n
        """
        result = await self._graph_db.query(query, {})
        
        return [record["n"] for record in result]
    
    async def get_edges_in_layer(self, layer_id: Union[str, UUID]) -> List[Dict[str, Any]]:
        """
        Get all edges in a specific layer.
        
        Args:
            layer_id: The ID of the layer
            
        Returns:
            List of edge data
        """
        await self._ensure_graph_db()
        
        # Convert UUID to string if needed
        if isinstance(layer_id, UUID):
            layer_id = str(layer_id)
        
        # Query to get all edges in the layer
        query = f"""
        MATCH (e:GraphEdge)-[:IN_LAYER]->(:GraphLayer {{id: '{layer_id}'}})
        RETURN e
        """
        result = await self._graph_db.query(query, {})
        
        return [record["e"] for record in result] 