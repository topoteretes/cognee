#!/usr/bin/env python3
"""
Debug script for understanding how NetworkXAdapter handles layered graphs.
"""

import asyncio
import json
import logging
import sys
import uuid
from uuid import UUID
from datetime import datetime, timezone

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import DataPoint
from pydantic import Field
from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode, GraphEdge, GraphLayer, LayeredKnowledgeGraphDP
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

async def create_test_graph() -> LayeredKnowledgeGraphDP:
    """Create a simple test layered graph."""
    # Create an empty graph
    graph = LayeredKnowledgeGraphDP.create_empty(
        name="Test Layered Graph",
        description="A test layered graph for debugging"
    )
    
    # Create base layer
    base_layer = GraphLayer.create(
        name="Base Layer",
        description="Base layer containing core entities",
        layer_type="base"
    )
    graph.add_layer(base_layer)
    
    # Create nodes in base layer
    node1 = GraphNode.create(
        name="Node 1",
        node_type="TestNode",
        description="Test node 1",
        properties={"prop1": "value1"}
    )
    graph.add_node(node1, base_layer.id)
    
    node2 = GraphNode.create(
        name="Node 2",
        node_type="TestNode",
        description="Test node 2",
        properties={"prop2": "value2"}
    )
    graph.add_node(node2, base_layer.id)
    
    # Create edge between nodes
    edge = GraphEdge.create(
        source_node_id=node1.id,
        target_node_id=node2.id,
        relationship_name="TEST_RELATIONSHIP",
        properties={"edge_prop": "edge_value"}
    )
    graph.add_edge(edge, base_layer.id)
    
    return graph

async def debug_nx_adapter():
    """Debug NetworkXAdapter with layered graphs."""
    logger.info("Starting NetworkXAdapter layered graph debug test")
    
    # Get graph engine
    adapter = await get_graph_engine()
    adapter_type = type(adapter).__name__
    logger.info(f"Using graph database: {adapter_type}")
    
    # Ensure graph is initialized and loaded
    if hasattr(adapter, 'graph') and adapter.graph is None:
        if hasattr(adapter, 'create_empty_graph'):
            logger.info(f"Creating empty graph at {adapter.filename}")
            await adapter.create_empty_graph(adapter.filename)
    
    if hasattr(adapter, 'load_graph_from_file'):
        logger.info(f"Loading graph from {adapter.filename}")
        await adapter.load_graph_from_file()
    
    # Create test graph
    test_graph = await create_test_graph()
    logger.info(f"Created test graph with {len(test_graph.layers)} layers, {len(test_graph.nodes)} nodes, and {len(test_graph.edges)} edges")
    
    # Store graph node
    logger.info(f"Storing graph node with ID: {test_graph.id}")
    await adapter.add_node(test_graph)
    
    # Store layers
    for layer_id, layer in test_graph.layers.items():
        logger.info(f"Storing layer: {layer.name} ({layer.id})")
        await adapter.add_node(layer)
        
        # Add CONTAINS_LAYER relationship
        logger.info(f"Adding CONTAINS_LAYER relationship: {test_graph.id} -> {layer.id}")
        await adapter.add_edge(
            str(test_graph.id),
            str(layer.id),
            "CONTAINS_LAYER",
            {"graph_id": str(test_graph.id), "layer_id": str(layer.id)}
        )
    
    # Store nodes
    for node_id, node in test_graph.nodes.items():
        logger.info(f"Storing node: {node.name} ({node.id})")
        await adapter.add_node(node)
        
        # Add IN_LAYER relationship
        if node.layer_id:
            logger.info(f"Adding IN_LAYER relationship: {node.id} -> {node.layer_id}")
            await adapter.add_edge(
                str(node.id),
                str(node.layer_id),
                "IN_LAYER",
                {"node_id": str(node.id), "layer_id": str(node.layer_id)}
            )
    
    # Store edges
    for edge_id, edge in test_graph.edges.items():
        logger.info(f"Storing edge: {edge.id}")
        await adapter.add_node(edge)
        
        # Add relationship between nodes
        logger.info(f"Adding relationship: {edge.source_node_id} --[{edge.relationship_name}]--> {edge.target_node_id}")
        await adapter.add_edge(
            str(edge.source_node_id),
            str(edge.target_node_id),
            edge.relationship_name,
            {"edge_id": str(edge.id), **edge.properties}
        )
        
        # Add IN_LAYER relationship
        if edge.layer_id:
            logger.info(f"Adding IN_LAYER relationship: {edge.id} -> {edge.layer_id}")
            await adapter.add_edge(
                str(edge.id),
                str(edge.layer_id),
                "IN_LAYER",
                {"edge_id": str(edge.id), "layer_id": str(edge.layer_id)}
            )
    
    # Save the graph to file
    if hasattr(adapter, 'save_graph_to_file'):
        logger.info(f"Saving graph to {adapter.filename}")
        await adapter.save_graph_to_file(adapter.filename)
    
    # Load the graph from file
    if hasattr(adapter, 'load_graph_from_file'):
        logger.info(f"Loading graph from {adapter.filename}")
        await adapter.load_graph_from_file()
    
    # Verify the graph exists
    logger.info("Verifying graph data...")
    
    # Check if the graph node exists
    graph_node = await adapter.extract_node(test_graph.id)
    if graph_node:
        logger.info(f"Graph node found: {graph_node}")
    else:
        logger.error(f"Graph node not found")
    
    # Get all graph data
    nodes_data, edges_data = await adapter.get_graph_data()
    logger.info(f"Graph has {len(nodes_data)} nodes and {len(edges_data)} edges")
    
    # List all nodes
    logger.info("All nodes in graph:")
    for node_id, node_props in nodes_data:
        logger.info(f"Node ID: {node_id}, Type: {type(node_id)}")
        logger.info(f"  Properties: {node_props}")
    
    # List all edges
    logger.info("All edges in graph:")
    for source, target, rel_type, props in edges_data:
        logger.info(f"Edge: {source} --[{rel_type}]--> {target}")
        logger.info(f"  Properties: {props}")
    
    # Clean up
    logger.info(f"Deleting test graph...")
    for node_id, _ in nodes_data:
        await adapter.delete_node(node_id)
    
    logger.info("Debug test completed")

if __name__ == "__main__":
    asyncio.run(debug_nx_adapter()) 