#!/usr/bin/env python3
"""
Test script for the enhanced LayeredGraphDBAdapter.

This script tests the enhanced LayeredGraphDBAdapter implementation to ensure
it works correctly with existing graph databases.

The test covers:
1. Creating a layered graph with multiple layers
2. Storing the graph in the database
3. Retrieving the graph from the database
4. Layer operations (hierarchy, merging, metrics)
5. Subgraph extraction
6. Cross-layer relationships
"""

import asyncio
import json
import logging
import sys
import uuid
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from datetime import datetime

from cognee.infrastructure.databases.graph import get_graph_engine, get_graph_config
from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode, GraphEdge, GraphLayer, LayeredKnowledgeGraphDP
)
from cognee.modules.graph.enhanced_layered_graph_adapter import LayeredGraphDBAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

async def create_test_graph() -> LayeredKnowledgeGraphDP:
    """
    Create a test layered knowledge graph with multiple layers, nodes, and edges.
    
    Returns:
        A test layered knowledge graph
    """
    # Create an empty graph
    graph = LayeredKnowledgeGraphDP.create_empty(
        name="Test Layered Graph",
        description="A test layered graph for verification"
    )
    
    # Create base layer
    base_layer = GraphLayer.create(
        name="Base Layer",
        description="Base layer containing core entities",
        layer_type="base"
    )
    graph.add_layer(base_layer)
    
    # Create nodes in base layer
    apple_node = GraphNode.create(
        name="Apple Inc.",
        node_type="Organization",
        description="American technology company",
        properties={"founded": "1976", "headquarters": "Cupertino, California"}
    )
    graph.add_node(apple_node, base_layer.id)
    
    google_node = GraphNode.create(
        name="Google LLC",
        node_type="Organization",
        description="American technology company",
        properties={"founded": "1998", "headquarters": "Mountain View, California"}
    )
    graph.add_node(google_node, base_layer.id)
    
    tim_cook_node = GraphNode.create(
        name="Tim Cook",
        node_type="Person",
        description="CEO of Apple Inc.",
        properties={"role": "CEO", "joined": "1998"}
    )
    graph.add_node(tim_cook_node, base_layer.id)
    
    sundar_pichai_node = GraphNode.create(
        name="Sundar Pichai",
        node_type="Person",
        description="CEO of Google LLC",
        properties={"role": "CEO", "joined": "2004"}
    )
    graph.add_node(sundar_pichai_node, base_layer.id)
    
    # Create relationships layer that extends the base layer
    relationships_layer = GraphLayer.create(
        name="Relationships Layer",
        description="Layer containing relationships between entities",
        layer_type="relationships",
        parent_layers=[base_layer.id]
    )
    graph.add_layer(relationships_layer)
    
    # Create edges in relationships layer
    tim_ceo_edge = GraphEdge.create(
        source_node_id=tim_cook_node.id,
        target_node_id=apple_node.id,
        relationship_name="IS_CEO_OF",
        properties={"since": "2011"}
    )
    graph.add_edge(tim_ceo_edge, relationships_layer.id)
    
    sundar_ceo_edge = GraphEdge.create(
        source_node_id=sundar_pichai_node.id,
        target_node_id=google_node.id,
        relationship_name="IS_CEO_OF",
        properties={"since": "2015"}
    )
    graph.add_edge(sundar_ceo_edge, relationships_layer.id)
    
    competition_edge = GraphEdge.create(
        source_node_id=apple_node.id,
        target_node_id=google_node.id,
        relationship_name="COMPETES_WITH",
        properties={"markets": ["smartphones", "software"]}
    )
    graph.add_edge(competition_edge, relationships_layer.id)
    
    # Create categories layer that also extends the base layer
    categories_layer = GraphLayer.create(
        name="Categories Layer",
        description="Layer categorizing entities",
        layer_type="categories",
        parent_layers=[base_layer.id]
    )
    graph.add_layer(categories_layer)
    
    # Add nodes to categories layer
    tech_company_node = GraphNode.create(
        name="Technology Company",
        node_type="Category",
        description="Companies in the technology sector",
        properties={}
    )
    graph.add_node(tech_company_node, categories_layer.id)
    
    executive_node = GraphNode.create(
        name="Executive",
        node_type="Category",
        description="Company executives",
        properties={}
    )
    graph.add_node(executive_node, categories_layer.id)
    
    # Create classification edges in the categories layer
    apple_tech_edge = GraphEdge.create(
        source_node_id=apple_node.id,
        target_node_id=tech_company_node.id,
        relationship_name="IS_A",
        properties={}
    )
    graph.add_edge(apple_tech_edge, categories_layer.id)
    
    google_tech_edge = GraphEdge.create(
        source_node_id=google_node.id,
        target_node_id=tech_company_node.id,
        relationship_name="IS_A",
        properties={}
    )
    graph.add_edge(google_tech_edge, categories_layer.id)
    
    tim_exec_edge = GraphEdge.create(
        source_node_id=tim_cook_node.id,
        target_node_id=executive_node.id,
        relationship_name="IS_A",
        properties={}
    )
    graph.add_edge(tim_exec_edge, categories_layer.id)
    
    sundar_exec_edge = GraphEdge.create(
        source_node_id=sundar_pichai_node.id,
        target_node_id=executive_node.id,
        relationship_name="IS_A",
        properties={}
    )
    graph.add_edge(sundar_exec_edge, categories_layer.id)
    
    return graph

async def test_store_and_retrieve_graph(adapter: LayeredGraphDBAdapter, test_graph: LayeredKnowledgeGraphDP):
    """
    Test storing and retrieving a layered graph.
    
    Args:
        adapter: The adapter to use for testing
        test_graph: The test graph to store and retrieve
    """
    logger.info("=== Testing Graph Storage and Retrieval ===")
    
    try:
        # Store the graph
        logger.info(f"Storing graph: {test_graph.name}")
        graph_id = await adapter.store_graph(test_graph)
        logger.info(f"Graph stored with ID: {graph_id}")
        
        # Retrieve the graph
        logger.info(f"Retrieving graph with ID: {graph_id}")
        retrieved_graph = await adapter.retrieve_graph(graph_id)
        
        # Verify the graph was retrieved correctly
        logger.info(f"Retrieved graph: {retrieved_graph.name}")
        logger.info(f"Layer count: {len(retrieved_graph.layers)}")
        logger.info(f"Node count: {len(retrieved_graph.nodes)}")
        logger.info(f"Edge count: {len(retrieved_graph.edges)}")
        
        # Check if all layers were retrieved
        for layer_id, layer in test_graph.layers.items():
            if layer_id in retrieved_graph.layers:
                logger.info(f"Layer '{layer.name}' successfully retrieved")
            else:
                logger.error(f"Layer '{layer.name}' not found in retrieved graph")
        
        # Check if all nodes were retrieved
        node_count = 0
        for node_id, node in test_graph.nodes.items():
            if node_id in retrieved_graph.nodes:
                node_count += 1
        logger.info(f"{node_count}/{len(test_graph.nodes)} nodes successfully retrieved")
        
        # Check if all edges were retrieved
        edge_count = 0
        for edge_id, edge in test_graph.edges.items():
            if edge_id in retrieved_graph.edges:
                edge_count += 1
        logger.info(f"{edge_count}/{len(test_graph.edges)} edges successfully retrieved")
        
        return graph_id, retrieved_graph
        
    except Exception as e:
        logger.error(f"Error during store/retrieve test: {str(e)}")
        raise

async def test_layer_operations(adapter: LayeredGraphDBAdapter, graph_id: Union[str, UUID], graph: LayeredKnowledgeGraphDP):
    """
    Test operations on layers.
    
    Args:
        adapter: The adapter to use for testing
        graph_id: The ID of the graph
        graph: The graph to test with
    """
    logger.info("\n=== Testing Layer Operations ===")
    
    try:
        # Get layer hierarchy
        logger.info("Getting layer hierarchy...")
        hierarchy = await adapter.get_layer_hierarchy(graph_id)
        logger.info(f"Layer hierarchy: {hierarchy}")
        
        # Get layer metrics for each layer
        for layer_id, layer in graph.layers.items():
            logger.info(f"Getting metrics for layer: {layer.name}")
            metrics = await adapter.get_layer_metrics(layer_id)
            logger.info(f"Layer metrics: {metrics}")
        
        # Test merging layers
        # Get the layer IDs we want to merge
        layer_ids = [layer_id for layer_id, layer in graph.layers.items() 
                    if layer.layer_type in ["relationships", "categories"]]
        
        logger.info(f"Merging {len(layer_ids)} layers...")
        merged_layer_id = await adapter.merge_layers(
            graph_id, 
            layer_ids, 
            "Merged Layer", 
            "A merged layer combining relationships and categories"
        )
        logger.info(f"Created merged layer with ID: {merged_layer_id}")
        
        # Retrieve the updated graph
        updated_graph = await adapter.retrieve_graph(graph_id)
        logger.info(f"Updated graph has {len(updated_graph.layers)} layers")
        
        # Verify the merged layer exists
        merged_layer = None
        for layer_id, layer in updated_graph.layers.items():
            if layer.name == "Merged Layer":
                merged_layer = layer
                break
        
        if merged_layer:
            logger.info(f"Merged layer found: {merged_layer.name}")
            
            # Count nodes and edges in the merged layer
            merged_nodes = [node for node in updated_graph.nodes.values() if node.layer_id == merged_layer.id]
            merged_edges = [edge for edge in updated_graph.edges.values() if edge.layer_id == merged_layer.id]
            
            logger.info(f"Merged layer contains {len(merged_nodes)} nodes and {len(merged_edges)} edges")
        else:
            logger.error("Merged layer not found in updated graph")
        
        return updated_graph
        
    except Exception as e:
        logger.error(f"Error during layer operations test: {str(e)}")
        raise

async def test_subgraph_extraction(adapter: LayeredGraphDBAdapter, graph_id: Union[str, UUID], graph: LayeredKnowledgeGraphDP):
    """
    Test extracting a subgraph based on selected layers.
    
    Args:
        adapter: The adapter to use for testing
        graph_id: The ID of the graph
        graph: The graph to test with
    """
    logger.info("\n=== Testing Subgraph Extraction ===")
    
    try:
        # Select a subset of layers for the subgraph
        base_layer_id = None
        for layer_id, layer in graph.layers.items():
            if layer.layer_type == "base":
                base_layer_id = layer_id
                break
        
        if not base_layer_id:
            logger.error("Base layer not found for subgraph extraction")
            return
        
        # Extract subgraph with just the base layer
        logger.info(f"Extracting subgraph with base layer...")
        subgraph = await adapter.extract_subgraph_by_layers(graph_id, [base_layer_id])
        
        logger.info(f"Subgraph created with name: {subgraph.name}")
        logger.info(f"Subgraph has {len(subgraph.layers)} layers")
        logger.info(f"Subgraph has {len(subgraph.nodes)} nodes")
        logger.info(f"Subgraph has {len(subgraph.edges)} edges")
        
        return subgraph
        
    except Exception as e:
        logger.error(f"Error during subgraph extraction test: {str(e)}")
        raise

async def test_cross_layer_relationships(adapter: LayeredGraphDBAdapter, graph_id: Union[str, UUID]):
    """
    Test finding relationships between nodes in different layers.
    
    Args:
        adapter: The adapter to use for testing
        graph_id: The ID of the graph
    """
    logger.info("\n=== Testing Cross-Layer Relationships ===")
    
    try:
        logger.info("Finding cross-layer relationships...")
        relationships = await adapter.find_cross_layer_relationships(graph_id)
        
        logger.info(f"Found {len(relationships)} cross-layer relationships")
        for rel in relationships:
            logger.info(f"Relationship: {rel['source_layer']} -> {rel['target_layer']} ({rel['relationship_type']})")
            
    except Exception as e:
        logger.error(f"Error during cross-layer relationships test: {str(e)}")
        raise

async def test_graph_deletion(adapter: LayeredGraphDBAdapter, graph_id: Union[str, UUID]):
    """
    Test deleting a graph.
    
    Args:
        adapter: The adapter to use for testing
        graph_id: The ID of the graph to delete
    """
    logger.info("\n=== Testing Graph Deletion ===")
    
    try:
        logger.info(f"Deleting graph with ID: {graph_id}")
        result = await adapter.delete_graph(graph_id)
        
        if result:
            logger.info("Graph deleted successfully")
        else:
            logger.error("Failed to delete graph")
        
        # Try to retrieve the deleted graph
        try:
            deleted_graph = await adapter.retrieve_graph(graph_id)
            logger.error("Graph was not deleted - it can still be retrieved")
        except Exception:
            logger.info("Verified graph cannot be retrieved after deletion")
            
    except Exception as e:
        logger.error(f"Error during graph deletion test: {str(e)}")
        raise

async def run_all_tests():
    """Run all tests for the enhanced LayeredGraphDBAdapter."""
    logger.info("Starting tests for enhanced LayeredGraphDBAdapter")
    
    # Get graph database configuration
    graph_config = get_graph_config()
    logger.info(f"Using graph database: {graph_config.graph_database_provider}")
    
    # Get graph database engine
    graph_db = await get_graph_engine()
    logger.info(f"Graph database engine: {type(graph_db).__name__}")
    
    # Create adapter
    adapter = LayeredGraphDBAdapter(graph_db=graph_db)
    logger.info(f"Created LayeredGraphDBAdapter: {adapter}")
    
    # Create test graph
    test_graph = await create_test_graph()
    logger.info(f"Created test graph with {len(test_graph.layers)} layers, {len(test_graph.nodes)} nodes, and {len(test_graph.edges)} edges")
    
    # Run tests
    graph_id, retrieved_graph = await test_store_and_retrieve_graph(adapter, test_graph)
    updated_graph = await test_layer_operations(adapter, graph_id, retrieved_graph)
    subgraph = await test_subgraph_extraction(adapter, graph_id, updated_graph)
    await test_cross_layer_relationships(adapter, graph_id)
    await test_graph_deletion(adapter, graph_id)
    
    logger.info("\nAll tests completed!")

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 