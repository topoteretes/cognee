#!/usr/bin/env python
"""
Test script for the enhanced LayeredGraphDBAdapter.

This script demonstrates how to create, store, and retrieve a layered knowledge graph
using the enhanced adapter with compatibility for different database backends.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Union

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode,
    GraphEdge,
    GraphLayer,
    LayeredKnowledgeGraphDP,
)
from cognee.modules.graph.enhanced_layered_graph_adapter import LayeredGraphDBAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def create_sample_layered_graph() -> LayeredKnowledgeGraphDP:
    """
    Create a sample layered knowledge graph for testing.

    Returns:
        LayeredKnowledgeGraphDP: A sample layered knowledge graph with two layers
    """
    logger.info("Creating sample layered knowledge graph")

    # Create the graph itself
    graph = LayeredKnowledgeGraphDP.create_empty(
        name="Test Knowledge Graph",
        description="A test graph for validating the enhanced adapter",
        metadata={"type": "LayeredKnowledgeGraph", "index_fields": ["name"]},
    )

    # Create two layers: a base layer and a derived layer
    base_layer = GraphLayer(
        id=uuid.uuid4(),
        name="Base Layer",
        description="The foundation layer with core entities",
        layer_type="base",
        parent_layers=[],
        properties={"importance": "high"},
        metadata={"type": "GraphLayer", "index_fields": ["name"]},
    )

    derived_layer = GraphLayer(
        id=uuid.uuid4(),
        name="Derived Layer",
        description="A layer built on top of the base layer",
        layer_type="derived",
        parent_layers=[base_layer.id],
        properties={"importance": "medium"},
        metadata={"type": "GraphLayer", "index_fields": ["name"]},
    )

    # Add layers to the graph
    graph.add_layer(base_layer)
    graph.add_layer(derived_layer)

    # Create nodes in the base layer
    company_node = GraphNode(
        id=uuid.uuid4(),
        name="Apple Inc.",
        node_type="Company",
        description="Technology company headquartered in Cupertino",
        properties={"founded": 1976, "industry": "Technology"},
        layer_id=base_layer.id,
        metadata={"type": "GraphNode", "index_fields": ["name"]},
    )

    product_node = GraphNode(
        id=uuid.uuid4(),
        name="iPhone",
        node_type="Product",
        description="Smartphone product line by Apple",
        properties={"launch_year": 2007, "category": "smartphone"},
        layer_id=base_layer.id,
        metadata={"type": "GraphNode", "index_fields": ["name"]},
    )

    # Add nodes to the graph
    graph.add_node(company_node, base_layer.id)
    graph.add_node(product_node, base_layer.id)

    # Create an edge in the base layer
    produces_edge = GraphEdge(
        id=uuid.uuid4(),
        source_node_id=company_node.id,
        target_node_id=product_node.id,
        relationship_name="PRODUCES",
        properties={"since": 2007},
        layer_id=base_layer.id,
        metadata={"type": "GraphEdge", "index_fields": ["relationship_name"]},
    )

    # Add edge to the graph
    graph.add_edge(produces_edge, base_layer.id)

    # Create nodes in the derived layer
    feature_node = GraphNode(
        id=uuid.uuid4(),
        name="Touchscreen",
        node_type="Feature",
        description="Touchscreen interface",
        properties={"type": "input", "importance": "critical"},
        layer_id=derived_layer.id,
        metadata={"type": "GraphNode", "index_fields": ["name"]},
    )

    # Add the derived node to the graph
    graph.add_node(feature_node, derived_layer.id)

    # Create an edge in the derived layer
    has_feature_edge = GraphEdge(
        id=uuid.uuid4(),
        source_node_id=product_node.id,
        target_node_id=feature_node.id,
        relationship_name="HAS_FEATURE",
        properties={"since_version": 1},
        layer_id=derived_layer.id,
        metadata={"type": "GraphEdge", "index_fields": ["relationship_name"]},
    )

    # Add the derived edge to the graph
    graph.add_edge(has_feature_edge, derived_layer.id)

    logger.info(
        f"Created graph with {len(graph.layers)} layers, {len(graph.nodes)} nodes, and {len(graph.edges)} edges"
    )
    return graph


async def test_layered_graph_adapter():
    """
    Test the LayeredGraphDBAdapter with a sample graph.

    This function:
    1. Creates a sample layered graph
    2. Initializes the adapter
    3. Stores the graph in the database
    4. Retrieves the graph from the database
    5. Validates the retrieved graph
    """
    try:
        # Create a sample layered graph
        original_graph = await create_sample_layered_graph()
        logger.info(f"Created original graph with ID: {original_graph.id}")

        # Initialize the graph database
        logger.info("Initializing graph database")
        graph_db = await get_graph_engine()

        # Initialize the adapter
        adapter = LayeredGraphDBAdapter(graph_db)

        # Store the graph
        logger.info("Storing graph in database")
        stored_graph_id = await adapter.store_graph(original_graph)
        logger.info(f"Successfully stored graph with ID: {stored_graph_id}")

        # Retrieve the graph
        logger.info(f"Retrieving graph with ID: {stored_graph_id}")
        retrieved_graph = await adapter.retrieve_graph(stored_graph_id)

        # Validate the retrieved graph
        validate_retrieved_graph(original_graph, retrieved_graph)

        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Error in test: {str(e)}", exc_info=True)
        raise


def validate_retrieved_graph(original: LayeredKnowledgeGraphDP, retrieved: LayeredKnowledgeGraphDP):
    """
    Validate that the retrieved graph matches the original graph.

    Args:
        original: The original graph that was stored
        retrieved: The graph that was retrieved from the database
    """
    logger.info("Validating retrieved graph against original")

    # Check basic graph properties
    assert original.id == retrieved.id, "Graph IDs do not match"
    assert original.name == retrieved.name, "Graph names do not match"
    assert original.description == retrieved.description, "Graph descriptions do not match"

    # Check layer count
    assert len(original.layers) == len(retrieved.layers), "Layer counts do not match"
    logger.info(f"Graph has {len(retrieved.layers)} layers as expected")

    # Check node count
    assert len(original.nodes) == len(retrieved.nodes), "Node counts do not match"
    logger.info(f"Graph has {len(retrieved.nodes)} nodes as expected")

    # Check edge count
    assert len(original.edges) == len(retrieved.edges), "Edge counts do not match"
    logger.info(f"Graph has {len(retrieved.edges)} edges as expected")

    # Check that all layer IDs exist
    for layer_id in original.layers:
        assert layer_id in retrieved.layers, f"Layer {layer_id} not found in retrieved graph"

    # Check that all node IDs exist
    for node_id in original.nodes:
        assert node_id in retrieved.nodes, f"Node {node_id} not found in retrieved graph"

    # Check that all edge IDs exist
    for edge_id in original.edges:
        assert edge_id in retrieved.edges, f"Edge {edge_id} not found in retrieved graph"

    # Validate layer relationships
    for layer_id, layer in original.layers.items():
        retrieved_layer = retrieved.layers[layer_id]
        assert layer.name == retrieved_layer.name, f"Layer {layer_id} names do not match"
        assert set(layer.parent_layers) == set(retrieved_layer.parent_layers), (
            f"Layer {layer_id} parent layers do not match"
        )

    # Validate node properties
    for node_id, node in original.nodes.items():
        retrieved_node = retrieved.nodes[node_id]
        assert node.name == retrieved_node.name, f"Node {node_id} names do not match"
        assert node.node_type == retrieved_node.node_type, f"Node {node_id} types do not match"
        assert node.layer_id == retrieved_node.layer_id, f"Node {node_id} layer IDs do not match"

    # Validate edge relationships
    for edge_id, edge in original.edges.items():
        retrieved_edge = retrieved.edges[edge_id]
        assert edge.source_node_id == retrieved_edge.source_node_id, (
            f"Edge {edge_id} source nodes do not match"
        )
        assert edge.target_node_id == retrieved_edge.target_node_id, (
            f"Edge {edge_id} target nodes do not match"
        )
        assert edge.relationship_name == retrieved_edge.relationship_name, (
            f"Edge {edge_id} relationship names do not match"
        )
        assert edge.layer_id == retrieved_edge.layer_id, f"Edge {edge_id} layer IDs do not match"

    logger.info("Graph validation successful - all properties and relationships match")


async def main():
    """Main entry point for the test script."""
    logger.info("Starting layered graph adapter test")
    await test_layered_graph_adapter()


if __name__ == "__main__":
    asyncio.run(main())
