"""
Simple standalone test script for layered graph functionality.

This script demonstrates the key features of layered knowledge graphs without
requiring the evaluation framework.
"""

import asyncio
import logging
from typing import Dict, List, Any

from cognee.shared.data_models import (
    KnowledgeGraph,
    LayeredKnowledgeGraph,
    Layer, 
    Node, 
    Edge
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_simple_graph() -> KnowledgeGraph:
    """
    Create a simple knowledge graph about cities and countries.
    
    Returns:
        A simple knowledge graph
    """
    # Create nodes
    nodes = [
        Node(
            id="usa",
            name="United States",
            type="Country",
            description="The United States of America"
        ),
        Node(
            id="nyc",
            name="New York City",
            type="City",
            description="New York City, largest city in the USA"
        ),
        Node(
            id="sf",
            name="San Francisco",
            type="City",
            description="San Francisco, California"
        )
    ]
    
    # Create edges
    edges = [
        Edge(
            source_node_id="nyc",
            target_node_id="usa",
            relationship_name="LOCATED_IN"
        ),
        Edge(
            source_node_id="sf",
            target_node_id="usa",
            relationship_name="LOCATED_IN"
        )
    ]
    
    # Create knowledge graph
    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        name="Cities and Countries",
        description="A graph of cities and their countries"
    )


def create_layered_graph() -> LayeredKnowledgeGraph:
    """
    Create a layered knowledge graph with multiple layers.
    
    Returns:
        A layered knowledge graph
    """
    # Initialize with empty base graph
    base_graph = KnowledgeGraph(nodes=[], edges=[])
    
    # Create layered graph
    layered_graph = LayeredKnowledgeGraph(
        base_graph=base_graph,
        layers=[],
        name="Test Layered Graph",
        description="A test layered knowledge graph"
    )
    
    # Create base layer
    base_layer = Layer(
        id="base",
        name="Base Layer",
        description="Geography base layer",
        layer_type="base"
    )
    layered_graph.add_layer(base_layer)
    
    # Add nodes and edges to base layer
    country_node = Node(
        id="germany",
        name="Germany",
        type="Country",
        description="Germany in Europe"
    )
    layered_graph.add_node_to_layer(country_node, "base")
    
    city_node = Node(
        id="berlin",
        name="Berlin",
        type="City",
        description="Berlin, capital of Germany"
    )
    layered_graph.add_node_to_layer(city_node, "base")
    
    # Add relationship between city and country
    city_country_edge = Edge(
        source_node_id="berlin",
        target_node_id="germany",
        relationship_name="CAPITAL_OF"
    )
    layered_graph.add_edge_to_layer(city_country_edge, "base")
    
    # Create landmarks layer that builds on the base layer
    landmarks_layer = Layer(
        id="landmarks",
        name="Landmarks Layer",
        description="Famous landmarks",
        layer_type="enrichment",
        parent_layers=["base"]
    )
    layered_graph.add_layer(landmarks_layer)
    
    # Add landmark nodes
    landmark_node = Node(
        id="brandenburg_gate",
        name="Brandenburg Gate",
        type="Landmark",
        description="The Brandenburg Gate in Berlin"
    )
    layered_graph.add_node_to_layer(landmark_node, "landmarks")
    
    # Add edge connecting landmark to city
    landmark_city_edge = Edge(
        source_node_id="brandenburg_gate",
        target_node_id="berlin",
        relationship_name="LOCATED_IN"
    )
    layered_graph.add_edge_to_layer(landmark_city_edge, "landmarks")
    
    return layered_graph


async def test_layer_graph_operations() -> None:
    """
    Test basic operations on layered knowledge graphs.
    """
    logger.info("Creating a layered knowledge graph...")
    layered_graph = create_layered_graph()
    
    # Get individual layer graphs
    logger.info("\nGetting individual layer graphs:")
    
    base_graph = layered_graph.get_layer_graph("base")
    logger.info(f"Base layer has {len(base_graph.nodes)} nodes and {len(base_graph.edges)} edges")
    for node in base_graph.nodes:
        logger.info(f"  Node: {node.name} (type: {node.type})")
    for edge in base_graph.edges:
        logger.info(f"  Edge: {edge.source_node_id} --[{edge.relationship_name}]--> {edge.target_node_id}")
    
    landmarks_graph = layered_graph.get_layer_graph("landmarks")
    logger.info(f"\nLandmarks layer has {len(landmarks_graph.nodes)} nodes and {len(landmarks_graph.edges)} edges")
    for node in landmarks_graph.nodes:
        logger.info(f"  Node: {node.name} (type: {node.type})")
    for edge in landmarks_graph.edges:
        logger.info(f"  Edge: {edge.source_node_id} --[{edge.relationship_name}]--> {edge.target_node_id}")
    
    # Get cumulative layer graph
    logger.info("\nGetting cumulative layer graph:")
    cumulative_graph = layered_graph.get_cumulative_layer_graph("landmarks")
    logger.info(f"Cumulative landmarks graph has {len(cumulative_graph.nodes)} nodes and {len(cumulative_graph.edges)} edges")
    for node in cumulative_graph.nodes:
        logger.info(f"  Node: {node.name} (type: {node.type})")
    for edge in cumulative_graph.edges:
        logger.info(f"  Edge: {edge.source_node_id} --[{edge.relationship_name}]--> {edge.target_node_id}")
    
    # Add a new layer
    logger.info("\nAdding a new layer with events:")
    events_layer = Layer(
        id="events",
        name="Events Layer",
        description="Historical events",
        layer_type="enrichment",
        parent_layers=["landmarks"]
    )
    layered_graph.add_layer(events_layer)
    
    # Add event node
    event_node = Node(
        id="wall_fall",
        name="Fall of Berlin Wall",
        type="Event",
        description="The fall of the Berlin Wall in 1989"
    )
    layered_graph.add_node_to_layer(event_node, "events")
    
    # Add edge connecting event to landmark
    event_landmark_edge = Edge(
        source_node_id="wall_fall",
        target_node_id="brandenburg_gate",
        relationship_name="NEAR"
    )
    layered_graph.add_edge_to_layer(event_landmark_edge, "events")
    
    # Get the new cumulative graph
    final_graph = layered_graph.get_cumulative_layer_graph("events")
    logger.info(f"Final cumulative graph has {len(final_graph.nodes)} nodes and {len(final_graph.edges)} edges")
    
    # Check the layer hierarchy
    logger.info("\nLayer hierarchy:")
    for layer in layered_graph.layers:
        parent_names = []
        for parent_id in layer.parent_layers:
            parent = next((l for l in layered_graph.layers if l.id == parent_id), None)
            if parent:
                parent_names.append(parent.name)
        
        if parent_names:
            logger.info(f"  {layer.name} depends on: {', '.join(parent_names)}")
        else:
            logger.info(f"  {layer.name} is a root layer")


async def main() -> None:
    """
    Main function that runs the test.
    """
    logger.info("=== Testing Layered Knowledge Graph Functionality ===")
    
    try:
        await test_layer_graph_operations()
        logger.info("\nLayered graph functionality test completed successfully!")
    except Exception as e:
        logger.error(f"Error testing layered graph functionality: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 