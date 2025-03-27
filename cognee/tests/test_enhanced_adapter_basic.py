#!/usr/bin/env python3
"""
Basic test script for the enhanced LayeredGraphDBAdapter.
"""

import asyncio
import json
import logging
import sys
import uuid
import os
from uuid import UUID
from datetime import datetime, timezone

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import DataPoint
from pydantic import Field
from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode,
    GraphEdge,
    GraphLayer,
    LayeredKnowledgeGraphDP,
)
from cognee.modules.graph.enhanced_layered_graph_adapter import LayeredGraphDBAdapter

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def create_test_graph() -> LayeredKnowledgeGraphDP:
    """Create a complex test layered graph."""
    # Create an empty graph
    graph = LayeredKnowledgeGraphDP.create_empty(
        name="Complex Test Graph",
        description="A complex test layered graph with multiple layers and relationships",
    )
    logger.debug(f"Created empty graph with ID: {graph.id}")

    # Create base layer
    base_layer = GraphLayer.create(
        name="Base Layer", description="Base layer containing core entities", layer_type="base"
    )
    logger.debug(f"Created base layer with ID: {base_layer.id}")
    graph.add_layer(base_layer)

    # Create derived layer 1
    derived_layer1 = GraphLayer.create(
        name="Derived Layer 1",
        description="First derived layer",
        layer_type="derived",
        parent_layers=[base_layer.id],
    )
    logger.debug(f"Created derived layer 1 with ID: {derived_layer1.id}")
    graph.add_layer(derived_layer1)

    # Create derived layer 2
    derived_layer2 = GraphLayer.create(
        name="Derived Layer 2",
        description="Second derived layer",
        layer_type="derived",
        parent_layers=[base_layer.id],
    )
    logger.debug(f"Created derived layer 2 with ID: {derived_layer2.id}")
    graph.add_layer(derived_layer2)

    # Create composite layer that extends both derived layers
    composite_layer = GraphLayer.create(
        name="Composite Layer",
        description="Layer extending both derived layers",
        layer_type="composite",
        parent_layers=[derived_layer1.id, derived_layer2.id],
    )
    logger.debug(f"Created composite layer with ID: {composite_layer.id}")
    graph.add_layer(composite_layer)

    # Create nodes in base layer
    base_nodes = []
    for i in range(3):
        node = GraphNode.create(
            name=f"Base Node {i + 1}",
            node_type="BaseNode",
            description=f"Base layer node {i + 1}",
            properties={"base_prop": f"value{i + 1}"},
        )
        logger.debug(f"Created base node {i + 1} with ID: {node.id}")
        graph.add_node(node, base_layer.id)
        base_nodes.append(node)

    # Create edges between base nodes
    for i in range(len(base_nodes)):
        for j in range(i + 1, len(base_nodes)):
            edge = GraphEdge.create(
                source_node_id=base_nodes[i].id,
                target_node_id=base_nodes[j].id,
                relationship_name="BASE_CONNECTS",
                properties={"weight": i + j},
            )
            logger.debug(f"Created base edge between nodes {i + 1} and {j + 1} with ID: {edge.id}")
            graph.add_edge(edge, base_layer.id)

    # Create nodes in derived layer 1
    derived1_nodes = []
    for i in range(2):
        node = GraphNode.create(
            name=f"Derived1 Node {i + 1}",
            node_type="Derived1Node",
            description=f"Derived layer 1 node {i + 1}",
            properties={"derived1_prop": f"value{i + 1}"},
        )
        logger.debug(f"Created derived1 node {i + 1} with ID: {node.id}")
        graph.add_node(node, derived_layer1.id)
        derived1_nodes.append(node)

    # Create edges between derived1 nodes and base nodes
    for derived_node in derived1_nodes:
        for base_node in base_nodes:
            edge = GraphEdge.create(
                source_node_id=derived_node.id,
                target_node_id=base_node.id,
                relationship_name="DERIVES_FROM",
                properties={"confidence": 0.8},
            )
            logger.debug(
                f"Created edge from derived1 node {derived_node.name} to base node {base_node.name} with ID: {edge.id}"
            )
            graph.add_edge(edge, derived_layer1.id)

    # Create nodes in derived layer 2
    derived2_nodes = []
    for i in range(2):
        node = GraphNode.create(
            name=f"Derived2 Node {i + 1}",
            node_type="Derived2Node",
            description=f"Derived layer 2 node {i + 1}",
            properties={"derived2_prop": f"value{i + 1}"},
        )
        logger.debug(f"Created derived2 node {i + 1} with ID: {node.id}")
        graph.add_node(node, derived_layer2.id)
        derived2_nodes.append(node)

    # Create edges between derived2 nodes and base nodes
    for derived_node in derived2_nodes:
        for base_node in base_nodes:
            edge = GraphEdge.create(
                source_node_id=derived_node.id,
                target_node_id=base_node.id,
                relationship_name="EXTENDS",
                properties={"confidence": 0.9},
            )
            logger.debug(
                f"Created edge from derived2 node {derived_node.name} to base node {base_node.name} with ID: {edge.id}"
            )
            graph.add_edge(edge, derived_layer2.id)

    # Create nodes in composite layer
    composite_nodes = []
    for i in range(2):
        node = GraphNode.create(
            name=f"Composite Node {i + 1}",
            node_type="CompositeNode",
            description=f"Composite layer node {i + 1}",
            properties={"composite_prop": f"value{i + 1}"},
        )
        logger.debug(f"Created composite node {i + 1} with ID: {node.id}")
        graph.add_node(node, composite_layer.id)
        composite_nodes.append(node)

    # Create edges from composite nodes to derived nodes
    for comp_node in composite_nodes:
        for d1_node in derived1_nodes:
            edge = GraphEdge.create(
                source_node_id=comp_node.id,
                target_node_id=d1_node.id,
                relationship_name="USES_DERIVED1",
                properties={"weight": 0.7},
            )
            logger.debug(
                f"Created edge from composite node {comp_node.name} to derived1 node {d1_node.name} with ID: {edge.id}"
            )
            graph.add_edge(edge, composite_layer.id)

        for d2_node in derived2_nodes:
            edge = GraphEdge.create(
                source_node_id=comp_node.id,
                target_node_id=d2_node.id,
                relationship_name="USES_DERIVED2",
                properties={"weight": 0.8},
            )
            logger.debug(
                f"Created edge from composite node {comp_node.name} to derived2 node {d2_node.name} with ID: {edge.id}"
            )
            graph.add_edge(edge, composite_layer.id)

    return graph


async def test_enhanced_adapter_basic():
    """Test basic functionality of the enhanced LayeredGraphDBAdapter."""
    logger.info("Starting enhanced LayeredGraphDBAdapter basic test")

    # Get graph engine
    graph_db = await get_graph_engine()
    adapter_type = type(graph_db).__name__
    logger.info(f"Using graph database: {adapter_type}")

    # Check if the graph file exists
    if hasattr(graph_db, "filename"):
        graph_file = graph_db.filename
        logger.info(f"Graph file path: {graph_file}")
        if os.path.exists(graph_file):
            logger.info(f"Graph file exists with size: {os.path.getsize(graph_file)} bytes")
            # Try to read the file
            try:
                with open(graph_file, "r") as f:
                    file_content = f.read()
                    logger.debug(f"Graph file content (first 500 chars): {file_content[:500]}")
            except Exception as e:
                logger.error(f"Error reading graph file: {str(e)}")
        else:
            logger.info(f"Graph file does not exist")

    # Create adapter
    adapter = LayeredGraphDBAdapter(graph_db=graph_db)
    logger.info(f"Created LayeredGraphDBAdapter: {adapter}")

    # Create test graph
    test_graph = await create_test_graph()
    logger.info(
        f"Created test graph with {len(test_graph.layers)} layers, {len(test_graph.nodes)} nodes, and {len(test_graph.edges)} edges"
    )

    # Log graph details
    logger.debug(f"Graph ID: {test_graph.id}")
    for layer_id, layer in test_graph.layers.items():
        logger.debug(f"Layer: {layer.name} (ID: {layer.id})")
    for node_id, node in test_graph.nodes.items():
        logger.debug(f"Node: {node.name} (ID: {node.id}, Layer: {node.layer_id})")
    for edge_id, edge in test_graph.edges.items():
        logger.debug(
            f"Edge: {edge.relationship_name} (ID: {edge.id}, Source: {edge.source_node_id}, Target: {edge.target_node_id}, Layer: {edge.layer_id})"
        )

    # Store graph
    logger.info(f"Storing graph: {test_graph.name}")
    graph_id = await adapter.store_graph(test_graph)
    logger.info(f"Graph stored with ID: {graph_id}")

    # Check if the graph file exists after storing
    if hasattr(graph_db, "filename"):
        graph_file = graph_db.filename
        logger.info(f"Graph file path after storing: {graph_file}")
        if os.path.exists(graph_file):
            logger.info(
                f"Graph file exists after storing with size: {os.path.getsize(graph_file)} bytes"
            )
            # Try to read the file
            try:
                with open(graph_file, "r") as f:
                    file_content = f.read()
                    logger.debug(f"Graph file content: {file_content}")
            except Exception as e:
                logger.error(f"Error reading graph file after storing: {str(e)}")
        else:
            logger.info(f"Graph file does not exist after storing")

    # Get graph data from database
    nodes_data, edges_data = await graph_db.get_graph_data()
    logger.debug(f"Database has {len(nodes_data)} nodes and {len(edges_data)} edges")

    # Log all nodes in database
    logger.debug("All nodes in database:")
    for node_id, node_props in nodes_data:
        logger.debug(f"Node ID: {node_id}, Type: {type(node_id)}")
        logger.debug(f"  Properties: {node_props}")

    # Log all edges in database
    logger.debug("All edges in database:")
    for source, target, rel_type, props in edges_data:
        logger.debug(f"Edge: {source} --[{rel_type}]--> {target}")
        logger.debug(f"  Properties: {props}")

    # Retrieve graph
    logger.info(f"Retrieving graph with ID: {graph_id}")
    try:
        # Force loading the graph from file
        if hasattr(graph_db, "load_graph_from_file") and hasattr(graph_db, "filename"):
            logger.info(f"Force loading graph from file before retrieving")
            # Create a temporary copy of the file to force loading
            temp_file = graph_db.filename + ".temp"
            try:
                # Copy the file
                with open(graph_db.filename, "r") as src, open(temp_file, "w") as dst:
                    dst.write(src.read())
                # Load from the temp file
                await graph_db.load_graph_from_file(temp_file)
                # Clean up
                os.remove(temp_file)
            except Exception as e:
                logger.error(f"Error forcing graph load: {str(e)}")

        # Check if the graph exists in the database
        if hasattr(graph_db, "has_node"):
            has_node = await graph_db.has_node(UUID(graph_id))
            logger.info(f"Graph node exists in database: {has_node}")

        # Get graph data from database after loading
        nodes_data, edges_data = await graph_db.get_graph_data()
        logger.debug(
            f"After loading, database has {len(nodes_data)} nodes and {len(edges_data)} edges"
        )

        # Log all nodes in database after loading
        logger.debug("All nodes in database after loading:")
        for node_id, node_props in nodes_data:
            logger.debug(f"Node ID: {node_id}, Type: {type(node_id)}")
            logger.debug(f"  Properties: {node_props}")

        # Log all edges in database after loading
        logger.debug("All edges in database after loading:")
        for source, target, rel_type, props in edges_data:
            logger.debug(f"Edge: {source} --[{rel_type}]--> {target}")
            logger.debug(f"  Properties: {props}")

        retrieved_graph = await adapter.retrieve_graph(graph_id)
        logger.info(f"Retrieved graph: {retrieved_graph.name}")
        logger.info(f"Layer count: {len(retrieved_graph.layers)}")
        logger.info(f"Node count: {len(retrieved_graph.nodes)}")
        logger.info(f"Edge count: {len(retrieved_graph.edges)}")

        # Log retrieved graph details
        logger.debug(f"Retrieved Graph ID: {retrieved_graph.id}")
        for layer_id, layer in retrieved_graph.layers.items():
            logger.debug(f"Retrieved Layer: {layer.name} (ID: {layer.id})")
        for node_id, node in retrieved_graph.nodes.items():
            logger.debug(f"Retrieved Node: {node.name} (ID: {node.id}, Layer: {node.layer_id})")
        for edge_id, edge in retrieved_graph.edges.items():
            logger.debug(
                f"Retrieved Edge: {edge.relationship_name} (ID: {edge.id}, Source: {edge.source_node_id}, Target: {edge.target_node_id}, Layer: {edge.layer_id})"
            )

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
                logger.debug(f"Node '{node.name}' successfully retrieved")
            else:
                logger.error(f"Node '{node.name}' not found in retrieved graph")
        logger.info(f"{node_count}/{len(test_graph.nodes)} nodes successfully retrieved")

        # Check if all edges were retrieved
        edge_count = 0
        for edge_id, edge in test_graph.edges.items():
            if edge_id in retrieved_graph.edges:
                edge_count += 1
                logger.debug(f"Edge '{edge.relationship_name}' successfully retrieved")
            else:
                logger.error(f"Edge '{edge.relationship_name}' not found in retrieved graph")
        logger.info(f"{edge_count}/{len(test_graph.edges)} edges successfully retrieved")
    except Exception as e:
        logger.error(f"Error retrieving graph: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())

    # Delete graph
    logger.info(f"Deleting graph with ID: {graph_id}")
    result = await adapter.delete_graph(graph_id)
    logger.info(f"Graph deleted: {result}")

    logger.info("Basic test completed")


if __name__ == "__main__":
    asyncio.run(test_enhanced_adapter_basic())
