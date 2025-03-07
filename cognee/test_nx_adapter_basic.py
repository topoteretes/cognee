#!/usr/bin/env python3
"""
Basic test script for NetworkXAdapter.
"""

import asyncio
import logging
import sys
import uuid
from uuid import UUID
from datetime import datetime, timezone

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.engine import DataPoint
from pydantic import Field

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Define a simple DataPoint class
class TestNode(DataPoint):
    """Test node for debugging."""
    name: str
    description: str = ""
    type: str = "TestNode"
    metadata: dict = Field(default_factory=lambda: {"type": "TestNode", "index_fields": ["name"]})

async def test_nx_adapter_basic():
    """Test basic functionality of NetworkXAdapter."""
    logger.info("Starting basic NetworkXAdapter test")
    
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
    
    # Create a test node
    test_id = uuid.uuid4()
    test_node = TestNode(
        id=test_id,
        name=f"Test Node {test_id}",
        description="A test node for debugging",
        created_at=int(datetime.now(timezone.utc).timestamp() * 1000)
    )
    
    logger.info(f"Created test node with ID: {test_id}")
    
    # Store the node
    logger.info("Storing node...")
    await adapter.add_node(test_node)
    
    # Create a second node
    test_id2 = uuid.uuid4()
    test_node2 = TestNode(
        id=test_id2,
        name=f"Test Node {test_id2}",
        description="Another test node for debugging",
        created_at=int(datetime.now(timezone.utc).timestamp() * 1000)
    )
    
    logger.info(f"Created second test node with ID: {test_id2}")
    
    # Store the second node
    logger.info("Storing second node...")
    await adapter.add_node(test_node2)
    
    # Create an edge between the nodes
    logger.info(f"Creating edge from {test_id} to {test_id2}...")
    await adapter.add_edge(
        str(test_id),
        str(test_id2),
        "TEST_RELATIONSHIP",
        {"property1": "value1", "property2": "value2"}
    )
    
    # Save the graph to file
    if hasattr(adapter, 'save_graph_to_file'):
        logger.info(f"Saving graph to {adapter.filename}")
        await adapter.save_graph_to_file(adapter.filename)
    
    # Load the graph from file
    if hasattr(adapter, 'load_graph_from_file'):
        logger.info(f"Loading graph from {adapter.filename}")
        await adapter.load_graph_from_file()
    
    # Verify the nodes and edge exist
    logger.info("Verifying nodes and edge...")
    
    # Check if the first node exists
    node1 = await adapter.extract_node(test_id)
    if node1:
        logger.info(f"Node 1 found: {node1}")
    else:
        logger.error(f"Node 1 not found")
    
    # Check if the second node exists
    node2 = await adapter.extract_node(test_id2)
    if node2:
        logger.info(f"Node 2 found: {node2}")
    else:
        logger.error(f"Node 2 not found")
    
    # Check if the edge exists
    has_edge = await adapter.has_edge(str(test_id), str(test_id2), "TEST_RELATIONSHIP")
    logger.info(f"Edge exists: {has_edge}")
    
    # Get graph data
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
    logger.info(f"Deleting test nodes...")
    await adapter.delete_node(test_id)
    await adapter.delete_node(test_id2)
    
    logger.info("Basic test completed")

if __name__ == "__main__":
    asyncio.run(test_nx_adapter_basic()) 