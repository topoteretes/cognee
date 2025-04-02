"""
Example demonstrating how to use the simplified LayeredKnowledgeGraph with database adapters.

This example shows how to:
1. Create a layered knowledge graph
2. Add nodes, edges, and layers
3. Retrieve layer data
4. Work with cumulative layers
"""

import asyncio
import logging
import uuid
from uuid import UUID
import os

from cognee.modules.graph.simplified_layered_graph import (
    LayeredKnowledgeGraph,
    GraphNode,
    GraphEdge,
    GraphLayer,
)
from cognee.modules.graph.enhanced_layered_graph_adapter import LayeredGraphDBAdapter
from cognee.infrastructure.databases.graph.networkx.adapter import NetworkXAdapter
from cognee.infrastructure.databases.graph import get_graph_engine

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main():
    print("Starting simplified layered graph example")

    # Initialize file path for the NetworkXAdapter
    db_dir = os.path.join(os.path.expanduser("~"), "cognee/cognee/.cognee_system/databases")
    os.makedirs(db_dir, exist_ok=True)
    db_file = os.path.join(db_dir, "cognee_graph.pkl")

    # Use NetworkXAdapter for the graph database
    adapter = NetworkXAdapter(filename=db_file)

    # Initialize the adapter by creating or loading the graph
    if not os.path.exists(db_file):
        await adapter.create_empty_graph(db_file)
    await adapter.load_graph_from_file()

    print(f"Using graph database adapter: {adapter.__class__.__name__}")

    # Create an empty graph
    graph = LayeredKnowledgeGraph.create_empty("Test Graph")
    graph.set_adapter(LayeredGraphDBAdapter(adapter))
    print(f"Created graph with ID: {graph.id}")

    # Create layers
    base_layer = await graph.add_layer(
        name="Base Layer", description="The foundation layer with base concepts", layer_type="base"
    )
    print(f"Added base layer with ID: {base_layer.id}")

    intermediate_layer = await graph.add_layer(
        name="Intermediate Layer",
        description="Layer that builds on base concepts",
        layer_type="intermediate",
        parent_layers=[base_layer.id],
    )
    print(f"Added intermediate layer with ID: {intermediate_layer.id}")

    derived_layer = await graph.add_layer(
        name="Derived Layer",
        description="Final layer with derived concepts",
        layer_type="derived",
        parent_layers=[intermediate_layer.id],
    )
    print(f"Added derived layer with ID: {derived_layer.id}")

    # Add nodes to layers
    node1 = await graph.add_node(
        name="Base Concept A",
        node_type="concept",
        properties={"importance": "high"},
        metadata={"source": "example"},
        layer_id=base_layer.id,
    )
    print(f"Added node1 with ID: {node1.id} to layer: {base_layer.id}")

    node2 = await graph.add_node(
        name="Base Concept B",
        node_type="concept",
        properties={"importance": "medium"},
        metadata={"source": "example"},
        layer_id=base_layer.id,
    )
    print(f"Added node2 with ID: {node2.id} to layer: {base_layer.id}")

    node3 = await graph.add_node(
        name="Intermediate Concept",
        node_type="concept",
        properties={"derived_from": ["Base Concept A"]},
        metadata={"source": "example"},
        layer_id=intermediate_layer.id,
    )
    print(f"Added node3 with ID: {node3.id} to layer: {intermediate_layer.id}")

    node4 = await graph.add_node(
        name="Derived Concept",
        node_type="concept",
        properties={"derived_from": ["Intermediate Concept"]},
        metadata={"source": "example"},
        layer_id=derived_layer.id,
    )
    print(f"Added node4 with ID: {node4.id} to layer: {derived_layer.id}")

    # Add edges between nodes
    edge1 = await graph.add_edge(
        source_id=node1.id,
        target_id=node2.id,
        edge_type="RELATES_TO",
        properties={"strength": "high"},
        metadata={"source": "example"},
        layer_id=base_layer.id,
    )
    print(f"Added edge1 with ID: {edge1.id} between {node1.id} and {node2.id}")

    edge2 = await graph.add_edge(
        source_id=node1.id,
        target_id=node3.id,
        edge_type="SUPPORTS",
        properties={"confidence": 0.9},
        metadata={"source": "example"},
        layer_id=intermediate_layer.id,
    )
    print(f"Added edge2 with ID: {edge2.id} between {node1.id} and {node3.id}")

    edge3 = await graph.add_edge(
        source_id=node3.id,
        target_id=node4.id,
        edge_type="EXTENDS",
        properties={"confidence": 0.8},
        metadata={"source": "example"},
        layer_id=derived_layer.id,
    )
    print(f"Added edge3 with ID: {edge3.id} between {node3.id} and {node4.id}")

    # Save the graph to the database
    # The graph is automatically saved when nodes and edges are added,
    # but for NetworkXAdapter we'll save the file explicitly
    if hasattr(adapter, "save_graph_to_file"):
        await adapter.save_graph_to_file(adapter.filename)
    print(f"Saving graph to file: {adapter.filename}")

    # Retrieve all layers
    layers = await graph.get_layers()
    print(f"Retrieved {len(layers)} layers")

    # Load the graph from the database
    print(f"Loading graph with ID: {graph.id} from database")
    # Create a new graph instance from the database
    loaded_graph = LayeredKnowledgeGraph(id=graph.id, name="Test Graph")
    loaded_graph.set_adapter(LayeredGraphDBAdapter(adapter))
    # Load layers, which will also load nodes and edges
    loaded_layers = await loaded_graph.get_layers()
    print(f"Successfully loaded graph: {loaded_graph} with {len(loaded_layers)} layers")

    # Display contents of each layer
    print("\n===== Individual Layer Contents =====")
    for layer in layers:
        # Get nodes and edges in the layer
        nodes = await graph.get_nodes_in_layer(layer.id)
        edges = await graph.get_edges_in_layer(layer.id)

        # Print summary
        print(f"Nodes in {layer.name.lower()} layer: {[node.name for node in nodes]}")
        print(f"Edges in {layer.name.lower()} layer: {[edge.edge_type for edge in edges]}")

    # Display cumulative layer views
    print("\n===== Cumulative Layer Views =====")

    # Intermediate layer - should include base layer nodes/edges
    print("\nCumulative layer graph for intermediate layer:")
    int_nodes, int_edges = await graph.get_cumulative_layer_graph(intermediate_layer.id)
    print(f"Intermediate cumulative nodes: {[node.name for node in int_nodes]}")
    print(f"Intermediate cumulative edges: {[edge.edge_type for edge in int_edges]}")

    # Derived layer - should include all nodes/edges
    print("\nCumulative layer graph for derived layer:")
    derived_nodes, derived_edges = await graph.get_cumulative_layer_graph(derived_layer.id)
    print(f"Derived cumulative nodes: {[node.name for node in derived_nodes]}")
    print(f"Derived cumulative edges: {[edge.edge_type for edge in derived_edges]}")

    # Test helper methods
    print("\n===== Helper Method Results =====")
    base_nodes = await graph.get_nodes_in_layer(base_layer.id)
    base_edges = await graph.get_edges_in_layer(base_layer.id)
    print(f"Base layer contains {len(base_nodes)} nodes and {len(base_edges)} edges")

    print("Example complete")


if __name__ == "__main__":
    asyncio.run(main())
