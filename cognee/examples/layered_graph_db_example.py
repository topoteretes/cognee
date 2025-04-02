"""
Example demonstrating how to use LayeredKnowledgeGraphDP with database adapters.

This example shows how to:
1. Create a layered knowledge graph
2. Set a database adapter
3. Add nodes, edges, and layers with automatic persistence to the database
4. Retrieve graph data from the database
"""

import asyncio
import uuid
import logging
import json
from uuid import UUID

from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode,
    GraphEdge,
    GraphLayer,
    LayeredKnowledgeGraphDP,
)
from cognee.modules.graph.enhanced_layered_graph_adapter import LayeredGraphDBAdapter
from cognee.infrastructure.databases.graph import get_graph_engine

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def retrieve_graph_manually(graph_id, adapter):
    """Retrieve a graph manually from the NetworkX adapter."""
    graph_db = adapter._graph_db
    if not hasattr(graph_db, "graph") or not graph_db.graph:
        await graph_db.load_graph_from_file()

    graph_id_str = str(graph_id)
    logger.info(f"Looking for graph with ID: {graph_id_str}")

    if hasattr(graph_db, "graph") and graph_db.graph.has_node(graph_id_str):
        # Get the graph node data
        graph_data = graph_db.graph.nodes[graph_id_str]
        logger.info(f"Found graph node data: {graph_data}")

        # Create the graph instance
        graph = LayeredKnowledgeGraphDP(
            id=graph_id,
            name=graph_data.get("name", ""),
            description=graph_data.get("description", ""),
            metadata=graph_data.get("metadata", {}),
        )

        # Set the adapter
        graph.set_adapter(adapter)

        # Find and add all layers, nodes, and edges
        nx_graph = graph_db.graph

        # Find all layers for this graph
        logger.info("Finding layers connected to the graph")
        found_layers = set()
        for source, target, key in nx_graph.edges(graph_id_str, keys=True):
            if key == "CONTAINS_LAYER":
                # Found layer
                layer_data = nx_graph.nodes[target]
                layer_id_str = target
                logger.info(f"Found layer: {layer_id_str} with data: {layer_data}")

                # Convert parent layers
                parent_layers = []
                if "parent_layers" in layer_data:
                    try:
                        if isinstance(layer_data["parent_layers"], str):
                            import json

                            parent_layers = [
                                UUID(p) for p in json.loads(layer_data["parent_layers"])
                            ]
                        elif isinstance(layer_data["parent_layers"], list):
                            parent_layers = [UUID(str(p)) for p in layer_data["parent_layers"]]
                    except Exception as e:
                        logger.error(f"Error processing parent layers: {e}")
                        parent_layers = []

                # Create layer
                try:
                    layer = GraphLayer(
                        id=UUID(layer_id_str),
                        name=layer_data.get("name", ""),
                        description=layer_data.get("description", ""),
                        layer_type=layer_data.get("layer_type", "default"),
                        parent_layers=parent_layers,
                        properties=layer_data.get("properties", {}),
                        metadata=layer_data.get("metadata", {}),
                    )
                    graph.layers[layer.id] = layer
                    found_layers.add(layer_id_str)
                except Exception as e:
                    logger.error(f"Error creating layer object: {e}")

        # Helper function to safely get UUID
        def safe_uuid(value):
            if isinstance(value, UUID):
                return value
            try:
                return UUID(str(value))
            except Exception as e:
                logger.error(f"Error converting to UUID: {value} - {e}")
                return None

        # Find all nodes for this graph
        logger.info("Finding nodes in the layers")
        for node_id, node_data in nx_graph.nodes(data=True):
            # First check if this is a node by its metadata
            if node_data.get("metadata", {}).get("type") == "GraphNode":
                # Get the layer ID from the node data
                if "layer_id" in node_data:
                    layer_id_str = node_data["layer_id"]
                    # Check if this layer ID is in our found layers
                    try:
                        layer_id = safe_uuid(layer_id_str)
                        if layer_id and layer_id in graph.layers:
                            logger.info(f"Found node with ID {node_id} in layer {layer_id_str}")

                            # Create the node
                            node = GraphNode(
                                id=safe_uuid(node_id),
                                name=node_data.get("name", ""),
                                node_type=node_data.get("node_type", ""),
                                description=node_data.get("description", ""),
                                properties=node_data.get("properties", {}),
                                layer_id=layer_id,
                                metadata=node_data.get("metadata", {}),
                            )
                            graph.nodes[node.id] = node
                            graph.node_layer_map[node.id] = layer_id
                    except Exception as e:
                        logger.error(f"Error processing node {node_id}: {e}")

        # Find all edges for this graph
        logger.info("Finding edges in the layers")
        for node_id, node_data in nx_graph.nodes(data=True):
            # First check if this is an edge by its metadata
            if node_data.get("metadata", {}).get("type") == "GraphEdge":
                # Get the layer ID from the edge data
                if "layer_id" in node_data:
                    layer_id_str = node_data["layer_id"]
                    # Check if this layer ID is in our found layers
                    try:
                        layer_id = safe_uuid(layer_id_str)
                        if layer_id and layer_id in graph.layers:
                            source_id = safe_uuid(node_data.get("source_node_id"))
                            target_id = safe_uuid(node_data.get("target_node_id"))

                            if source_id and target_id:
                                logger.info(f"Found edge with ID {node_id} in layer {layer_id_str}")

                                # Create the edge
                                edge = GraphEdge(
                                    id=safe_uuid(node_id),
                                    source_node_id=source_id,
                                    target_node_id=target_id,
                                    relationship_name=node_data.get("relationship_name", ""),
                                    properties=node_data.get("properties", {}),
                                    layer_id=layer_id,
                                    metadata=node_data.get("metadata", {}),
                                )
                                graph.edges[edge.id] = edge
                                graph.edge_layer_map[edge.id] = layer_id
                    except Exception as e:
                        logger.error(f"Error processing edge {node_id}: {e}")

        logger.info(
            f"Manually retrieved graph with {len(graph.layers)} layers, {len(graph.nodes)} nodes, and {len(graph.edges)} edges"
        )
        return graph
    else:
        logger.error(f"Could not find graph with ID {graph_id}")
        return None


def get_nodes_and_edges_from_graph(graph, layer_id):
    """
    Get nodes and edges from a layer in the graph, avoiding database calls.

    Args:
        graph: The LayeredKnowledgeGraphDP instance
        layer_id: The UUID of the layer

    Returns:
        Tuple of (nodes, edges) lists
    """
    nodes = [node for node in graph.nodes.values() if node.layer_id == layer_id]
    edges = [edge for edge in graph.edges.values() if edge.layer_id == layer_id]
    return nodes, edges


async def main():
    logger.info("Starting layered graph database example")

    # Get the default graph engine (typically NetworkXAdapter)
    graph_db = await get_graph_engine()
    logger.info(f"Using graph database adapter: {type(graph_db).__name__}")

    # Create an adapter using the graph engine
    adapter = LayeredGraphDBAdapter(graph_db)

    # Create a new empty graph
    graph = LayeredKnowledgeGraphDP.create_empty(
        name="Example Database Graph",
        description="A graph that persists to the database",
        metadata={
            "type": "LayeredKnowledgeGraph",
            "index_fields": ["name"],
        },  # Ensure proper metadata
    )
    logger.info(f"Created graph with ID: {graph.id}")

    # Set the adapter for this graph
    graph.set_adapter(adapter)

    # Create and add a base layer
    base_layer = GraphLayer.create(
        name="Base Layer", description="The foundation layer of the graph", layer_type="base"
    )
    graph.add_layer(base_layer)
    logger.info(f"Added base layer with ID: {base_layer.id}")

    # Create and add a derived layer that extends the base layer
    derived_layer = GraphLayer.create(
        name="Derived Layer",
        description="A layer that extends the base layer",
        layer_type="derived",
        parent_layers=[base_layer.id],
    )
    graph.add_layer(derived_layer)
    logger.info(f"Added derived layer with ID: {derived_layer.id}")

    # Create and add nodes to the base layer
    node1 = GraphNode.create(
        name="Concept A", node_type="concept", description="A foundational concept"
    )
    graph.add_node(node1, base_layer.id)
    logger.info(f"Added node1 with ID: {node1.id} to layer: {base_layer.id}")

    node2 = GraphNode.create(
        name="Concept B", node_type="concept", description="Another foundational concept"
    )
    graph.add_node(node2, base_layer.id)
    logger.info(f"Added node2 with ID: {node2.id} to layer: {base_layer.id}")

    # Create and add a node to the derived layer
    node3 = GraphNode.create(
        name="Derived Concept",
        node_type="concept",
        description="A concept derived from foundational concepts",
    )
    graph.add_node(node3, derived_layer.id)
    logger.info(f"Added node3 with ID: {node3.id} to layer: {derived_layer.id}")

    # Create and add edges between nodes
    edge1 = GraphEdge.create(
        source_node_id=node1.id, target_node_id=node2.id, relationship_name="RELATES_TO"
    )
    graph.add_edge(edge1, base_layer.id)
    logger.info(f"Added edge1 with ID: {edge1.id} between {node1.id} and {node2.id}")

    edge2 = GraphEdge.create(
        source_node_id=node1.id, target_node_id=node3.id, relationship_name="EXPANDS_TO"
    )
    graph.add_edge(edge2, derived_layer.id)
    logger.info(f"Added edge2 with ID: {edge2.id} between {node1.id} and {node3.id}")

    edge3 = GraphEdge.create(
        source_node_id=node2.id, target_node_id=node3.id, relationship_name="CONTRIBUTES_TO"
    )
    graph.add_edge(edge3, derived_layer.id)
    logger.info(f"Added edge3 with ID: {edge3.id} between {node2.id} and {node3.id}")

    # Save the graph state to a file for NetworkXAdapter
    if hasattr(graph_db, "save_graph_to_file"):
        logger.info(f"Saving graph to file: {getattr(graph_db, 'filename', 'unknown')}")
        await graph_db.save_graph_to_file()

    # Persist the entire graph to the database
    # This is optional since the graph is already being persisted incrementally
    # when add_layer, add_node, and add_edge are called
    logger.info("Persisting entire graph to database")
    graph_id = await graph.persist()
    logger.info(f"Graph persisted with ID: {graph_id}")

    # Check if the graph exists in the database
    if hasattr(graph_db, "graph"):
        logger.info(f"Checking if graph exists in memory: {graph_db.graph.has_node(str(graph.id))}")

        # Check the node data
        if graph_db.graph.has_node(str(graph.id)):
            node_data = graph_db.graph.nodes[str(graph.id)]
            logger.info(f"Graph node data: {node_data}")

        # List all nodes in the graph
        logger.info("Nodes in the graph:")
        for node_id, node_data in graph_db.graph.nodes(data=True):
            logger.info(
                f"  Node {node_id}: type={node_data.get('metadata', {}).get('type', 'unknown')}"
            )

    # Try to retrieve the graph using our from_database method
    retrieved_graph = None
    try:
        logger.info(f"Retrieving graph from database with ID: {graph.id}")
        retrieved_graph = await LayeredKnowledgeGraphDP.from_database(graph.id, adapter)
        logger.info(f"Retrieved graph: {retrieved_graph}")
        logger.info(
            f"Retrieved {len(retrieved_graph.layers)} layers, {len(retrieved_graph.nodes)} nodes, and {len(retrieved_graph.edges)} edges"
        )
    except ValueError as e:
        logger.error(f"Error retrieving graph using from_database: {str(e)}")

        # Try using manual retrieval as a fallback
        logger.info("Trying manual retrieval as a fallback")
        retrieved_graph = await retrieve_graph_manually(graph.id, adapter)

        if retrieved_graph:
            logger.info(f"Successfully retrieved graph manually: {retrieved_graph}")
        else:
            logger.error("Failed to retrieve graph manually")
            return

    # Use our helper function to get nodes and edges
    if retrieved_graph:
        # Get nodes in the base layer
        base_nodes, base_edges = get_nodes_and_edges_from_graph(retrieved_graph, base_layer.id)
        logger.info(f"Nodes in base layer: {[node.name for node in base_nodes]}")
        logger.info(f"Edges in base layer: {[edge.relationship_name for edge in base_edges]}")

        # Get nodes in the derived layer
        derived_nodes, derived_edges = get_nodes_and_edges_from_graph(
            retrieved_graph, derived_layer.id
        )
        logger.info(f"Nodes in derived layer: {[node.name for node in derived_nodes]}")
        logger.info(f"Edges in derived layer: {[edge.relationship_name for edge in derived_edges]}")
    else:
        logger.error("No graph was retrieved, cannot display nodes and edges")


if __name__ == "__main__":
    asyncio.run(main())
