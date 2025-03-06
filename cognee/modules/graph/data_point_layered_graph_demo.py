#!/usr/bin/env python3
"""
Demo script for the DataPoint-based layered knowledge graph implementation.

This script demonstrates the core functionality of the layered graph system
using the DataPoint-based implementation.
"""

import asyncio
import logging
import sys
from uuid import uuid4
from datetime import datetime

from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode,
    GraphEdge,
    GraphLayer,
    LayeredKnowledgeGraphDP
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


async def create_technology_stack_graph():
    """
    Create a layered knowledge graph representing a technology stack.
    
    This example creates a graph with multiple layers:
    1. Infrastructure layer (base)
    2. Backend services layer (depends on infrastructure)
    3. Frontend applications layer (depends on backend)
    4. User layer (depends on frontend)
    
    Returns:
        The created layered knowledge graph
    """
    logger.info("Creating technology stack layered knowledge graph...")
    
    # Create the graph
    graph = LayeredKnowledgeGraphDP.create_empty(
        name="Technology Stack Graph",
        description="A layered knowledge graph representing a technology stack",
        metadata={"domain": "technology", "created_at": datetime.now().isoformat()}
    )
    
    # Create layers
    infrastructure_layer = GraphLayer(
        id=uuid4(),
        name="Infrastructure",
        description="Base infrastructure components",
        layer_type="infrastructure"
    )
    
    backend_layer = GraphLayer(
        id=uuid4(),
        name="Backend",
        description="Backend services and APIs",
        layer_type="backend",
        parent_layers=[infrastructure_layer.id]
    )
    
    frontend_layer = GraphLayer(
        id=uuid4(),
        name="Frontend",
        description="Frontend applications and UIs",
        layer_type="frontend",
        parent_layers=[backend_layer.id]
    )
    
    user_layer = GraphLayer(
        id=uuid4(),
        name="User",
        description="User interactions and journeys",
        layer_type="user",
        parent_layers=[frontend_layer.id]
    )
    
    # Add layers to graph
    graph.add_layer(infrastructure_layer)
    graph.add_layer(backend_layer)
    graph.add_layer(frontend_layer)
    graph.add_layer(user_layer)
    
    logger.info(f"Created {len(graph.layers)} layers")
    
    # Add infrastructure nodes
    db_node = GraphNode(
        id=uuid4(),
        name="PostgreSQL",
        node_type="Database",
        description="PostgreSQL database server",
        properties={"version": "14.2", "cores": 8, "memory": "32GB"}
    )
    
    redis_node = GraphNode(
        id=uuid4(),
        name="Redis",
        node_type="Cache",
        description="Redis cache server",
        properties={"version": "6.2.6", "memory": "16GB"}
    )
    
    k8s_node = GraphNode(
        id=uuid4(),
        name="Kubernetes",
        node_type="Orchestration",
        description="Kubernetes cluster",
        properties={"version": "1.24", "nodes": 5}
    )
    
    # Add infrastructure nodes to layer
    graph.add_node_to_layer(db_node, infrastructure_layer.id)
    graph.add_node_to_layer(redis_node, infrastructure_layer.id)
    graph.add_node_to_layer(k8s_node, infrastructure_layer.id)
    
    # Add infrastructure edges
    k8s_db_edge = GraphEdge(
        source_node_id=k8s_node.id,
        target_node_id=db_node.id,
        relationship_name="MANAGES",
        properties={"persistent_volume": "true"}
    )
    
    k8s_redis_edge = GraphEdge(
        source_node_id=k8s_node.id,
        target_node_id=redis_node.id,
        relationship_name="MANAGES",
        properties={"persistent_volume": "false"}
    )
    
    # Add infrastructure edges to layer
    graph.add_edge_to_layer(k8s_db_edge, infrastructure_layer.id)
    graph.add_edge_to_layer(k8s_redis_edge, infrastructure_layer.id)
    
    logger.info(f"Added {len(graph.get_layer_nodes(infrastructure_layer.id))} nodes and "
                f"{len(graph.get_layer_edges(infrastructure_layer.id))} edges to infrastructure layer")
    
    # Add backend nodes
    api_node = GraphNode(
        id=uuid4(),
        name="REST API",
        node_type="Service",
        description="REST API service",
        properties={"language": "Python", "framework": "FastAPI"}
    )
    
    auth_node = GraphNode(
        id=uuid4(),
        name="Auth Service",
        node_type="Service",
        description="Authentication service",
        properties={"language": "Go", "jwt": "true"}
    )
    
    ml_node = GraphNode(
        id=uuid4(),
        name="ML Service",
        node_type="Service",
        description="Machine learning service",
        properties={"framework": "PyTorch", "gpu": "true"}
    )
    
    # Add backend nodes to layer
    graph.add_node_to_layer(api_node, backend_layer.id)
    graph.add_node_to_layer(auth_node, backend_layer.id)
    graph.add_node_to_layer(ml_node, backend_layer.id)
    
    # Add backend edges
    api_db_edge = GraphEdge(
        source_node_id=api_node.id,
        target_node_id=db_node.id,
        relationship_name="USES",
        properties={"connection_pool": "20"}
    )
    
    api_redis_edge = GraphEdge(
        source_node_id=api_node.id,
        target_node_id=redis_node.id,
        relationship_name="USES",
        properties={"for": "caching"}
    )
    
    auth_db_edge = GraphEdge(
        source_node_id=auth_node.id,
        target_node_id=db_node.id,
        relationship_name="USES",
        properties={"for": "user_storage"}
    )
    
    ml_api_edge = GraphEdge(
        source_node_id=ml_node.id,
        target_node_id=api_node.id,
        relationship_name="PROVIDES_DATA_TO",
        properties={"format": "json"}
    )
    
    # Add backend edges to layer
    graph.add_edge_to_layer(api_db_edge, backend_layer.id)
    graph.add_edge_to_layer(api_redis_edge, backend_layer.id)
    graph.add_edge_to_layer(auth_db_edge, backend_layer.id)
    graph.add_edge_to_layer(ml_api_edge, backend_layer.id)
    
    logger.info(f"Added {len(graph.get_layer_nodes(backend_layer.id))} nodes and "
                f"{len(graph.get_layer_edges(backend_layer.id))} edges to backend layer")
    
    # Add frontend nodes
    web_node = GraphNode(
        id=uuid4(),
        name="Web App",
        node_type="Application",
        description="Web application",
        properties={"framework": "React", "version": "18.2"}
    )
    
    mobile_node = GraphNode(
        id=uuid4(),
        name="Mobile App",
        node_type="Application",
        description="Mobile application",
        properties={"framework": "React Native", "platforms": "iOS, Android"}
    )
    
    # Add frontend nodes to layer
    graph.add_node_to_layer(web_node, frontend_layer.id)
    graph.add_node_to_layer(mobile_node, frontend_layer.id)
    
    # Add frontend edges
    web_api_edge = GraphEdge(
        source_node_id=web_node.id,
        target_node_id=api_node.id,
        relationship_name="CONSUMES",
        properties={"protocol": "https"}
    )
    
    web_auth_edge = GraphEdge(
        source_node_id=web_node.id,
        target_node_id=auth_node.id,
        relationship_name="AUTHENTICATES_WITH",
        properties={}
    )
    
    mobile_api_edge = GraphEdge(
        source_node_id=mobile_node.id,
        target_node_id=api_node.id,
        relationship_name="CONSUMES",
        properties={"protocol": "https"}
    )
    
    mobile_auth_edge = GraphEdge(
        source_node_id=mobile_node.id,
        target_node_id=auth_node.id,
        relationship_name="AUTHENTICATES_WITH",
        properties={}
    )
    
    # Add frontend edges to layer
    graph.add_edge_to_layer(web_api_edge, frontend_layer.id)
    graph.add_edge_to_layer(web_auth_edge, frontend_layer.id)
    graph.add_edge_to_layer(mobile_api_edge, frontend_layer.id)
    graph.add_edge_to_layer(mobile_auth_edge, frontend_layer.id)
    
    logger.info(f"Added {len(graph.get_layer_nodes(frontend_layer.id))} nodes and "
                f"{len(graph.get_layer_edges(frontend_layer.id))} edges to frontend layer")
    
    # Add user nodes
    admin_node = GraphNode(
        id=uuid4(),
        name="Admin",
        node_type="User",
        description="Administrator user",
        properties={"permissions": "all"}
    )
    
    customer_node = GraphNode(
        id=uuid4(),
        name="Customer",
        node_type="User",
        description="Customer user",
        properties={"permissions": "limited"}
    )
    
    # Add user nodes to layer
    graph.add_node_to_layer(admin_node, user_layer.id)
    graph.add_node_to_layer(customer_node, user_layer.id)
    
    # Add user edges
    admin_web_edge = GraphEdge(
        source_node_id=admin_node.id,
        target_node_id=web_node.id,
        relationship_name="USES",
        properties={"frequency": "daily"}
    )
    
    customer_mobile_edge = GraphEdge(
        source_node_id=customer_node.id,
        target_node_id=mobile_node.id,
        relationship_name="USES",
        properties={"frequency": "weekly"}
    )
    
    customer_web_edge = GraphEdge(
        source_node_id=customer_node.id,
        target_node_id=web_node.id,
        relationship_name="USES",
        properties={"frequency": "monthly"}
    )
    
    # Add user edges to layer
    graph.add_edge_to_layer(admin_web_edge, user_layer.id)
    graph.add_edge_to_layer(customer_mobile_edge, user_layer.id)
    graph.add_edge_to_layer(customer_web_edge, user_layer.id)
    
    logger.info(f"Added {len(graph.get_layer_nodes(user_layer.id))} nodes and "
                f"{len(graph.get_layer_edges(user_layer.id))} edges to user layer")
    
    return graph


async def analyze_graph(graph: LayeredKnowledgeGraphDP):
    """
    Analyze a layered knowledge graph and print insights.
    
    Args:
        graph: The layered knowledge graph to analyze
    """
    logger.info("\n=== Layered Graph Analysis ===")
    
    # Print layer information
    logger.info("\nLayers in order:")
    for i, layer_id in enumerate(graph.layers):
        layer = graph._get_layer(layer_id)
        parent_layer_names = []
        for parent_id in layer.parent_layers:
            parent_layer_names.append(graph._get_layer(parent_id).name)
        
        parent_info = f"Parent layers: {', '.join(parent_layer_names)}" if parent_layer_names else "Base layer"
        logger.info(f"{i+1}. {layer.name} ({layer.layer_type}) - {parent_info}")
    
    # Print node and edge counts per layer
    logger.info("\nNode and edge counts per layer:")
    for layer_id in graph.layers:
        layer = graph._get_layer(layer_id)
        nodes = graph.get_layer_nodes(layer_id)
        edges = graph.get_layer_edges(layer_id)
        logger.info(f"{layer.name}: {len(nodes)} nodes, {len(edges)} edges")
    
    # Print node and edge counts for cumulative layers
    logger.info("\nCumulative node and edge counts:")
    for i, layer_id in enumerate(graph.layers):
        layer = graph._get_layer(layer_id)
        kg = graph.get_cumulative_layer_graph(layer_id)
        logger.info(f"Up to {layer.name}: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    
    # Print node types per layer
    logger.info("\nNode types per layer:")
    for layer_id in graph.layers:
        layer = graph._get_layer(layer_id)
        nodes = graph.get_layer_nodes(layer_id)
        node_types = {}
        for node in nodes:
            if node.node_type not in node_types:
                node_types[node.node_type] = 0
            node_types[node.node_type] += 1
        
        type_info = ", ".join([f"{type_name}: {count}" for type_name, count in node_types.items()])
        logger.info(f"{layer.name}: {type_info}")
    
    # Print relationship types per layer
    logger.info("\nRelationship types per layer:")
    for layer_id in graph.layers:
        layer = graph._get_layer(layer_id)
        edges = graph.get_layer_edges(layer_id)
        rel_types = {}
        for edge in edges:
            if edge.relationship_name not in rel_types:
                rel_types[edge.relationship_name] = 0
            rel_types[edge.relationship_name] += 1
        
        rel_info = ", ".join([f"{rel_name}: {count}" for rel_name, count in rel_types.items()])
        logger.info(f"{layer.name}: {rel_info}")
    
    # Print cross-layer relationships
    logger.info("\nCross-layer relationships:")
    # Skip the first layer (no parents)
    layer_ids = list(graph.layers.keys())
    if len(layer_ids) > 1:
        for layer_id in layer_ids[1:]:  
            layer = graph.get_layer(layer_id)
            edges = graph.get_layer_edges(layer_id)
            
            cross_layer_edges = []
            for edge in edges:
                source_node = graph.nodes[edge.source_node_id]
                target_node = graph.nodes[edge.target_node_id]
                
                # Check if these nodes are from different layers
                source_layer = graph.node_layer_map.get(edge.source_node_id)
                target_layer = graph.node_layer_map.get(edge.target_node_id)
                
                if source_layer == layer_id and target_layer != layer_id:
                    target_layer_name = graph.get_layer(target_layer).name
                    cross_layer_edges.append({
                        "source": source_node.name,
                        "target": target_node.name,
                        "relationship": edge.relationship_name,
                        "target_layer": target_layer_name
                    })
            
            if cross_layer_edges:
                logger.info(f"{layer.name} layer has {len(cross_layer_edges)} cross-layer relationships:")
                for edge_info in cross_layer_edges:
                    logger.info(f"  {edge_info['source']} --[{edge_info['relationship']}]--> "
                               f"{edge_info['target']} (in {edge_info['target_layer']} layer)")
            else:
                logger.info(f"{layer.name} layer has no cross-layer relationships")


async def test_serialization(graph: LayeredKnowledgeGraphDP):
    """
    Test serialization and deserialization of a layered knowledge graph.
    
    Args:
        graph: The layered knowledge graph to test
    """
    logger.info("\n=== Testing Serialization ===")
    
    # Serialize to dictionary
    logger.info("Serializing graph to dictionary...")
    serialized = graph.to_dict()
    
    # Print serialized data summary
    logger.info(f"Serialized data has {len(serialized['metadata'])} metadata fields, "
               f"{len(serialized['layers'])} layers, "
               f"{len(serialized['nodes'])} nodes, "
               f"{len(serialized['edges'])} edges")
    
    # Deserialize back to graph
    logger.info("Deserializing back to graph...")
    deserialized_graph = LayeredKnowledgeGraphDP.from_dict(serialized)
    
    # Verify the deserialized graph
    logger.info(f"Deserialized graph name: {deserialized_graph.name}")
    logger.info(f"Deserialized graph layers: {len(deserialized_graph.layers)}")
    
    # Compare node counts
    original_node_count = 0
    deserialized_node_count = 0
    for layer_id in graph.layers:
        original_node_count += len(graph.get_layer_nodes(layer_id))
    
    for layer_id in deserialized_graph.layers:
        deserialized_node_count += len(deserialized_graph.get_layer_nodes(layer_id))
    
    logger.info(f"Original node count: {original_node_count}")
    logger.info(f"Deserialized node count: {deserialized_node_count}")
    
    # Compare edge counts
    original_edge_count = 0
    deserialized_edge_count = 0
    for layer_id in graph.layers:
        original_edge_count += len(graph.get_layer_edges(layer_id))
    
    for layer_id in deserialized_graph.layers:
        deserialized_edge_count += len(deserialized_graph.get_layer_edges(layer_id))
    
    logger.info(f"Original edge count: {original_edge_count}")
    logger.info(f"Deserialized edge count: {deserialized_edge_count}")
    
    # Test getting a cumulative graph from the deserialized graph
    # Get the last layer ID (assuming we want the layer with the highest dependency)
    layer_ids = list(deserialized_graph.layers.keys())
    if layer_ids:
        last_layer_id = layer_ids[-1]
        cumulative_graph = deserialized_graph.get_cumulative_layer_graph(last_layer_id)
        logger.info(f"Cumulative graph for last layer: {len(cumulative_graph.nodes)} nodes, {len(cumulative_graph.edges)} edges")


async def main():
    """Main function."""
    # Create a layered knowledge graph
    graph = await create_technology_stack_graph()
    
    # Analyze the graph
    await analyze_graph(graph)
    
    # Test serialization
    await test_serialization(graph)
    
    logger.info("\nDemo completed successfully.")


if __name__ == "__main__":
    asyncio.run(main()) 