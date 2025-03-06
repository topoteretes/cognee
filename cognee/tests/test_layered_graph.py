"""
Unit tests for the layered graph implementation.
"""

import pytest
import uuid
from typing import Dict, List, Any

from cognee.shared.data_models import (
    KnowledgeGraph,
    LayeredKnowledgeGraph,
    Node,
    Edge,
    Layer
)
from cognee.modules.graph.layered_graph_builder import LayeredGraphBuilder, convert_to_layered_graph
from cognee.modules.graph.layered_graph_service import LayeredGraphService


def create_test_knowledge_graph() -> KnowledgeGraph:
    """
    Create a simple test knowledge graph.
    
    Returns:
        A test knowledge graph
    """
    nodes = [
        Node(id="1", name="Entity 1", type="TestEntity", description="Test entity 1"),
        Node(id="2", name="Entity 2", type="TestEntity", description="Test entity 2"),
        Node(id="3", name="Entity 3", type="TestEntity", description="Test entity 3")
    ]
    
    edges = [
        Edge(source_node_id="1", target_node_id="2", relationship_name="RELATES_TO"),
        Edge(source_node_id="2", target_node_id="3", relationship_name="RELATES_TO")
    ]
    
    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        name="Test Graph",
        description="A test knowledge graph"
    )


@pytest.fixture
def layered_graph_fixture() -> Dict[str, Any]:
    """
    Fixture that creates a layered knowledge graph with multiple layers for testing.
    
    Returns:
        Dictionary with layered graph and layer IDs
    """
    # Create builder
    builder = LayeredGraphBuilder(
        name="Test Layered Graph",
        description="A test layered knowledge graph"
    )
    
    # Create base layer
    base_layer_id = builder.create_layer(
        name="Base Layer",
        description="Base layer for testing",
        layer_type="base"
    )
    
    # Add base nodes
    builder.add_node_to_layer(
        layer_id=base_layer_id,
        node_id="A1",
        name="Base Node 1",
        node_type="BaseNode",
        description="A base node"
    )
    
    builder.add_node_to_layer(
        layer_id=base_layer_id,
        node_id="A2",
        name="Base Node 2",
        node_type="BaseNode",
        description="Another base node"
    )
    
    # Add base edge
    builder.add_edge_to_layer(
        layer_id=base_layer_id,
        source_node_id="A1",
        target_node_id="A2",
        relationship_name="BASE_RELATION"
    )
    
    # Create second layer
    second_layer_id = builder.create_layer(
        name="Second Layer",
        description="Second layer for testing",
        layer_type="enrichment",
        parent_layers=[base_layer_id]
    )
    
    # Add second layer nodes
    builder.add_node_to_layer(
        layer_id=second_layer_id,
        node_id="B1",
        name="Second Node 1",
        node_type="SecondNode",
        description="A second layer node"
    )
    
    # Add second layer edges
    builder.add_edge_to_layer(
        layer_id=second_layer_id,
        source_node_id="A1",
        target_node_id="B1",
        relationship_name="CONNECTS_TO"
    )
    
    # Create third layer
    third_layer_id = builder.create_layer(
        name="Third Layer",
        description="Third layer for testing",
        layer_type="inference",
        parent_layers=[second_layer_id]
    )
    
    # Add third layer nodes
    builder.add_node_to_layer(
        layer_id=third_layer_id,
        node_id="C1",
        name="Third Node 1",
        node_type="ThirdNode",
        description="A third layer node"
    )
    
    # Add third layer edges
    builder.add_edge_to_layer(
        layer_id=third_layer_id,
        source_node_id="B1",
        target_node_id="C1",
        relationship_name="INFERS"
    )
    
    # Build the layered graph
    layered_graph = builder.build()
    
    return {
        "layered_graph": layered_graph,
        "base_layer_id": base_layer_id,
        "second_layer_id": second_layer_id,
        "third_layer_id": third_layer_id
    }


@pytest.mark.asyncio
async def test_layered_graph_creation():
    """
    Test that a layered graph can be created with the builder.
    """
    # Create a simple layered graph
    builder = LayeredGraphBuilder(
        name="Simple Layered Graph",
        description="A simple layered graph for testing"
    )
    
    # Create a layer
    layer_id = builder.create_layer(
        name="Test Layer",
        description="Test layer",
        layer_type="test"
    )
    
    # Add nodes and edges
    builder.add_node_to_layer(
        layer_id=layer_id,
        node_id="test1",
        name="Test Node 1",
        node_type="TestNode",
        description="A test node"
    )
    
    builder.add_node_to_layer(
        layer_id=layer_id,
        node_id="test2",
        name="Test Node 2",
        node_type="TestNode",
        description="Another test node"
    )
    
    builder.add_edge_to_layer(
        layer_id=layer_id,
        source_node_id="test1",
        target_node_id="test2",
        relationship_name="TEST_RELATION"
    )
    
    # Build the graph
    graph = builder.build()
    
    # Check the graph properties
    assert graph.name == "Simple Layered Graph"
    assert len(graph.layers) == 1
    assert graph.layers[0].name == "Test Layer"
    
    # Get layer graph
    layer_graph = graph.get_layer_graph(layer_id)
    assert len(layer_graph.nodes) == 2
    assert len(layer_graph.edges) == 1


@pytest.mark.asyncio
async def test_layered_graph_layer_hierarchy(layered_graph_fixture):
    """
    Test that the layer hierarchy is maintained in the layered graph.
    """
    layered_graph = layered_graph_fixture["layered_graph"]
    base_layer_id = layered_graph_fixture["base_layer_id"]
    second_layer_id = layered_graph_fixture["second_layer_id"]
    third_layer_id = layered_graph_fixture["third_layer_id"]
    
    # Check the layers
    assert len(layered_graph.layers) == 3
    
    # Get layers by ID
    base_layer = next((layer for layer in layered_graph.layers if layer.id == base_layer_id), None)
    second_layer = next((layer for layer in layered_graph.layers if layer.id == second_layer_id), None)
    third_layer = next((layer for layer in layered_graph.layers if layer.id == third_layer_id), None)
    
    # Check layer hierarchy
    assert base_layer is not None
    assert second_layer is not None
    assert third_layer is not None
    
    assert len(base_layer.parent_layers) == 0
    assert second_layer.parent_layers == [base_layer_id]
    assert third_layer.parent_layers == [second_layer_id]


@pytest.mark.asyncio
async def test_layered_graph_get_layer_graph(layered_graph_fixture):
    """
    Test retrieving individual layer graphs.
    """
    layered_graph = layered_graph_fixture["layered_graph"]
    base_layer_id = layered_graph_fixture["base_layer_id"]
    second_layer_id = layered_graph_fixture["second_layer_id"]
    third_layer_id = layered_graph_fixture["third_layer_id"]
    
    # Get individual layer graphs
    base_graph = layered_graph.get_layer_graph(base_layer_id)
    second_graph = layered_graph.get_layer_graph(second_layer_id)
    third_graph = layered_graph.get_layer_graph(third_layer_id)
    
    # Check base layer graph
    assert len(base_graph.nodes) == 2
    assert len(base_graph.edges) == 1
    assert any(node.id == "A1" for node in base_graph.nodes)
    assert any(node.id == "A2" for node in base_graph.nodes)
    assert any(edge.source_node_id == "A1" and edge.target_node_id == "A2" for edge in base_graph.edges)
    
    # Check second layer graph
    assert len(second_graph.nodes) == 1
    assert len(second_graph.edges) == 1
    assert any(node.id == "B1" for node in second_graph.nodes)
    assert any(edge.source_node_id == "A1" and edge.target_node_id == "B1" for edge in second_graph.edges)
    
    # Check third layer graph
    assert len(third_graph.nodes) == 1
    assert len(third_graph.edges) == 1
    assert any(node.id == "C1" for node in third_graph.nodes)
    assert any(edge.source_node_id == "B1" and edge.target_node_id == "C1" for edge in third_graph.edges)


@pytest.mark.asyncio
async def test_layered_graph_get_cumulative_graph(layered_graph_fixture):
    """
    Test retrieving cumulative layer graphs (including parent layers).
    """
    layered_graph = layered_graph_fixture["layered_graph"]
    base_layer_id = layered_graph_fixture["base_layer_id"]
    second_layer_id = layered_graph_fixture["second_layer_id"]
    third_layer_id = layered_graph_fixture["third_layer_id"]
    
    # Get cumulative layer graphs
    base_cumulative = layered_graph.get_cumulative_layer_graph(base_layer_id)
    second_cumulative = layered_graph.get_cumulative_layer_graph(second_layer_id)
    third_cumulative = layered_graph.get_cumulative_layer_graph(third_layer_id)
    
    # Check base cumulative (same as base layer)
    assert len(base_cumulative.nodes) == 2
    assert len(base_cumulative.edges) == 1
    
    # Check second cumulative (base + second)
    assert len(second_cumulative.nodes) == 3  # A1, A2, B1
    assert len(second_cumulative.edges) == 2  # A1->A2, A1->B1
    
    # Check third cumulative (base + second + third)
    assert len(third_cumulative.nodes) == 4  # A1, A2, B1, C1
    assert len(third_cumulative.edges) == 3  # A1->A2, A1->B1, B1->C1


@pytest.mark.asyncio
async def test_add_subgraph_to_layer():
    """
    Test adding an entire subgraph to a layer.
    """
    # Create a test graph to add
    test_graph = create_test_knowledge_graph()
    
    # Create a layered graph
    builder = LayeredGraphBuilder(
        name="Subgraph Test",
        description="Testing adding subgraphs"
    )
    
    # Create a layer
    layer_id = builder.create_layer(
        name="Subgraph Layer",
        description="Layer for subgraph",
        layer_type="test"
    )
    
    # Add the subgraph to the layer
    id_mapping = builder.add_subgraph_to_layer(layer_id, test_graph, id_prefix="sub_")
    
    # Build the graph
    layered_graph = builder.build()
    
    # Get the layer graph
    layer_graph = layered_graph.get_layer_graph(layer_id)
    
    # Check the results
    assert len(layer_graph.nodes) == 3
    assert len(layer_graph.edges) == 2
    
    # Check ID mapping
    assert len(id_mapping) == 3
    assert id_mapping["1"] == "sub_1"
    assert id_mapping["2"] == "sub_2"
    assert id_mapping["3"] == "sub_3"
    
    # Check that edge relationships are preserved
    edge_relationships = [(edge.source_node_id, edge.target_node_id, edge.relationship_name) 
                         for edge in layer_graph.edges]
    assert ("sub_1", "sub_2", "RELATES_TO") in edge_relationships
    assert ("sub_2", "sub_3", "RELATES_TO") in edge_relationships


@pytest.mark.asyncio
async def test_convert_regular_graph_to_layered():
    """
    Test converting a regular knowledge graph to a layered graph.
    """
    # Create a test graph
    test_graph = create_test_knowledge_graph()
    
    # Convert to layered graph
    layered_graph = await convert_to_layered_graph(
        knowledge_graph=test_graph,
        layer_name="Converted Layer",
        layer_description="Converted from regular graph",
        graph_name="Converted Graph",
        graph_description="Test converted graph"
    )
    
    # Check the layered graph
    assert layered_graph.name == "Converted Graph"
    assert len(layered_graph.layers) == 1
    assert layered_graph.layers[0].name == "Converted Layer"
    
    # Get the layer graph
    layer_id = layered_graph.layers[0].id
    layer_graph = layered_graph.get_layer_graph(layer_id)
    
    # Check the layer contents
    assert len(layer_graph.nodes) == 3
    assert len(layer_graph.edges) == 2
    
    # Check that node and edge data is preserved
    node_ids = [node.id for node in layer_graph.nodes]
    assert "1" in node_ids
    assert "2" in node_ids
    assert "3" in node_ids
    
    edge_relationships = [(edge.source_node_id, edge.target_node_id) for edge in layer_graph.edges]
    assert ("1", "2") in edge_relationships
    assert ("2", "3") in edge_relationships


@pytest.mark.asyncio
async def test_layered_graph_service_diff_layers(layered_graph_fixture):
    """
    Test the diff_layers method of LayeredGraphService.
    """
    layered_graph = layered_graph_fixture["layered_graph"]
    base_layer_id = layered_graph_fixture["base_layer_id"]
    second_layer_id = layered_graph_fixture["second_layer_id"]
    
    # Compare base and second layers
    diff_result = await LayeredGraphService.diff_layers(
        layered_graph=layered_graph,
        base_layer_id=base_layer_id,
        comparison_layer_id=second_layer_id
    )
    
    # Check diff results
    assert len(diff_result["added_nodes"]) == 1
    assert "B1" in diff_result["added_nodes"]
    
    assert len(diff_result["removed_nodes"]) == 2
    assert "A1" in diff_result["removed_nodes"]
    assert "A2" in diff_result["removed_nodes"]
    
    assert len(diff_result["added_edges"]) == 1
    assert any(edge[0] == "A1" and edge[1] == "B1" for edge in diff_result["added_edges"])
    
    assert len(diff_result["removed_edges"]) == 1
    assert any(edge[0] == "A1" and edge[1] == "A2" for edge in diff_result["removed_edges"])


@pytest.mark.asyncio
async def test_layered_graph_service_merge_layers(layered_graph_fixture):
    """
    Test the merge_layers method of LayeredGraphService.
    """
    layered_graph = layered_graph_fixture["layered_graph"]
    base_layer_id = layered_graph_fixture["base_layer_id"]
    second_layer_id = layered_graph_fixture["second_layer_id"]
    
    # Merge the base and second layers
    new_layer_id, merged_graph = await LayeredGraphService.merge_layers(
        layered_graph=layered_graph,
        layer_ids=[base_layer_id, second_layer_id],
        new_layer_name="Merged Layer",
        new_layer_description="Merged from base and second layers",
        new_layer_type="merged"
    )
    
    # Check the merged graph
    assert len(merged_graph.layers) == 2  # Merged layer and third layer
    
    # Get the merged layer graph
    merged_layer_graph = merged_graph.get_layer_graph(new_layer_id)
    
    # Check merged layer contents
    assert len(merged_layer_graph.nodes) == 3  # A1, A2, B1
    assert len(merged_layer_graph.edges) == 2  # A1->A2, A1->B1


@pytest.mark.asyncio
async def test_layered_graph_service_extract_subgraph(layered_graph_fixture):
    """
    Test the extract_subgraph method of LayeredGraphService.
    """
    layered_graph = layered_graph_fixture["layered_graph"]
    base_layer_id = layered_graph_fixture["base_layer_id"]
    second_layer_id = layered_graph_fixture["second_layer_id"]
    
    # Extract a subgraph with filtered nodes
    subgraph = await LayeredGraphService.extract_subgraph(
        layered_graph=layered_graph,
        layer_ids=[base_layer_id, second_layer_id],
        include_cumulative=True,
        node_filter=lambda node: node.type == "BaseNode"
    )
    
    # Check subgraph contents
    assert len(subgraph.nodes) == 2  # A1, A2 (BaseNode type)
    assert len(subgraph.edges) == 1  # A1->A2


@pytest.mark.asyncio
async def test_layered_graph_service_calculate_metrics(layered_graph_fixture):
    """
    Test the calculate_layer_metrics method of LayeredGraphService.
    """
    layered_graph = layered_graph_fixture["layered_graph"]
    
    # Calculate metrics
    metrics = await LayeredGraphService.calculate_layer_metrics(layered_graph)
    
    # Check metrics keys
    for layer in layered_graph.layers:
        assert layer.id in metrics
        layer_metrics = metrics[layer.id]
        
        # Check required metrics
        assert "node_count" in layer_metrics
        assert "edge_count" in layer_metrics
        assert "cumulative_node_count" in layer_metrics
        assert "cumulative_edge_count" in layer_metrics


@pytest.mark.asyncio
async def test_layered_graph_service_sort_layers(layered_graph_fixture):
    """
    Test the sort_layers_topologically method of LayeredGraphService.
    """
    layered_graph = layered_graph_fixture["layered_graph"]
    base_layer_id = layered_graph_fixture["base_layer_id"]
    second_layer_id = layered_graph_fixture["second_layer_id"]
    third_layer_id = layered_graph_fixture["third_layer_id"]
    
    # Sort layers
    sorted_layers = await LayeredGraphService.sort_layers_topologically(layered_graph)
    
    # Check the order (base -> second -> third)
    assert len(sorted_layers) == 3
    assert sorted_layers[0] == base_layer_id
    assert sorted_layers[1] == second_layer_id
    assert sorted_layers[2] == third_layer_id 