"""
Tests for the DataPoint-based layered knowledge graph implementation.

This module tests the integration of layered knowledge graphs with the DataPoint class.
"""

import unittest
import json
import asyncio
from uuid import UUID, uuid4

from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode,
    GraphEdge,
    GraphLayer,
    LayeredKnowledgeGraphDP
)
from cognee.shared.data_models import Node, Edge, KnowledgeGraph, Layer


class TestGraphNode(unittest.TestCase):
    """Tests for the GraphNode class."""
    
    def test_creation(self):
        """Test creating a GraphNode."""
        node = GraphNode.create(
            name="Test Node",
            node_type="TestType",
            description="A test node"
        )
        
        self.assertIsInstance(node, GraphNode)
        self.assertIsInstance(node.id, UUID)
        self.assertEqual(node.name, "Test Node")
        self.assertEqual(node.node_type, "TestType")
        self.assertEqual(node.description, "A test node")
        self.assertEqual(node.properties, {})
        self.assertIsNone(node.layer_id)
        
        # Check metadata
        self.assertIn("type", node.metadata)
        self.assertEqual(node.metadata["type"], "GraphNode")
        self.assertIn("index_fields", node.metadata)
    
    def test_conversion_from_basic_node(self):
        """Test converting a basic Node to a GraphNode."""
        # Create a basic Node with valid UUID format for ID
        basic_node = Node(
            id=str(uuid4()),  # Use valid UUID string
            name="Basic Node",
            type="TestType",
            description="A basic node for testing"
        )
        
        node = GraphNode.from_basic_node(basic_node)
        
        self.assertIsInstance(node, GraphNode)
        self.assertEqual(node.name, "Basic Node")
        self.assertEqual(node.node_type, "TestType")
        self.assertEqual(node.description, "A basic node for testing")
        self.assertEqual(node.properties, {})
        self.assertIsNone(node.layer_id)
    
    def test_serialization(self):
        """Test serialization and deserialization of a GraphNode."""
        original_node = GraphNode.create(
            name="Serialization Test",
            node_type="TestType",
            description="Testing serialization",
            properties={"test_key": "test_value"}
        )
        
        # Serialize to JSON
        node_json = original_node.to_json()
        self.assertIsInstance(node_json, str)
        
        # Load from JSON into a Python dictionary for inspection
        node_dict = json.loads(node_json)
        self.assertEqual(node_dict["name"], "Serialization Test")
        self.assertEqual(node_dict["node_type"], "TestType")
        
        # Ensure metadata fields are present
        self.assertIn("metadata", node_dict)
        self.assertIn("type", node_dict["metadata"])
        self.assertIn("index_fields", node_dict["metadata"])
        
        # Use model_validate_json instead of from_json directly
        restored_node = GraphNode.model_validate_json(node_json)
        
        self.assertIsInstance(restored_node, GraphNode)
        self.assertEqual(restored_node.id, original_node.id)
        self.assertEqual(restored_node.name, original_node.name)
        self.assertEqual(restored_node.node_type, original_node.node_type)
        self.assertEqual(restored_node.description, original_node.description)
        self.assertEqual(restored_node.properties, original_node.properties)


class TestGraphEdge(unittest.TestCase):
    """Tests for the GraphEdge class."""
    
    def test_creation(self):
        """Test creating a GraphEdge."""
        source_id = uuid4()
        target_id = uuid4()
        
        edge = GraphEdge.create(
            source_node_id=source_id,
            target_node_id=target_id,
            relationship_name="TEST_RELATIONSHIP"
        )
        
        self.assertIsInstance(edge, GraphEdge)
        self.assertIsInstance(edge.id, UUID)
        self.assertEqual(edge.source_node_id, source_id)
        self.assertEqual(edge.target_node_id, target_id)
        self.assertEqual(edge.relationship_name, "TEST_RELATIONSHIP")
        self.assertEqual(edge.properties, {})
        self.assertIsNone(edge.layer_id)
        
        # Check metadata
        self.assertIn("type", edge.metadata)
        self.assertEqual(edge.metadata["type"], "GraphEdge")
        self.assertIn("index_fields", edge.metadata)
    
    def test_string_id_conversion(self):
        """Test conversion of string IDs to UUIDs."""
        source_id = uuid4()
        target_id = uuid4()
        
        edge = GraphEdge.create(
            source_node_id=str(source_id),
            target_node_id=str(target_id),
            relationship_name="STRING_IDS"
        )
        
        self.assertIsInstance(edge.source_node_id, UUID)
        self.assertIsInstance(edge.target_node_id, UUID)
        self.assertEqual(edge.source_node_id, source_id)
        self.assertEqual(edge.target_node_id, target_id)
    
    def test_conversion_from_basic_edge(self):
        """Test converting a basic Edge to a GraphEdge."""
        # Create a basic Edge with valid UUID format for IDs
        source_id = str(uuid4())
        target_id = str(uuid4())
        
        basic_edge = Edge(
            id=str(uuid4()),
            source_node_id=source_id,
            target_node_id=target_id,
            relationship_name="TEST_CONVERSION"
        )
        
        edge = GraphEdge.from_basic_edge(basic_edge)
        
        self.assertIsInstance(edge, GraphEdge)
        self.assertEqual(str(edge.source_node_id), source_id)
        self.assertEqual(str(edge.target_node_id), target_id)
        self.assertEqual(edge.relationship_name, "TEST_CONVERSION")
        self.assertEqual(edge.properties, {})
        self.assertIsNone(edge.layer_id)


class TestGraphLayer(unittest.TestCase):
    """Tests for the GraphLayer class."""
    
    def test_creation(self):
        """Test creating a GraphLayer."""
        layer = GraphLayer.create(
            name="Test Layer",
            description="A test layer",
            layer_type="test"
        )
        
        self.assertIsInstance(layer, GraphLayer)
        self.assertIsInstance(layer.id, UUID)
        self.assertEqual(layer.name, "Test Layer")
        self.assertEqual(layer.description, "A test layer")
        self.assertEqual(layer.layer_type, "test")
        self.assertEqual(layer.parent_layers, [])
        self.assertEqual(layer.properties, {})
        
        # Check metadata
        self.assertIn("type", layer.metadata)
        self.assertEqual(layer.metadata["type"], "GraphLayer")
        self.assertIn("index_fields", layer.metadata)
    
    def test_parent_layers(self):
        """Test handling parent layers."""
        parent1 = GraphLayer.create(
            name="Parent 1",
            description="First parent layer",
            layer_type="parent"
        )
        
        parent2 = GraphLayer.create(
            name="Parent 2",
            description="Second parent layer",
            layer_type="parent"
        )
        
        child = GraphLayer.create(
            name="Child Layer",
            description="A child layer",
            layer_type="child",
            parent_layers=[parent1.id, parent2.id]
        )
        
        self.assertEqual(len(child.parent_layers), 2)
        self.assertIn(parent1.id, child.parent_layers)
        self.assertIn(parent2.id, child.parent_layers)
    
    def test_conversion_from_basic_layer(self):
        """Test converting a basic Layer to a GraphLayer."""
        # Create a basic Layer with valid UUID format for ID
        layer_id = str(uuid4())
        parent1_id = str(uuid4())
        parent2_id = str(uuid4())
        
        basic_layer = Layer(
            id=layer_id,
            name="Basic Layer",
            description="A basic layer for testing",
            layer_type="test",
            parent_layers=[parent1_id, parent2_id]
        )
        
        layer = GraphLayer.from_basic_layer(basic_layer)
        
        self.assertIsInstance(layer, GraphLayer)
        self.assertEqual(str(layer.id), layer_id)
        self.assertEqual(layer.name, "Basic Layer")
        self.assertEqual(layer.description, "A basic layer for testing")
        self.assertEqual(layer.layer_type, "test")
        self.assertEqual(len(layer.parent_layers), 2)
        self.assertEqual(str(layer.parent_layers[0]), parent1_id)
        self.assertEqual(str(layer.parent_layers[1]), parent2_id)


class TestLayeredKnowledgeGraphDP(unittest.TestCase):
    """Tests for the LayeredKnowledgeGraphDP class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a graph
        self.graph = LayeredKnowledgeGraphDP.create_empty(
            name="Test Graph",
            description="A test graph"
        )
        
        # Create layers
        self.base_layer = GraphLayer.create(
            name="Base Layer",
            description="Base layer for testing",
            layer_type="base"
        )
        
        self.enrichment_layer = GraphLayer.create(
            name="Enrichment Layer",
            description="Enrichment layer for testing",
            layer_type="enrichment",
            parent_layers=[self.base_layer.id]
        )
        
        # Add layers to the graph
        self.graph.add_layer(self.base_layer)
        self.graph.add_layer(self.enrichment_layer)
        
        # Create and add nodes to base layer
        self.node1 = GraphNode.create(
            name="Node 1",
            node_type="TestType",
            description="First test node"
        )
        
        self.node2 = GraphNode.create(
            name="Node 2",
            node_type="TestType",
            description="Second test node"
        )
        
        self.graph.add_node(self.node1, self.base_layer.id)
        self.graph.add_node(self.node2, self.base_layer.id)
        
        # Create and add node to enrichment layer
        self.node3 = GraphNode.create(
            name="Node 3",
            node_type="EnrichedType",
            description="An enriched node"
        )
        
        self.graph.add_node(self.node3, self.enrichment_layer.id)
        
        # Create and add edges
        self.edge1 = GraphEdge.create(
            source_node_id=self.node1.id,
            target_node_id=self.node2.id,
            relationship_name="RELATED_TO"
        )
        
        self.edge2 = GraphEdge.create(
            source_node_id=self.node3.id,
            target_node_id=self.node1.id,
            relationship_name="ENRICHES"
        )
        
        self.graph.add_edge(self.edge1, self.base_layer.id)
        self.graph.add_edge(self.edge2, self.enrichment_layer.id)
    
    def test_creation(self):
        """Test creating a LayeredKnowledgeGraphDP."""
        graph = LayeredKnowledgeGraphDP.create_empty(
            name="Empty Graph",
            description="An empty graph for testing"
        )
        
        self.assertIsInstance(graph, LayeredKnowledgeGraphDP)
        self.assertIsInstance(graph.id, UUID)
        self.assertEqual(graph.name, "Empty Graph")
        self.assertEqual(graph.description, "An empty graph for testing")
        self.assertEqual(graph.layers, {})  # Now a dictionary, not a list
        self.assertEqual(graph.nodes, {})
        self.assertEqual(graph.edges, {})
    
    def test_layer_management(self):
        """Test layer management in the graph."""
        # Check layers
        self.assertEqual(len(self.graph.layers), 2)
        self.assertIn(self.base_layer.id, self.graph.layers)
        self.assertIn(self.enrichment_layer.id, self.graph.layers)
    
    def test_node_assignment(self):
        """Test node assignment to layers."""
        # Check node assignment
        base_nodes = self.graph.get_layer_nodes(self.base_layer.id)
        self.assertEqual(len(base_nodes), 2)
        
        enrichment_nodes = self.graph.get_layer_nodes(self.enrichment_layer.id)
        self.assertEqual(len(enrichment_nodes), 1)
        
        # Check node layer map
        self.assertEqual(self.graph.node_layer_map[self.node1.id], self.base_layer.id)
        self.assertEqual(self.graph.node_layer_map[self.node2.id], self.base_layer.id)
        self.assertEqual(self.graph.node_layer_map[self.node3.id], self.enrichment_layer.id)
    
    def test_edge_assignment(self):
        """Test edge assignment to layers."""
        # Check edge assignment
        base_edges = self.graph.get_layer_edges(self.base_layer.id)
        self.assertEqual(len(base_edges), 1)
        
        enrichment_edges = self.graph.get_layer_edges(self.enrichment_layer.id)
        self.assertEqual(len(enrichment_edges), 1)
        
        # Check edge layer map
        self.assertEqual(self.graph.edge_layer_map[self.edge1.id], self.base_layer.id)
        self.assertEqual(self.graph.edge_layer_map[self.edge2.id], self.enrichment_layer.id)
    
    def test_get_layer_graph(self):
        """Test getting a layer graph."""
        # Get base layer graph
        base_graph = self.graph.get_layer_graph(self.base_layer.id)
        
        self.assertIsInstance(base_graph, KnowledgeGraph)
        self.assertEqual(len(base_graph.nodes), 2)
        self.assertEqual(len(base_graph.edges), 1)
        
        # Get enrichment layer graph
        enrichment_graph = self.graph.get_layer_graph(self.enrichment_layer.id)
        
        self.assertIsInstance(enrichment_graph, KnowledgeGraph)
        self.assertEqual(len(enrichment_graph.nodes), 1)
        self.assertEqual(len(enrichment_graph.edges), 1)
    
    def test_get_cumulative_layer_graph(self):
        """Test getting a cumulative layer graph."""
        # Get cumulative graph for enrichment layer
        cumulative_graph = self.graph.get_cumulative_layer_graph(self.enrichment_layer.id)
        
        self.assertIsInstance(cumulative_graph, KnowledgeGraph)
        self.assertEqual(len(cumulative_graph.nodes), 3)  # All nodes
        self.assertEqual(len(cumulative_graph.edges), 2)  # All edges
    
    def test_serialization(self):
        """Test serialization and deserialization of the graph."""
        # Serialize to dictionary
        serialized = self.graph.to_dict()
        
        self.assertIsInstance(serialized, dict)
        self.assertEqual(serialized["name"], "Test Graph")
        self.assertEqual(serialized["description"], "A test graph")
        self.assertEqual(len(serialized["layers"]), 2)
        self.assertEqual(len(serialized["nodes"]), 3)
        self.assertEqual(len(serialized["edges"]), 2)
        
        # Serialize to JSON
        json_str = self.graph.to_json()
        self.assertIsInstance(json_str, str)
        
        # Deserialize from dictionary
        deserialized = LayeredKnowledgeGraphDP.from_dict(serialized)
        
        self.assertIsInstance(deserialized, LayeredKnowledgeGraphDP)
        self.assertEqual(deserialized.id, self.graph.id)
        self.assertEqual(deserialized.name, self.graph.name)
        self.assertEqual(deserialized.description, self.graph.description)
        self.assertEqual(len(deserialized.layers), 2)
        self.assertEqual(len(deserialized.nodes), 3)
        self.assertEqual(len(deserialized.edges), 2)


class TestIntegrationWithExisting(unittest.TestCase):
    """Tests for integration with existing KnowledgeGraph."""
    
    def test_conversion_from_basic(self):
        """Test conversion from basic KnowledgeGraph to LayeredKnowledgeGraphDP."""
        # Create a basic KnowledgeGraph
        node1 = Node(
            id=str(uuid4()),
            name="Basic Node 1",
            type="TestType",
            description="First basic node"
        )
        
        node2 = Node(
            id=str(uuid4()),
            name="Basic Node 2",
            type="TestType",
            description="Second basic node"
        )
        
        edge = Edge(
            id=str(uuid4()),
            source_node_id=node1.id,
            target_node_id=node2.id,
            relationship_name="BASIC_RELATED"
        )
        
        basic_graph = KnowledgeGraph(
            name="Basic Graph",
            description="A basic graph for testing",
            nodes=[node1, node2],
            edges=[edge]
        )
        
        # Convert to LayeredKnowledgeGraphDP
        layered_graph = LayeredKnowledgeGraphDP.from_basic_graph(basic_graph)
        
        self.assertIsInstance(layered_graph, LayeredKnowledgeGraphDP)
        self.assertEqual(layered_graph.name, "Basic Graph")
        self.assertEqual(layered_graph.description, "A basic graph for testing")
        self.assertEqual(len(layered_graph.layers), 1)  # Base layer
        self.assertEqual(len(layered_graph.nodes), 2)
        self.assertEqual(len(layered_graph.edges), 1)
        
        # Check layer
        layer_id = next(iter(layered_graph.layers.keys()))
        layer = layered_graph.get_layer(layer_id)
        self.assertEqual(layer.name, "Base Layer")
        
        # Check nodes
        for node in layered_graph.nodes.values():
            self.assertIsInstance(node, GraphNode)
            self.assertEqual(node.layer_id, layer_id)
        
        # Check edge
        edge = next(iter(layered_graph.edges.values()))
        self.assertIsInstance(edge, GraphEdge)
        self.assertEqual(edge.layer_id, layer_id)


if __name__ == "__main__":
    unittest.main() 