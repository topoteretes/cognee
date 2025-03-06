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
    """Test cases for the GraphNode class."""
    
    def test_creation(self):
        """Test creating a GraphNode."""
        node = GraphNode(
            name="Test Node",
            node_type="TestType",
            description="Test description"
        )
        
        self.assertIsInstance(node.id, UUID)
        self.assertEqual(node.name, "Test Node")
        self.assertEqual(node.node_type, "TestType")
        self.assertEqual(node.description, "Test description")
        self.assertIsNone(node.layer_id)
        self.assertEqual(node.type, "GraphNode")
        self.assertEqual(node.metadata["type"], "GraphNode")
        self.assertEqual(node.metadata["index_fields"], ["name", "node_type", "layer_id"])
    
    def test_conversion_from_basic_node(self):
        """Test converting a basic Node to a GraphNode."""
        basic_node = Node(
            id="test-id",
            name="Basic Node",
            type="BasicType",
            description="Basic description",
            layer_id="layer-123"
        )
        
        node = GraphNode.from_basic_node(basic_node)
        
        self.assertEqual(str(node.id), "test-id")
        self.assertEqual(node.name, "Basic Node")
        self.assertEqual(node.node_type, "BasicType")
        self.assertEqual(node.description, "Basic description")
        self.assertEqual(str(node.layer_id), "layer-123")
    
    def test_serialization(self):
        """Test serialization and deserialization of a GraphNode."""
        node = GraphNode(
            name="Serialization Test",
            node_type="TestType",
            description="Test serialization"
        )
        
        # Serialize to JSON
        node_json = node.to_json()
        self.assertIsInstance(node_json, str)
        
        # Deserialize from JSON
        restored_node = GraphNode.from_json(node_json)
        self.assertEqual(restored_node.id, node.id)
        self.assertEqual(restored_node.name, node.name)
        self.assertEqual(restored_node.node_type, node.node_type)
        self.assertEqual(restored_node.description, node.description)


class TestGraphEdge(unittest.TestCase):
    """Test cases for the GraphEdge class."""
    
    def test_creation(self):
        """Test creating a GraphEdge."""
        source_id = uuid4()
        target_id = uuid4()
        
        edge = GraphEdge(
            source_node_id=source_id,
            target_node_id=target_id,
            relationship_name="TEST_RELATION"
        )
        
        self.assertIsInstance(edge.id, UUID)
        self.assertEqual(edge.source_node_id, source_id)
        self.assertEqual(edge.target_node_id, target_id)
        self.assertEqual(edge.relationship_name, "TEST_RELATION")
        self.assertIsNone(edge.layer_id)
        self.assertEqual(edge.type, "GraphEdge")
        self.assertEqual(edge.metadata["type"], "GraphEdge")
        self.assertEqual(edge.metadata["index_fields"], ["relationship_name", "layer_id"])
    
    def test_string_id_conversion(self):
        """Test creating a GraphEdge with string IDs."""
        edge = GraphEdge(
            source_node_id="00000000-0000-0000-0000-000000000001",
            target_node_id="00000000-0000-0000-0000-000000000002",
            relationship_name="STRING_ID_TEST"
        )
        
        self.assertIsInstance(edge.source_node_id, UUID)
        self.assertIsInstance(edge.target_node_id, UUID)
        self.assertEqual(str(edge.source_node_id), "00000000-0000-0000-0000-000000000001")
        self.assertEqual(str(edge.target_node_id), "00000000-0000-0000-0000-000000000002")
    
    def test_conversion_from_basic_edge(self):
        """Test converting a basic Edge to a GraphEdge."""
        basic_edge = Edge(
            source_node_id="source-123",
            target_node_id="target-456",
            relationship_name="BASIC_RELATION",
            layer_id="layer-789"
        )
        
        edge = GraphEdge.from_basic_edge(basic_edge)
        
        self.assertEqual(str(edge.source_node_id), "source-123")
        self.assertEqual(str(edge.target_node_id), "target-456")
        self.assertEqual(edge.relationship_name, "BASIC_RELATION")
        self.assertEqual(str(edge.layer_id), "layer-789")


class TestGraphLayer(unittest.TestCase):
    """Test cases for the GraphLayer class."""
    
    def test_creation(self):
        """Test creating a GraphLayer."""
        layer = GraphLayer(
            name="Test Layer",
            description="Test layer description",
            layer_type="test"
        )
        
        self.assertIsInstance(layer.id, UUID)
        self.assertEqual(layer.name, "Test Layer")
        self.assertEqual(layer.description, "Test layer description")
        self.assertEqual(layer.layer_type, "test")
        self.assertEqual(layer.parent_layers, [])
        self.assertEqual(layer.type, "GraphLayer")
        self.assertEqual(layer.metadata["type"], "GraphLayer")
        self.assertEqual(layer.metadata["index_fields"], ["name", "layer_type"])
    
    def test_parent_layers(self):
        """Test handling parent layers in a GraphLayer."""
        parent1_id = uuid4()
        parent2_id = uuid4()
        
        layer = GraphLayer(
            name="Child Layer",
            description="Child layer with parents",
            layer_type="child",
            parent_layers=[parent1_id, str(parent2_id)]
        )
        
        self.assertEqual(len(layer.parent_layers), 2)
        self.assertEqual(layer.parent_layers[0], parent1_id)
        self.assertEqual(layer.parent_layers[1], parent2_id)
    
    def test_conversion_from_basic_layer(self):
        """Test converting a basic Layer to a GraphLayer."""
        basic_layer = Layer(
            id="layer-123",
            name="Basic Layer",
            description="Basic layer description",
            layer_type="basic",
            parent_layers=["parent-456", "parent-789"]
        )
        
        layer = GraphLayer.from_basic_layer(basic_layer)
        
        self.assertEqual(str(layer.id), "layer-123")
        self.assertEqual(layer.name, "Basic Layer")
        self.assertEqual(layer.description, "Basic layer description")
        self.assertEqual(layer.layer_type, "basic")
        self.assertEqual(len(layer.parent_layers), 2)
        self.assertEqual(str(layer.parent_layers[0]), "parent-456")
        self.assertEqual(str(layer.parent_layers[1]), "parent-789")


class TestLayeredKnowledgeGraphDP(unittest.TestCase):
    """Test cases for the LayeredKnowledgeGraphDP class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a graph for testing
        self.graph = LayeredKnowledgeGraphDP(
            name="Test Graph",
            description="Test layered graph"
        )
        
        # Create base layer
        self.base_layer = GraphLayer(
            name="Base Layer",
            description="Base layer for testing",
            layer_type="base"
        )
        self.graph.add_layer(self.base_layer)
        
        # Create enrichment layer
        self.enrich_layer = GraphLayer(
            name="Enrichment Layer",
            description="Enrichment layer for testing",
            layer_type="enrichment",
            parent_layers=[self.base_layer.id]
        )
        self.graph.add_layer(self.enrich_layer)
        
        # Add nodes to base layer
        self.node1 = GraphNode(
            name="Node 1",
            node_type="TestType",
            description="Test node 1"
        )
        self.node2 = GraphNode(
            name="Node 2",
            node_type="TestType",
            description="Test node 2"
        )
        self.graph.add_node_to_layer(self.node1, self.base_layer.id)
        self.graph.add_node_to_layer(self.node2, self.base_layer.id)
        
        # Add edge to base layer
        self.edge1 = GraphEdge(
            source_node_id=self.node1.id,
            target_node_id=self.node2.id,
            relationship_name="TEST_RELATION"
        )
        self.graph.add_edge_to_layer(self.edge1, self.base_layer.id)
        
        # Add node to enrichment layer
        self.node3 = GraphNode(
            name="Node 3",
            node_type="EnrichType",
            description="Enrichment node"
        )
        self.graph.add_node_to_layer(self.node3, self.enrich_layer.id)
        
        # Add edges to enrichment layer
        self.edge2 = GraphEdge(
            source_node_id=self.node3.id,
            target_node_id=self.node1.id,
            relationship_name="ENRICH_RELATION"
        )
        self.graph.add_edge_to_layer(self.edge2, self.enrich_layer.id)
    
    def test_creation(self):
        """Test creating a LayeredKnowledgeGraphDP."""
        graph = LayeredKnowledgeGraphDP(
            name="Empty Graph",
            description="Empty graph for testing"
        )
        
        self.assertIsInstance(graph.id, UUID)
        self.assertEqual(graph.name, "Empty Graph")
        self.assertEqual(graph.description, "Empty graph for testing")
        self.assertEqual(graph.layers, [])
        self.assertEqual(graph.type, "LayeredKnowledgeGraphDP")
    
    def test_layer_management(self):
        """Test layer management in the graph."""
        self.assertEqual(len(self.graph.layers), 2)
        self.assertEqual(self.graph.layers[0], self.base_layer.id)
        self.assertEqual(self.graph.layers[1], self.enrich_layer.id)
    
    def test_node_assignment(self):
        """Test node assignment to layers."""
        base_nodes = self.graph.get_layer_nodes(self.base_layer.id)
        enrich_nodes = self.graph.get_layer_nodes(self.enrich_layer.id)
        
        self.assertEqual(len(base_nodes), 2)
        self.assertEqual(len(enrich_nodes), 1)
        
        self.assertEqual(base_nodes[0].name, "Node 1")
        self.assertEqual(base_nodes[1].name, "Node 2")
        self.assertEqual(enrich_nodes[0].name, "Node 3")
        
        self.assertEqual(base_nodes[0].layer_id, self.base_layer.id)
        self.assertEqual(enrich_nodes[0].layer_id, self.enrich_layer.id)
    
    def test_edge_assignment(self):
        """Test edge assignment to layers."""
        base_edges = self.graph.get_layer_edges(self.base_layer.id)
        enrich_edges = self.graph.get_layer_edges(self.enrich_layer.id)
        
        self.assertEqual(len(base_edges), 1)
        self.assertEqual(len(enrich_edges), 1)
        
        self.assertEqual(base_edges[0].relationship_name, "TEST_RELATION")
        self.assertEqual(enrich_edges[0].relationship_name, "ENRICH_RELATION")
        
        self.assertEqual(base_edges[0].layer_id, self.base_layer.id)
        self.assertEqual(enrich_edges[0].layer_id, self.enrich_layer.id)
    
    def test_get_layer_graph(self):
        """Test retrieving a layer-specific graph."""
        base_graph = self.graph.get_layer_graph(self.base_layer.id)
        enrich_graph = self.graph.get_layer_graph(self.enrich_layer.id)
        
        self.assertIsInstance(base_graph, KnowledgeGraph)
        self.assertIsInstance(enrich_graph, KnowledgeGraph)
        
        self.assertEqual(len(base_graph.nodes), 2)
        self.assertEqual(len(base_graph.edges), 1)
        self.assertEqual(len(enrich_graph.nodes), 1)
        self.assertEqual(len(enrich_graph.edges), 1)
    
    def test_get_cumulative_layer_graph(self):
        """Test retrieving a cumulative layer graph."""
        cumulative_graph = self.graph.get_cumulative_layer_graph(self.enrich_layer.id)
        
        self.assertIsInstance(cumulative_graph, KnowledgeGraph)
        self.assertEqual(len(cumulative_graph.nodes), 3)
        self.assertEqual(len(cumulative_graph.edges), 2)
    
    def test_serialization(self):
        """Test serialization and deserialization of the graph."""
        serialized = self.graph.to_serializable_dict()
        
        # Check the serialized data
        self.assertEqual(serialized["name"], "Test Graph")
        self.assertEqual(len(serialized["layers"]), 2)
        self.assertEqual(len(serialized["layer_data"]), 2)
        self.assertEqual(len(serialized["node_data"]), 3)
        self.assertEqual(len(serialized["edge_data"]), 2)
        
        # Convert to JSON and back to test full serialization
        json_str = json.dumps(serialized)
        deserialized = json.loads(json_str)
        
        # Deserialize to a graph
        restored_graph = LayeredKnowledgeGraphDP.from_serializable_dict(deserialized)
        
        # Test the restored graph
        self.assertEqual(restored_graph.name, "Test Graph")
        self.assertEqual(len(restored_graph.layers), 2)
        
        # Check that we can retrieve the layers
        base_graph = restored_graph.get_layer_graph(self.base_layer.id)
        enrich_graph = restored_graph.get_layer_graph(self.enrich_layer.id)
        
        self.assertEqual(len(base_graph.nodes), 2)
        self.assertEqual(len(enrich_graph.nodes), 1)
        self.assertEqual(len(base_graph.edges), 1)
        self.assertEqual(len(enrich_graph.edges), 1)
        
        # Check cumulative graph
        cumulative_graph = restored_graph.get_cumulative_layer_graph(self.enrich_layer.id)
        self.assertEqual(len(cumulative_graph.nodes), 3)
        self.assertEqual(len(cumulative_graph.edges), 2)


class TestIntegrationWithExisting(unittest.TestCase):
    """Test integration with existing KnowledgeGraph implementations."""
    
    def test_conversion_from_basic(self):
        """Test conversion from basic KnowledgeGraph to LayeredKnowledgeGraphDP."""
        # Create basic nodes and edges
        nodes = [
            Node(id="node1", name="Basic Node 1", type="BasicType", description="Basic node 1"),
            Node(id="node2", name="Basic Node 2", type="BasicType", description="Basic node 2")
        ]
        
        edges = [
            Edge(source_node_id="node1", target_node_id="node2", relationship_name="BASIC_RELATION")
        ]
        
        # Create basic graph
        basic_graph = KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            name="Basic Graph",
            description="Basic knowledge graph"
        )
        
        # Create layered graph
        layered_graph = LayeredKnowledgeGraphDP.create_empty(
            name="Converted Graph",
            description="Converted from basic graph"
        )
        
        # Create and add a layer
        layer = GraphLayer(
            id=uuid4(),
            name="Basic Layer",
            description="Converted basic layer",
            layer_type="base"
        )
        layered_graph.add_layer(layer)
        
        # Convert and add nodes
        for basic_node in basic_graph.nodes:
            node = GraphNode.from_basic_node(basic_node, layer.id)
            layered_graph.add_node_to_layer(node, layer.id)
        
        # Convert and add edges
        for basic_edge in basic_graph.edges:
            # Need to map the string IDs to UUID objects
            node_map = {
                node.name: node.id 
                for node in layered_graph._node_cache.values()
            }
            
            source_name = next(node.name for node in basic_graph.nodes if node.id == basic_edge.source_node_id)
            target_name = next(node.name for node in basic_graph.nodes if node.id == basic_edge.target_node_id)
            
            edge = GraphEdge(
                source_node_id=node_map[source_name],
                target_node_id=node_map[target_name],
                relationship_name=basic_edge.relationship_name,
                layer_id=layer.id
            )
            layered_graph.add_edge_to_layer(edge, layer.id)
        
        # Test the resulting graph
        graph = layered_graph.get_layer_graph(layer.id)
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)
        
        # Test node names preserved
        node_names = {node.name for node in graph.nodes}
        self.assertEqual(node_names, {"Basic Node 1", "Basic Node 2"})
        
        # Test relationships preserved
        self.assertEqual(graph.edges[0].relationship_name, "BASIC_RELATION")


if __name__ == "__main__":
    unittest.main() 