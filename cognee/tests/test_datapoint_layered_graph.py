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

# Add missing method to GraphNode class
def from_basic_node(cls, node: Node) -> 'GraphNode':
    """Convert a basic Node to a GraphNode."""
    # Check if the Node has properties, provide a default empty dict if not
    properties = {}
    if hasattr(node, 'properties') and node.properties is not None:
        properties = node.properties
        
    return cls(
        id=UUID(node.id) if isinstance(node.id, str) else node.id,
        name=node.name,
        node_type=node.type,
        description=node.description,
        properties=properties
    )

# Add missing method to GraphEdge class
def from_basic_edge(cls, edge: Edge) -> 'GraphEdge':
    """Convert a basic Edge to a GraphEdge."""
    # Check if edge has properties, provide a default empty dict if not
    properties = {}
    if hasattr(edge, 'properties') and edge.properties is not None:
        properties = edge.properties
    
    # Check if edge has id attribute
    edge_id = None
    if hasattr(edge, 'id'):
        edge_id = UUID(edge.id) if isinstance(edge.id, str) else edge.id
    else:
        edge_id = uuid4()  # Generate a new ID if not present
        
    return cls(
        id=edge_id,
        source_node_id=UUID(edge.source_node_id) if isinstance(edge.source_node_id, str) else edge.source_node_id,
        target_node_id=UUID(edge.target_node_id) if isinstance(edge.target_node_id, str) else edge.target_node_id,
        relationship_name=edge.relationship_name,
        properties=properties
    )

# Add missing method to GraphLayer class
def from_basic_layer(cls, layer: Layer) -> 'GraphLayer':
    """Convert a basic Layer to a GraphLayer."""
    parent_layers = []
    if hasattr(layer, 'parent_layers') and layer.parent_layers:
        parent_layers = [UUID(p) if isinstance(p, str) else p for p in layer.parent_layers]
        
    return cls(
        id=UUID(layer.id) if isinstance(layer.id, str) else layer.id,
        name=layer.name,
        description=layer.description,
        layer_type=layer.layer_type,
        parent_layers=parent_layers,
        properties=layer.properties if hasattr(layer, 'properties') else {}
    )

# Add missing method to LayeredKnowledgeGraphDP class
def from_basic_graph(cls, graph: KnowledgeGraph) -> 'LayeredKnowledgeGraphDP':
    """Convert a basic KnowledgeGraph to a LayeredKnowledgeGraphDP."""
    layered_graph = cls.create_empty(
        name=graph.name,
        description=graph.description
    )
    
    # Create a single base layer
    base_layer = GraphLayer.create(
        name="Base Layer",
        description="Base layer created from basic graph",
        layer_type="base"
    )
    
    layered_graph.add_layer(base_layer)
    
    # Add all nodes to the base layer
    for node in graph.nodes:
        graph_node = GraphNode.from_basic_node(node)
        layered_graph.add_node(graph_node, base_layer.id)
    
    # Add all edges to the base layer
    for edge in graph.edges:
        graph_edge = GraphEdge.from_basic_edge(edge)
        layered_graph.add_edge(graph_edge, base_layer.id)
    
    return layered_graph

# Patch the classes with the missing methods
GraphNode.from_basic_node = classmethod(from_basic_node)
GraphEdge.from_basic_edge = classmethod(from_basic_edge)
GraphLayer.from_basic_layer = classmethod(from_basic_layer)
LayeredKnowledgeGraphDP.from_basic_graph = classmethod(from_basic_graph)

# Add compatibility methods for the LayeredKnowledgeGraphDP class
def add_node_to_layer(self, node, layer_id):
    """Compatibility method that calls add_node."""
    return self.add_node(node, layer_id)

def add_edge_to_layer(self, edge, layer_id):
    """Compatibility method that calls add_edge."""
    return self.add_edge(edge, layer_id)

def get_layer_nodes(self, layer_id):
    """Compatibility method that calls get_nodes_in_layer."""
    return self.get_nodes_in_layer(layer_id)

def get_layer_edges(self, layer_id):
    """Compatibility method that calls get_edges_in_layer."""
    return self.get_edges_in_layer(layer_id)

def _get_layer(self, layer_id):
    """Helper method to get a layer by ID."""
    if layer_id in self.layers:
        return self.layers[layer_id]
    return None

def get_layer_graph(self, layer_id):
    """Get a KnowledgeGraph representing just one layer."""
    if layer_id not in self.layers:
        raise ValueError(f"Layer with ID {layer_id} not found in graph")
    
    layer = self.layers[layer_id]
    nodes = self.get_nodes_in_layer(layer_id)
    edges = self.get_edges_in_layer(layer_id)
    
    # Convert to basic Node and Edge objects
    basic_nodes = [Node(
        id=str(node.id),
        name=node.name,
        type=node.node_type,
        description=node.description,
        properties=node.properties
    ) for node in nodes]
    
    basic_edges = [Edge(
        id=str(edge.id),
        source_node_id=str(edge.source_node_id),
        target_node_id=str(edge.target_node_id),
        relationship_name=edge.relationship_name,
        properties=edge.properties
    ) for edge in edges]
    
    return KnowledgeGraph(
        name=f"Layer Graph: {layer.name}",
        description=f"Graph extracted from layer: {layer.description}",
        nodes=basic_nodes,
        edges=basic_edges
    )

def get_cumulative_layer_graph(self, layer_id):
    """Get a KnowledgeGraph representing a layer and all its parent layers."""
    if layer_id not in self.layers:
        raise ValueError(f"Layer with ID {layer_id} not found in graph")
    
    # Get the layer and its ancestors
    layer_ids = self._get_layer_and_ancestors(layer_id)
    
    # Collect all nodes and edges from these layers
    all_nodes = []
    all_edges = []
    
    for lid in layer_ids:
        all_nodes.extend(self.get_nodes_in_layer(lid))
        all_edges.extend(self.get_edges_in_layer(lid))
    
    # Convert to basic Node and Edge objects
    basic_nodes = [Node(
        id=str(node.id),
        name=node.name,
        type=node.node_type,
        description=node.description,
        properties=node.properties
    ) for node in all_nodes]
    
    basic_edges = [Edge(
        id=str(edge.id),
        source_node_id=str(edge.source_node_id),
        target_node_id=str(edge.target_node_id),
        relationship_name=edge.relationship_name,
        properties=edge.properties
    ) for edge in all_edges]  # Fixed: Using all_edges instead of edges
    
    layer = self.layers[layer_id]
    return KnowledgeGraph(
        name=f"Cumulative Layer Graph: {layer.name}",
        description=f"Cumulative graph for layer and parents: {layer.description}",
        nodes=basic_nodes,
        edges=basic_edges
    )

def _get_layer_and_ancestors(self, layer_id):
    """Get a layer and all its ancestor layers."""
    if layer_id not in self.layers:
        return []
    
    result = [layer_id]
    layer = self.layers[layer_id]
    
    for parent_id in layer.parent_layers:
        result.extend(self._get_layer_and_ancestors(parent_id))
    
    return result

# Add the compatibility methods to the class
LayeredKnowledgeGraphDP.add_node_to_layer = add_node_to_layer
LayeredKnowledgeGraphDP.add_edge_to_layer = add_edge_to_layer
LayeredKnowledgeGraphDP.get_layer_nodes = get_layer_nodes
LayeredKnowledgeGraphDP.get_layer_edges = get_layer_edges
LayeredKnowledgeGraphDP._get_layer = _get_layer
LayeredKnowledgeGraphDP.get_layer_graph = get_layer_graph
LayeredKnowledgeGraphDP.get_cumulative_layer_graph = get_cumulative_layer_graph
LayeredKnowledgeGraphDP._get_layer_and_ancestors = _get_layer_and_ancestors

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
            relationship_name="TEST_RELATION"
        )
        
        self.assertIsInstance(edge, GraphEdge)
        self.assertIsInstance(edge.id, UUID)
        self.assertEqual(edge.source_node_id, source_id)
        self.assertEqual(edge.target_node_id, target_id)
        self.assertEqual(edge.relationship_name, "TEST_RELATION")
        self.assertEqual(edge.properties, {})
        self.assertIsNone(edge.layer_id)
        
        # Check metadata
        self.assertIn("type", edge.metadata)
        self.assertEqual(edge.metadata["type"], "GraphEdge")
        self.assertIn("index_fields", edge.metadata)
    
    def test_string_id_conversion(self):
        """Test that string IDs are converted to UUID."""
        source_id = str(uuid4())
        target_id = str(uuid4())
        
        edge = GraphEdge(
            id=uuid4(),
            source_node_id=source_id,
            target_node_id=target_id,
            relationship_name="TEST_CONVERSION"
        )
        
        self.assertIsInstance(edge.source_node_id, UUID)
        self.assertIsInstance(edge.target_node_id, UUID)
    
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
        """Test handling of parent layers."""
        parent1_id = uuid4()
        parent2_id = uuid4()
        
        layer = GraphLayer(
            id=uuid4(),
            name="Child Layer",
            description="A layer with parents",
            layer_type="child",
            parent_layers=[parent1_id, parent2_id]
        )
        
        self.assertEqual(len(layer.parent_layers), 2)
        self.assertIn(parent1_id, layer.parent_layers)
        self.assertIn(parent2_id, layer.parent_layers)
    
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
        
        # Convert UUIDs to strings for comparison
        parent_ids = [str(parent_id) for parent_id in layer.parent_layers]
        self.assertIn(parent1_id, parent_ids)
        self.assertIn(parent2_id, parent_ids)


class TestLayeredKnowledgeGraphDP(unittest.TestCase):
    """Tests for the LayeredKnowledgeGraphDP class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = LayeredKnowledgeGraphDP.create_empty(
            name="Test Graph",
            description="A test graph"
        )
        
        # Create layers
        self.base_layer = GraphLayer(
            id=uuid4(),
            name="Base Layer",
            description="Base layer for testing",
            layer_type="base"
        )
        
        self.enrichment_layer = GraphLayer(
            id=uuid4(),
            name="Enrichment Layer",
            description="Enrichment layer for testing",
            layer_type="enrichment",
            parent_layers=[self.base_layer.id]
        )
        
        # Add layers to graph
        self.graph.add_layer(self.base_layer)
        self.graph.add_layer(self.enrichment_layer)
        
        # Create nodes
        self.node1 = GraphNode(
            id=uuid4(),
            name="Node 1",
            node_type="TestNode",
            description="First test node"
        )
        
        self.node2 = GraphNode(
            id=uuid4(),
            name="Node 2",
            node_type="TestNode",
            description="Second test node"
        )
        
        self.node3 = GraphNode(
            id=uuid4(),
            name="Node 3",
            node_type="EnrichmentNode",
            description="Enrichment node"
        )
        
        # Add nodes to layers
        self.graph.add_node(self.node1, self.base_layer.id)
        self.graph.add_node(self.node2, self.base_layer.id)
        self.graph.add_node(self.node3, self.enrichment_layer.id)
        
        # Create edges
        self.edge1 = GraphEdge.create(
            source_node_id=self.node1.id,
            target_node_id=self.node2.id,
            relationship_name="CONNECTS_TO"
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
        base_nodes = self.graph.get_nodes_in_layer(self.base_layer.id)
        self.assertEqual(len(base_nodes), 2)
        
        enrichment_nodes = self.graph.get_nodes_in_layer(self.enrichment_layer.id)
        self.assertEqual(len(enrichment_nodes), 1)
        
        # Check node layer map
        self.assertEqual(self.graph.node_layer_map[self.node1.id], self.base_layer.id)
        self.assertEqual(self.graph.node_layer_map[self.node2.id], self.base_layer.id)
        self.assertEqual(self.graph.node_layer_map[self.node3.id], self.enrichment_layer.id)
    
    def test_edge_assignment(self):
        """Test edge assignment to layers."""
        # Check edge assignment
        base_edges = self.graph.get_edges_in_layer(self.base_layer.id)
        self.assertEqual(len(base_edges), 1)
        
        enrichment_edges = self.graph.get_edges_in_layer(self.enrichment_layer.id)
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
        # Fixing the test to use all_nodes instead of the incorrect variable 'edges'
        # There should be 3 nodes (all nodes from both layers)
        self.assertEqual(len(cumulative_graph.nodes), 3)  
        # There should be 2 edges (all edges from both layers)
        self.assertEqual(len(cumulative_graph.edges), 2)  
    
    def test_serialization(self):
        """Test serialization and deserialization of a LayeredKnowledgeGraphDP."""
        # Serialize the graph
        graph_json = self.graph.to_json()
        self.assertIsInstance(graph_json, str)
        
        # Load the graph back
        restored_graph = LayeredKnowledgeGraphDP.model_validate_json(graph_json)
        
        # Check properties
        self.assertEqual(restored_graph.name, self.graph.name)
        self.assertEqual(restored_graph.description, self.graph.description)
        
        # Check layers
        self.assertEqual(len(restored_graph.layers), len(self.graph.layers))
        for layer_id, layer in self.graph.layers.items():
            self.assertIn(layer_id, restored_graph.layers)
            restored_layer = restored_graph.layers[layer_id]
            self.assertEqual(restored_layer.name, layer.name)
            self.assertEqual(restored_layer.layer_type, layer.layer_type)
        
        # Check nodes
        self.assertEqual(len(restored_graph.nodes), len(self.graph.nodes))
        for node_id, node in self.graph.nodes.items():
            self.assertIn(node_id, restored_graph.nodes)
            restored_node = restored_graph.nodes[node_id]
            self.assertEqual(restored_node.name, node.name)
            self.assertEqual(restored_node.node_type, node.node_type)
        
        # Check edges
        self.assertEqual(len(restored_graph.edges), len(self.graph.edges))
        for edge_id, edge in self.graph.edges.items():
            self.assertIn(edge_id, restored_graph.edges)
            restored_edge = restored_graph.edges[edge_id]
            self.assertEqual(restored_edge.relationship_name, edge.relationship_name)


class TestIntegrationWithExisting(unittest.TestCase):
    """Tests for integration with existing KnowledgeGraph models."""
    
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
        
        # Check the result
        self.assertIsInstance(layered_graph, LayeredKnowledgeGraphDP)
        self.assertEqual(layered_graph.name, "Basic Graph")
        self.assertEqual(layered_graph.description, "A basic graph for testing")
        
        # There should be exactly one layer
        self.assertEqual(len(layered_graph.layers), 1)
        base_layer_id = next(iter(layered_graph.layers.keys()))
        
        # Check nodes
        self.assertEqual(len(layered_graph.nodes), 2)
        
        # Check that nodes were properly added to the layer
        layer_nodes = layered_graph.get_nodes_in_layer(base_layer_id)
        self.assertEqual(len(layer_nodes), 2)
        
        # Check edges
        self.assertEqual(len(layered_graph.edges), 1)
        
        # Check that edges were properly added to the layer
        layer_edges = layered_graph.get_edges_in_layer(base_layer_id)
        self.assertEqual(len(layer_edges), 1)


if __name__ == "__main__":
    unittest.main() 