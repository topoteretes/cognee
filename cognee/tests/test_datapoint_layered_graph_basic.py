import unittest
import asyncio
from uuid import uuid4, UUID

from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode,
    GraphEdge,
    GraphLayer,
    LayeredKnowledgeGraphDP
)


class TestDataPointLayeredGraphBasic(unittest.TestCase):
    """Basic tests for the DataPoint-based layered graph implementation."""

    def setUp(self):
        """Set up test resources."""
        self.graph = LayeredKnowledgeGraphDP.create_empty(
            name="Test Graph",
            description="Test graph for unit tests"
        )
        
        # Create basic layers
        self.base_layer = GraphLayer(
            id=uuid4(),
            name="Base Layer",
            description="Base layer containing foundational nodes",
            layer_type="base"
        )
        
        self.detail_layer = GraphLayer(
            id=uuid4(),
            name="Detail Layer",
            description="Detail layer containing additional information",
            layer_type="detail",
            parent_layers=[self.base_layer.id]
        )
        
        # Add layers to graph
        self.graph.add_layer(self.base_layer)
        self.graph.add_layer(self.detail_layer)
        
        # Create basic nodes
        self.company_node = GraphNode(
            id=uuid4(),
            name="ACME Corp",
            node_type="Company",
            description="A fictional company"
        )
        
        self.dept_node = GraphNode(
            id=uuid4(),
            name="Engineering",
            node_type="Department",
            description="Engineering department"
        )
        
        self.person_node = GraphNode(
            id=uuid4(),
            name="John Doe",
            node_type="Person",
            description="A person",
            properties={"title": "Software Engineer", "level": 3}
        )
        
        # Add nodes to layers
        self.graph.add_node_to_layer(self.company_node, self.base_layer.id)
        self.graph.add_node_to_layer(self.dept_node, self.base_layer.id)
        self.graph.add_node_to_layer(self.person_node, self.detail_layer.id)
        
        # Create edges
        self.dept_edge = GraphEdge(
            source_node_id=self.dept_node.id,
            target_node_id=self.company_node.id,
            relationship_name="PART_OF"
        )
        
        self.person_edge = GraphEdge(
            source_node_id=self.person_node.id,
            target_node_id=self.dept_node.id,
            relationship_name="WORKS_IN"
        )
        
        # Add edges to layers
        self.graph.add_edge_to_layer(self.dept_edge, self.base_layer.id)
        self.graph.add_edge_to_layer(self.person_edge, self.detail_layer.id)

    def test_layer_management(self):
        """Test that layers are correctly managed."""
        # Check layer counts
        self.assertEqual(len(self.graph.layers), 2)
        
        # Verify layers are retrievable
        self.assertIn(self.base_layer.id, self.graph.layers)
        self.assertIn(self.detail_layer.id, self.graph.layers)
        
        # Verify parent layer relationships
        detail_layer = self.graph._get_layer(self.detail_layer.id)
        self.assertIn(self.base_layer.id, detail_layer.parent_layers)
        
        # Test get_parent_layers method
        parent_layers = self.graph.get_parent_layers(self.detail_layer.id)
        self.assertEqual(len(parent_layers), 1)
        self.assertEqual(parent_layers[0].id, self.base_layer.id)

    def test_node_management(self):
        """Test that nodes are correctly assigned to layers."""
        # Check node counts per layer
        base_nodes = self.graph.get_layer_nodes(self.base_layer.id)
        detail_nodes = self.graph.get_layer_nodes(self.detail_layer.id)
        
        self.assertEqual(len(base_nodes), 2)
        self.assertEqual(len(detail_nodes), 1)
        
        # Verify nodes are in correct layers
        base_node_ids = [node.id for node in base_nodes]
        detail_node_ids = [node.id for node in detail_nodes]
        
        self.assertIn(self.company_node.id, base_node_ids)
        self.assertIn(self.dept_node.id, base_node_ids)
        self.assertIn(self.person_node.id, detail_node_ids)
        
        # Verify node properties
        company = next(node for node in base_nodes if node.id == self.company_node.id)
        person = detail_nodes[0]
        
        self.assertEqual(company.name, "ACME Corp")
        self.assertEqual(company.node_type, "Company")
        
        self.assertEqual(person.name, "John Doe")
        self.assertEqual(person.properties.get("title"), "Software Engineer")
        self.assertEqual(person.properties.get("level"), 3)

    def test_edge_management(self):
        """Test that edges are correctly assigned to layers."""
        # Check edge counts per layer
        base_edges = self.graph.get_layer_edges(self.base_layer.id)
        detail_edges = self.graph.get_layer_edges(self.detail_layer.id)
        
        self.assertEqual(len(base_edges), 1)
        self.assertEqual(len(detail_edges), 1)
        
        # Verify edges are in correct layers
        base_edge = base_edges[0]
        detail_edge = detail_edges[0]
        
        self.assertEqual(base_edge.source_node_id, self.dept_node.id)
        self.assertEqual(base_edge.target_node_id, self.company_node.id)
        self.assertEqual(base_edge.relationship_name, "PART_OF")
        
        self.assertEqual(detail_edge.source_node_id, self.person_node.id)
        self.assertEqual(detail_edge.target_node_id, self.dept_node.id)
        self.assertEqual(detail_edge.relationship_name, "WORKS_IN")

    def test_layer_graph_extraction(self):
        """Test extracting individual layer graphs."""
        # Get base layer graph
        base_graph = self.graph.get_layer_graph(self.base_layer.id)
        
        self.assertEqual(len(base_graph.nodes), 2)
        self.assertEqual(len(base_graph.edges), 1)
        
        # Get detail layer graph
        detail_graph = self.graph.get_layer_graph(self.detail_layer.id)
        
        self.assertEqual(len(detail_graph.nodes), 1)
        self.assertEqual(len(detail_graph.edges), 1)

    def test_cumulative_graph_extraction(self):
        """Test extracting cumulative layer graphs."""
        # Get cumulative graph for base layer (should be same as base layer)
        base_cumulative = self.graph.get_cumulative_layer_graph(self.base_layer.id)
        
        self.assertEqual(len(base_cumulative.nodes), 2)
        self.assertEqual(len(base_cumulative.edges), 1)
        
        # Get cumulative graph for detail layer (should include both layers)
        detail_cumulative = self.graph.get_cumulative_layer_graph(self.detail_layer.id)
        
        self.assertEqual(len(detail_cumulative.nodes), 3)
        self.assertEqual(len(detail_cumulative.edges), 2)
        
        # Verify all nodes are in the cumulative graph
        all_node_ids = [node.id for node in detail_cumulative.nodes]
        self.assertIn(self.company_node.id, all_node_ids)
        self.assertIn(self.dept_node.id, all_node_ids)
        self.assertIn(self.person_node.id, all_node_ids)

    def test_serialization(self):
        """Test serialization and deserialization."""
        # Serialize the graph
        serialized = self.graph.to_serializable_dict()
        
        # Verify serialized contains required keys
        self.assertIn("metadata", serialized)
        self.assertIn("layers", serialized)
        self.assertIn("nodes", serialized)
        self.assertIn("edges", serialized)
        
        # Deserialize to new graph
        new_graph = LayeredKnowledgeGraphDP.from_serializable_dict(serialized)
        
        # Verify layer count matches
        self.assertEqual(len(new_graph.layers), len(self.graph.layers))
        
        # Verify node and edge counts
        original_node_count = sum(len(self.graph.get_layer_nodes(layer_id)) 
                                  for layer_id in self.graph.layers)
        new_node_count = sum(len(new_graph.get_layer_nodes(layer_id)) 
                             for layer_id in new_graph.layers)
        
        original_edge_count = sum(len(self.graph.get_layer_edges(layer_id)) 
                                  for layer_id in self.graph.layers)
        new_edge_count = sum(len(new_graph.get_layer_edges(layer_id)) 
                             for layer_id in new_graph.layers)
        
        self.assertEqual(new_node_count, original_node_count)
        self.assertEqual(new_edge_count, original_edge_count)
        
        # Verify cumulative graph extraction still works
        detail_layer_id = [lid for lid in new_graph.layers 
                           if new_graph._get_layer(lid).name == "Detail Layer"][0]
        cumulative = new_graph.get_cumulative_layer_graph(detail_layer_id)
        self.assertEqual(len(cumulative.nodes), 3)
        self.assertEqual(len(cumulative.edges), 2)


class TestAsyncFunctionality(unittest.IsolatedAsyncioTestCase):
    """Tests for asynchronous functionality in the layered graph implementation."""
    
    async def asyncSetUp(self):
        """Set up test resources asynchronously."""
        self.graph = LayeredKnowledgeGraphDP.create_empty(
            name="Async Test Graph",
            description="Test graph for async operations"
        )
        
        # Create and add layers
        self.base_layer = GraphLayer(
            id=uuid4(),
            name="Base Layer",
            description="Base layer with foundational data",
            layer_type="base"
        )
        
        self.enrichment_layer = GraphLayer(
            id=uuid4(),
            name="Enrichment Layer",
            description="Layer with enrichment data",
            layer_type="enrichment",
            parent_layers=[self.base_layer.id]
        )
        
        self.graph.add_layer(self.base_layer)
        self.graph.add_layer(self.enrichment_layer)
    
    async def test_async_node_addition(self):
        """Test adding nodes asynchronously."""
        # Create nodes
        nodes = []
        for i in range(10):
            node = GraphNode(
                id=uuid4(),
                name=f"Node {i}",
                node_type="TestNode",
                description=f"Test node {i}"
            )
            nodes.append(node)
        
        # Add nodes to base layer
        tasks = []
        for node in nodes[:5]:
            tasks.append(asyncio.create_task(
                self.graph.add_node_to_layer_async(node, self.base_layer.id)
            ))
        
        # Add nodes to enrichment layer
        for node in nodes[5:]:
            tasks.append(asyncio.create_task(
                self.graph.add_node_to_layer_async(node, self.enrichment_layer.id)
            ))
        
        # Wait for all tasks
        await asyncio.gather(*tasks)
        
        # Verify nodes were added correctly
        base_nodes = self.graph.get_layer_nodes(self.base_layer.id)
        enrichment_nodes = self.graph.get_layer_nodes(self.enrichment_layer.id)
        
        self.assertEqual(len(base_nodes), 5)
        self.assertEqual(len(enrichment_nodes), 5)
    
    async def test_async_edge_addition(self):
        """Test adding edges asynchronously."""
        # Create and add nodes first
        node_a = GraphNode(id=uuid4(), name="Node A", node_type="TestNode", description="Test node A")
        node_b = GraphNode(id=uuid4(), name="Node B", node_type="TestNode", description="Test node B")
        node_c = GraphNode(id=uuid4(), name="Node C", node_type="TestNode", description="Test node C")
        
        self.graph.add_node_to_layer(node_a, self.base_layer.id)
        self.graph.add_node_to_layer(node_b, self.base_layer.id)
        self.graph.add_node_to_layer(node_c, self.enrichment_layer.id)
        
        # Create edges
        edge1 = GraphEdge(source_node_id=node_a.id, target_node_id=node_b.id, relationship_name="RELATED_TO")
        edge2 = GraphEdge(source_node_id=node_c.id, target_node_id=node_a.id, relationship_name="DEPENDS_ON")
        
        # Add edges asynchronously
        tasks = [
            asyncio.create_task(self.graph.add_edge_to_layer_async(edge1, self.base_layer.id)),
            asyncio.create_task(self.graph.add_edge_to_layer_async(edge2, self.enrichment_layer.id))
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify edges were added correctly
        base_edges = self.graph.get_layer_edges(self.base_layer.id)
        enrichment_edges = self.graph.get_layer_edges(self.enrichment_layer.id)
        
        self.assertEqual(len(base_edges), 1)
        self.assertEqual(len(enrichment_edges), 1)
        
        self.assertEqual(base_edges[0].source_node_id, node_a.id)
        self.assertEqual(base_edges[0].target_node_id, node_b.id)
        
        self.assertEqual(enrichment_edges[0].source_node_id, node_c.id)
        self.assertEqual(enrichment_edges[0].target_node_id, node_a.id)
    
    async def test_async_graph_build(self):
        """Test building a graph with async operations."""
        # Node creation function
        async def create_nodes(layer_id, count, prefix):
            for i in range(count):
                node = GraphNode(
                    id=uuid4(),
                    name=f"{prefix} {i}",
                    node_type=f"{prefix}Type",
                    description=f"Test {prefix} node {i}"
                )
                await self.graph.add_node_to_layer_async(node, layer_id)
                yield node
        
        # Create nodes for base layer
        base_nodes = []
        async for node in create_nodes(self.base_layer.id, 5, "Base"):
            base_nodes.append(node)
        
        # Create nodes for enrichment layer that connect to base nodes
        async for i, enrichment_node in enumerate(create_nodes(self.enrichment_layer.id, 10, "Enrichment")):
            # Connect to a base node (cycling through the base nodes)
            base_node = base_nodes[i % len(base_nodes)]
            edge = GraphEdge(
                source_node_id=enrichment_node.id,
                target_node_id=base_node.id,
                relationship_name="ENRICHES"
            )
            await self.graph.add_edge_to_layer_async(edge, self.enrichment_layer.id)
        
        # Verify the graph structure
        base_layer_nodes = self.graph.get_layer_nodes(self.base_layer.id)
        enrichment_layer_nodes = self.graph.get_layer_nodes(self.enrichment_layer.id)
        enrichment_layer_edges = self.graph.get_layer_edges(self.enrichment_layer.id)
        
        self.assertEqual(len(base_layer_nodes), 5)
        self.assertEqual(len(enrichment_layer_nodes), 10)
        self.assertEqual(len(enrichment_layer_edges), 10)
        
        # Verify cumulative graph contains all nodes and edges
        cumulative = self.graph.get_cumulative_layer_graph(self.enrichment_layer.id)
        self.assertEqual(len(cumulative.nodes), 15)  # 5 base + 10 enrichment
        self.assertEqual(len(cumulative.edges), 10)  # 10 enrichment edges


if __name__ == "__main__":
    unittest.main() 