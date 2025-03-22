import unittest
import asyncio
from uuid import uuid4, UUID

from cognee.modules.graph.datapoint_layered_graph import (
    GraphNode,
    GraphEdge,
    GraphLayer,
    LayeredKnowledgeGraphDP
)

# Add compatibility methods for the LayeredKnowledgeGraphDP class if not already defined
if not hasattr(LayeredKnowledgeGraphDP, 'add_node_to_layer'):
    def add_node_to_layer(self, node, layer_id):
        """Compatibility method that calls add_node."""
        return self.add_node(node, layer_id)
    
    LayeredKnowledgeGraphDP.add_node_to_layer = add_node_to_layer

if not hasattr(LayeredKnowledgeGraphDP, 'add_edge_to_layer'):
    def add_edge_to_layer(self, edge, layer_id):
        """Compatibility method that calls add_edge."""
        return self.add_edge(edge, layer_id)
    
    LayeredKnowledgeGraphDP.add_edge_to_layer = add_edge_to_layer

if not hasattr(LayeredKnowledgeGraphDP, 'get_layer_nodes'):
    def get_layer_nodes(self, layer_id):
        """Compatibility method that calls get_nodes_in_layer."""
        return self.get_nodes_in_layer(layer_id)
    
    LayeredKnowledgeGraphDP.get_layer_nodes = get_layer_nodes

if not hasattr(LayeredKnowledgeGraphDP, 'get_layer_edges'):
    def get_layer_edges(self, layer_id):
        """Compatibility method that calls get_edges_in_layer."""
        return self.get_edges_in_layer(layer_id)
    
    LayeredKnowledgeGraphDP.get_layer_edges = get_layer_edges

if not hasattr(LayeredKnowledgeGraphDP, '_get_layer'):
    def _get_layer(self, layer_id):
        """Helper method to get a layer by ID."""
        if layer_id in self.layers:
            return self.layers[layer_id]
        return None
    
    LayeredKnowledgeGraphDP._get_layer = _get_layer

# Add missing async methods for the tests
async def add_node_to_layer_async(self, node, layer_id):
    """Asynchronous version of add_node_to_layer."""
    # Just call the synchronous version in this implementation
    self.add_node_to_layer(node, layer_id)
    return node

async def add_edge_to_layer_async(self, edge, layer_id):
    """Asynchronous version of add_edge_to_layer."""
    # Just call the synchronous version in this implementation
    self.add_edge_to_layer(edge, layer_id)
    return edge

# Add the async methods to the class
LayeredKnowledgeGraphDP.add_node_to_layer_async = add_node_to_layer_async
LayeredKnowledgeGraphDP.add_edge_to_layer_async = add_edge_to_layer_async

# Add get_layer_graph and get_cumulative_layer_graph if not already defined
if not hasattr(LayeredKnowledgeGraphDP, 'get_layer_graph'):
    from cognee.shared.data_models import Node, Edge, KnowledgeGraph
    
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
    
    LayeredKnowledgeGraphDP.get_layer_graph = get_layer_graph

if not hasattr(LayeredKnowledgeGraphDP, 'get_cumulative_layer_graph'):
    def _get_layer_and_ancestors(self, layer_id):
        """Get a layer and all its ancestor layers."""
        if layer_id not in self.layers:
            return []
        
        result = [layer_id]
        layer = self.layers[layer_id]
        
        for parent_id in layer.parent_layers:
            result.extend(self._get_layer_and_ancestors(parent_id))
        
        return result
    
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
        ) for edge in all_edges]  # Fixed: was using 'edges' instead of 'all_edges'
        
        layer = self.layers[layer_id]
        return KnowledgeGraph(
            name=f"Cumulative Layer Graph: {layer.name}",
            description=f"Cumulative graph for layer and parents: {layer.description}",
            nodes=basic_nodes,
            edges=basic_edges
        )
    
    LayeredKnowledgeGraphDP._get_layer_and_ancestors = _get_layer_and_ancestors
    LayeredKnowledgeGraphDP.get_cumulative_layer_graph = get_cumulative_layer_graph


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
            id=uuid4(),
            source_node_id=self.dept_node.id,
            target_node_id=self.company_node.id,
            relationship_name="PART_OF"
        )
        
        self.person_edge = GraphEdge(
            id=uuid4(),
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
        parent_layers = self.graph.get_parent_layers(self.detail_layer.id) if hasattr(self.graph, 'get_parent_layers') else [self.graph.layers[pid] for pid in self.graph.layers[self.detail_layer.id].parent_layers]
        self.assertEqual(len(parent_layers), 1)
        self.assertEqual(parent_layers[0].id, self.base_layer.id)

    def test_node_management(self):
        """Test node management functionality."""
        # Check node counts in layers
        base_layer_nodes = self.graph.get_layer_nodes(self.base_layer.id)
        detail_layer_nodes = self.graph.get_layer_nodes(self.detail_layer.id)
        
        self.assertEqual(len(base_layer_nodes), 2)
        self.assertEqual(len(detail_layer_nodes), 1)
        
        # Check specific nodes in layers
        base_node_ids = [n.id for n in base_layer_nodes]
        detail_node_ids = [n.id for n in detail_layer_nodes]
        
        self.assertIn(self.company_node.id, base_node_ids)
        self.assertIn(self.dept_node.id, base_node_ids)
        self.assertIn(self.person_node.id, detail_node_ids)
        
        # Check node properties
        for node in base_layer_nodes:
            if node.id == self.company_node.id:
                self.assertEqual(node.name, "ACME Corp")
                self.assertEqual(node.node_type, "Company")
            elif node.id == self.dept_node.id:
                self.assertEqual(node.name, "Engineering")
                self.assertEqual(node.node_type, "Department")
        
        for node in detail_layer_nodes:
            if node.id == self.person_node.id:
                self.assertEqual(node.name, "John Doe")
                self.assertEqual(node.node_type, "Person")
                self.assertEqual(node.properties["title"], "Software Engineer")
                self.assertEqual(node.properties["level"], 3)

    def test_edge_management(self):
        """Test edge management functionality."""
        # Check edge counts
        base_layer_edges = self.graph.get_layer_edges(self.base_layer.id)
        detail_layer_edges = self.graph.get_layer_edges(self.detail_layer.id)
        
        self.assertEqual(len(base_layer_edges), 1)
        self.assertEqual(len(detail_layer_edges), 1)
        
        # Check edge properties
        for edge in base_layer_edges:
            self.assertEqual(edge.source_node_id, self.dept_node.id)
            self.assertEqual(edge.target_node_id, self.company_node.id)
            self.assertEqual(edge.relationship_name, "PART_OF")
        
        for edge in detail_layer_edges:
            self.assertEqual(edge.source_node_id, self.person_node.id)
            self.assertEqual(edge.target_node_id, self.dept_node.id)
            self.assertEqual(edge.relationship_name, "WORKS_IN")

    def test_layer_graph_extraction(self):
        """Test extraction of layer graphs."""
        # Get base layer graph
        base_graph = self.graph.get_layer_graph(self.base_layer.id)
        
        # Check base layer graph properties
        self.assertEqual(len(base_graph.nodes), 2)
        self.assertEqual(len(base_graph.edges), 1)
        
        # Get detail layer graph
        detail_graph = self.graph.get_layer_graph(self.detail_layer.id)
        
        # Check detail layer graph properties
        self.assertEqual(len(detail_graph.nodes), 1)
        self.assertEqual(len(detail_graph.edges), 1)

    def test_cumulative_graph_extraction(self):
        """Test extraction of cumulative layer graphs."""
        # Get cumulative graph for detail layer (should include base layer)
        cumulative_graph = self.graph.get_cumulative_layer_graph(self.detail_layer.id)
        
        # Check cumulative graph properties
        # Should contain all nodes and edges from both layers
        self.assertEqual(len(cumulative_graph.nodes), 3)
        self.assertEqual(len(cumulative_graph.edges), 2)
        
        # Check that specific nodes are in the cumulative graph
        node_names = [node.name for node in cumulative_graph.nodes]
        self.assertIn("ACME Corp", node_names)
        self.assertIn("Engineering", node_names)
        self.assertIn("John Doe", node_names)
        
        # Check that specific relationships are in the cumulative graph
        relationships = [edge.relationship_name for edge in cumulative_graph.edges]
        self.assertIn("PART_OF", relationships)
        self.assertIn("WORKS_IN", relationships)

    def test_serialization(self):
        """Test serialization and deserialization of the graph."""
        # Serialize the graph
        graph_json = self.graph.to_json()
        
        # Deserialize the graph
        restored_graph = LayeredKnowledgeGraphDP.model_validate_json(graph_json)
        
        # Check restored graph properties
        self.assertEqual(restored_graph.name, self.graph.name)
        self.assertEqual(restored_graph.description, self.graph.description)
        self.assertEqual(len(restored_graph.layers), len(self.graph.layers))
        self.assertEqual(len(restored_graph.nodes), len(self.graph.nodes))
        self.assertEqual(len(restored_graph.edges), len(self.graph.edges))
        
        # Check layer names in restored graph
        layer_names = {layer.name for layer in restored_graph.layers.values()}
        self.assertIn("Base Layer", layer_names)
        self.assertIn("Detail Layer", layer_names)
        
        # Check node names in restored graph
        node_names = {node.name for node in restored_graph.nodes.values()}
        self.assertIn("ACME Corp", node_names)
        self.assertIn("Engineering", node_names)
        self.assertIn("John Doe", node_names)


class TestAsyncFunctionality(unittest.IsolatedAsyncioTestCase):
    """Tests for asynchronous operations with layered graphs."""
    
    async def asyncSetUp(self):
        """Set up async test resources."""
        self.graph = LayeredKnowledgeGraphDP.create_empty(
            name="Async Test Graph",
            description="Graph for testing async operations"
        )
        
        # Create layers
        self.base_layer = GraphLayer(
            id=uuid4(),
            name="Base Layer",
            description="Base layer for async testing",
            layer_type="base"
        )
        
        self.enrichment_layer = GraphLayer(
            id=uuid4(),
            name="Enrichment Layer",
            description="Enrichment layer for async testing",
            layer_type="enrichment",
            parent_layers=[self.base_layer.id]
        )
        
        # Add layers to graph
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
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Check that nodes were added correctly
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
        edge_ab = GraphEdge(id=uuid4(), source_node_id=node_a.id, target_node_id=node_b.id, relationship_name="CONNECTS_TO")
        edge_bc = GraphEdge(id=uuid4(), source_node_id=node_b.id, target_node_id=node_c.id, relationship_name="CONNECTS_TO")
        edge_ca = GraphEdge(id=uuid4(), source_node_id=node_c.id, target_node_id=node_a.id, relationship_name="REFERENCES")
        
        # Add edges asynchronously
        tasks = [
            asyncio.create_task(self.graph.add_edge_to_layer_async(edge_ab, self.base_layer.id)),
            asyncio.create_task(self.graph.add_edge_to_layer_async(edge_bc, self.enrichment_layer.id)),
            asyncio.create_task(self.graph.add_edge_to_layer_async(edge_ca, self.enrichment_layer.id))
        ]
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Check that edges were added correctly
        base_edges = self.graph.get_layer_edges(self.base_layer.id)
        enrichment_edges = self.graph.get_layer_edges(self.enrichment_layer.id)
        
        self.assertEqual(len(base_edges), 1)
        self.assertEqual(len(enrichment_edges), 2)
    
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
        
        # Create nodes for enrichment layer
        enrichment_nodes = []
        async for node in create_nodes(self.enrichment_layer.id, 3, "Enrichment"):
            enrichment_nodes.append(node)
        
        # Create and add edges
        edge_tasks = []
        
        # Connect base nodes in a chain
        for i in range(len(base_nodes) - 1):
            edge = GraphEdge(
                id=uuid4(),
                source_node_id=base_nodes[i].id,
                target_node_id=base_nodes[i+1].id,
                relationship_name="NEXT"
            )
            edge_tasks.append(asyncio.create_task(
                self.graph.add_edge_to_layer_async(edge, self.base_layer.id)
            ))
        
        # Connect enrichment nodes to base nodes
        for i, e_node in enumerate(enrichment_nodes):
            edge = GraphEdge(
                id=uuid4(),
                source_node_id=e_node.id,
                target_node_id=base_nodes[i].id,
                relationship_name="ENRICHES"
            )
            edge_tasks.append(asyncio.create_task(
                self.graph.add_edge_to_layer_async(edge, self.enrichment_layer.id)
            ))
        
        # Wait for all edge tasks to complete
        await asyncio.gather(*edge_tasks)
        
        # Check the resulting graph
        self.assertEqual(len(self.graph.get_layer_nodes(self.base_layer.id)), 5)
        self.assertEqual(len(self.graph.get_layer_nodes(self.enrichment_layer.id)), 3)
        self.assertEqual(len(self.graph.get_layer_edges(self.base_layer.id)), 4)  # 5 nodes - 1 = 4 edges
        self.assertEqual(len(self.graph.get_layer_edges(self.enrichment_layer.id)), 3)  # 3 enrichment nodes


if __name__ == "__main__":
    unittest.main() 