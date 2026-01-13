"""
Test Suite: Usage Frequency Tracking

Comprehensive tests for the usage frequency tracking implementation.
Tests cover extraction logic, adapter integration, edge cases, and end-to-end workflows.

Run with:
    pytest test_usage_frequency_comprehensive.py -v

Or without pytest:
    python test_usage_frequency_comprehensive.py
"""

import asyncio
import unittest
from datetime import datetime, timedelta
from typing import List, Dict

# Mock imports for testing without full Cognee setup
try:
    from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
    from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge
    from cognee.tasks.memify.extract_usage_frequency import (
        extract_usage_frequency,
        add_frequency_weights,
        run_usage_frequency_update,
    )

    COGNEE_AVAILABLE = True
except ImportError:
    COGNEE_AVAILABLE = False
    print("⚠ Cognee not fully available - some tests will be skipped")


class TestUsageFrequencyExtraction(unittest.TestCase):
    """Test the core frequency extraction logic."""

    def setUp(self):
        """Set up test fixtures."""
        if not COGNEE_AVAILABLE:
            self.skipTest("Cognee modules not available")

    def create_mock_graph(self, num_interactions: int = 3, num_elements: int = 5):
        """Create a mock graph with interactions and elements."""
        graph = CogneeGraph()

        # Create interaction nodes
        current_time = datetime.now()
        for i in range(num_interactions):
            interaction_node = Node(
                id=f"interaction_{i}",
                node_type="CogneeUserInteraction",
                attributes={
                    "type": "CogneeUserInteraction",
                    "query_text": f"Test query {i}",
                    "timestamp": int((current_time - timedelta(hours=i)).timestamp() * 1000),
                },
            )
            graph.add_node(interaction_node)

        # Create graph element nodes
        for i in range(num_elements):
            element_node = Node(
                id=f"element_{i}",
                node_type="DocumentChunk",
                attributes={"type": "DocumentChunk", "text": f"Element content {i}"},
            )
            graph.add_node(element_node)

        # Create usage edges (interactions reference elements)
        for i in range(num_interactions):
            # Each interaction uses 2-3 elements
            for j in range(2):
                element_idx = (i + j) % num_elements
                edge = Edge(
                    node1=graph.get_node(f"interaction_{i}"),
                    node2=graph.get_node(f"element_{element_idx}"),
                    edge_type="used_graph_element_to_answer",
                    attributes={"relationship_type": "used_graph_element_to_answer"},
                )
                graph.add_edge(edge)

        return graph

    async def test_basic_frequency_extraction(self):
        """Test basic frequency extraction with simple graph."""
        graph = self.create_mock_graph(num_interactions=3, num_elements=5)

        result = await extract_usage_frequency(
            subgraphs=[graph], time_window=timedelta(days=7), min_interaction_threshold=1
        )

        self.assertIn("node_frequencies", result)
        self.assertIn("total_interactions", result)
        self.assertEqual(result["total_interactions"], 3)
        self.assertGreater(len(result["node_frequencies"]), 0)

    async def test_time_window_filtering(self):
        """Test that time window correctly filters old interactions."""
        graph = CogneeGraph()

        current_time = datetime.now()

        # Add recent interaction (within window)
        recent_node = Node(
            id="recent_interaction",
            node_type="CogneeUserInteraction",
            attributes={
                "type": "CogneeUserInteraction",
                "timestamp": int(current_time.timestamp() * 1000),
            },
        )
        graph.add_node(recent_node)

        # Add old interaction (outside window)
        old_node = Node(
            id="old_interaction",
            node_type="CogneeUserInteraction",
            attributes={
                "type": "CogneeUserInteraction",
                "timestamp": int((current_time - timedelta(days=10)).timestamp() * 1000),
            },
        )
        graph.add_node(old_node)

        # Add element
        element = Node(
            id="element_1", node_type="DocumentChunk", attributes={"type": "DocumentChunk"}
        )
        graph.add_node(element)

        # Add edges
        graph.add_edge(
            Edge(
                node1=recent_node,
                node2=element,
                edge_type="used_graph_element_to_answer",
                attributes={"relationship_type": "used_graph_element_to_answer"},
            )
        )
        graph.add_edge(
            Edge(
                node1=old_node,
                node2=element,
                edge_type="used_graph_element_to_answer",
                attributes={"relationship_type": "used_graph_element_to_answer"},
            )
        )

        # Extract with 7-day window
        result = await extract_usage_frequency(
            subgraphs=[graph], time_window=timedelta(days=7), min_interaction_threshold=1
        )

        # Should only count recent interaction
        self.assertEqual(result["interactions_in_window"], 1)
        self.assertEqual(result["total_interactions"], 2)

    async def test_threshold_filtering(self):
        """Test that minimum threshold filters low-frequency nodes."""
        graph = self.create_mock_graph(num_interactions=5, num_elements=10)

        # Extract with threshold of 3
        result = await extract_usage_frequency(
            subgraphs=[graph], time_window=timedelta(days=7), min_interaction_threshold=3
        )

        # Only nodes with 3+ accesses should be included
        for node_id, freq in result["node_frequencies"].items():
            self.assertGreaterEqual(freq, 3)

    async def test_element_type_tracking(self):
        """Test that element types are properly tracked."""
        graph = CogneeGraph()

        # Create interaction
        interaction = Node(
            id="interaction_1",
            node_type="CogneeUserInteraction",
            attributes={
                "type": "CogneeUserInteraction",
                "timestamp": int(datetime.now().timestamp() * 1000),
            },
        )
        graph.add_node(interaction)

        # Create elements of different types
        chunk = Node(id="chunk_1", node_type="DocumentChunk", attributes={"type": "DocumentChunk"})
        entity = Node(id="entity_1", node_type="Entity", attributes={"type": "Entity"})

        graph.add_node(chunk)
        graph.add_node(entity)

        # Add edges
        for element in [chunk, entity]:
            graph.add_edge(
                Edge(
                    node1=interaction,
                    node2=element,
                    edge_type="used_graph_element_to_answer",
                    attributes={"relationship_type": "used_graph_element_to_answer"},
                )
            )

        result = await extract_usage_frequency(subgraphs=[graph], time_window=timedelta(days=7))

        # Check element types were tracked
        self.assertIn("element_type_frequencies", result)
        types = result["element_type_frequencies"]
        self.assertIn("DocumentChunk", types)
        self.assertIn("Entity", types)

    async def test_empty_graph(self):
        """Test handling of empty graph."""
        graph = CogneeGraph()

        result = await extract_usage_frequency(subgraphs=[graph], time_window=timedelta(days=7))

        self.assertEqual(result["total_interactions"], 0)
        self.assertEqual(len(result["node_frequencies"]), 0)

    async def test_no_interactions_in_window(self):
        """Test handling when all interactions are outside time window."""
        graph = CogneeGraph()

        # Add old interaction
        old_time = datetime.now() - timedelta(days=30)
        old_interaction = Node(
            id="old_interaction",
            node_type="CogneeUserInteraction",
            attributes={
                "type": "CogneeUserInteraction",
                "timestamp": int(old_time.timestamp() * 1000),
            },
        )
        graph.add_node(old_interaction)

        result = await extract_usage_frequency(subgraphs=[graph], time_window=timedelta(days=7))

        self.assertEqual(result["interactions_in_window"], 0)
        self.assertEqual(result["total_interactions"], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""

    def setUp(self):
        """Set up test fixtures."""
        if not COGNEE_AVAILABLE:
            self.skipTest("Cognee modules not available")

    async def test_end_to_end_workflow(self):
        """Test the complete end-to-end frequency tracking workflow."""
        # This would require a full Cognee setup with database
        # Skipped in unit tests, run as part of example_usage_frequency_e2e.py
        self.skipTest("E2E test - run example_usage_frequency_e2e.py instead")


# ============================================================================
# Test Runner
# ============================================================================


def run_async_test(test_func):
    """Helper to run async test functions."""
    asyncio.run(test_func())


def main():
    """Run all tests."""
    if not COGNEE_AVAILABLE:
        print("⚠ Cognee not available - skipping tests")
        print("Install with: pip install cognee[neo4j]")
        return

    print("=" * 80)
    print("Running Usage Frequency Tests")
    print("=" * 80)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestUsageFrequencyExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(main())
