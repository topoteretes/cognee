import os
import shutil
import pathlib
import pytest
import asyncio
from datetime import datetime, timezone

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger

logger = get_logger()


async def test_kuzu_frequency_tracking():
    """Test frequency tracking functionality in Kuzu adapter."""
    # Clean up test directories before starting
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_kuzu_frequency")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_kuzu_frequency")
        ).resolve()
    )

    try:
        # Set Kuzu as the graph database provider
        cognee.config.set_graph_database_provider("kuzu")
        cognee.config.data_root_directory(data_directory_path)
        cognee.config.system_root_directory(cognee_directory_path)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        # Get the Kuzu graph engine
        graph_engine = await get_graph_engine()

        # Ensure database is empty
        is_empty = await graph_engine.is_empty()
        assert is_empty, "Kuzu graph database is not empty"

        # Test 1: Add some test nodes
        from cognee.infrastructure.engine import DataPoint

        class TestNode(DataPoint):
            pass

        test_nodes = [
            TestNode(id="node1", name="Test Node 1", type="TestType"),
            TestNode(id="node2", name="Test Node 2", type="TestType"),
            TestNode(id="node3", name="Test Node 3", type="TestType"),
        ]

        await graph_engine.add_nodes(test_nodes)

        # Test 2: Update frequency weights
        frequency_data = {
            "node1": 5.0,
            "node2": 3.0,
            "node3": 1.5,
        }

        await graph_engine.update_node_frequencies(frequency_data)
        logger.info("Successfully updated frequency weights")

        # Test 3: Retrieve nodes and verify frequency weights
        node1_data = await graph_engine.get_node("node1")
        assert node1_data is not None, "Node1 not found"
        assert node1_data.get("frequency_weight") == 5.0, (
            f"Expected frequency_weight 5.0, got {node1_data.get('frequency_weight')}"
        )
        assert node1_data.get("frequency_updated_at") is not None, "frequency_updated_at not set"

        node2_data = await graph_engine.get_node("node2")
        assert node2_data is not None, "Node2 not found"
        assert node2_data.get("frequency_weight") == 3.0, (
            f"Expected frequency_weight 3.0, got {node2_data.get('frequency_weight')}"
        )

        node3_data = await graph_engine.get_node("node3")
        assert node3_data is not None, "Node3 not found"
        assert node3_data.get("frequency_weight") == 1.5, (
            f"Expected frequency_weight 1.5, got {node3_data.get('frequency_weight')}"
        )

        # Test 4: Test get_node_frequencies method
        frequencies = await graph_engine.get_node_frequencies(["node1", "node2", "node3"])
        assert frequencies["node1"] == 5.0, (
            f"Expected frequency 5.0 for node1, got {frequencies.get('node1')}"
        )
        assert frequencies["node2"] == 3.0, (
            f"Expected frequency 3.0 for node2, got {frequencies.get('node2')}"
        )
        assert frequencies["node3"] == 1.5, (
            f"Expected frequency 1.5 for node3, got {frequencies.get('node3')}"
        )

        # Test 5: Update frequencies again to test updating existing values
        updated_frequency_data = {
            "node1": 10.0,
            "node2": 7.0,
        }

        await graph_engine.update_node_frequencies(updated_frequency_data)

        # Verify updated frequencies
        updated_frequencies = await graph_engine.get_node_frequencies(["node1", "node2"])
        assert updated_frequencies["node1"] == 10.0, (
            f"Expected updated frequency 10.0 for node1, got {updated_frequencies.get('node1')}"
        )
        assert updated_frequencies["node2"] == 7.0, (
            f"Expected updated frequency 7.0 for node2, got {updated_frequencies.get('node2')}"
        )

        # Test 6: Test get_node_frequencies without specific node IDs (get all non-zero frequencies)
        all_frequencies = await graph_engine.get_node_frequencies()
        assert len(all_frequencies) >= 3, (
            f"Expected at least 3 nodes with frequencies, got {len(all_frequencies)}"
        )
        assert all_frequencies.get("node1") == 10.0
        assert all_frequencies.get("node2") == 7.0
        assert all_frequencies.get("node3") == 1.5

        # Test 7: Test empty frequency update (should not error)
        await graph_engine.update_node_frequencies({})

        # Test 8: Test extraction methods return frequency fields
        extracted_nodes = await graph_engine.extract_nodes(["node1", "node2"])
        assert len(extracted_nodes) == 2, f"Expected 2 extracted nodes, got {len(extracted_nodes)}"
        for node in extracted_nodes:
            assert "frequency_weight" in node, f"frequency_weight not in extracted node: {node}"
            assert "frequency_updated_at" in node, (
                f"frequency_updated_at not in extracted node: {node}"
            )

        # Test 9: Test graph_data methods include frequency fields
        nodes, edges = await graph_engine.get_graph_data()
        if nodes:
            # Check that at least some nodes have frequency fields
            node_with_freq = False
            for node_id, node_props in nodes:
                if "frequency_weight" in node_props or "frequency_updated_at" in node_props:
                    node_with_freq = True
                    break
            assert node_with_freq, "No nodes found with frequency fields in get_graph_data"

        logger.info("All frequency tracking tests passed!")

        # Clean up
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

    finally:
        # Ensure cleanup even if tests fail
        for path in [data_directory_path, cognee_directory_path]:
            if os.path.exists(path):
                shutil.rmtree(path)


if __name__ == "__main__":
    asyncio.run(test_kuzu_frequency_tracking())
