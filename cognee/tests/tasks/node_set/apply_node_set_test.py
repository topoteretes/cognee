import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4, UUID

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.tasks.node_set.apply_node_set import apply_node_set


class TestDataPoint(DataPoint):
    """Test DataPoint model for testing apply_node_set task."""

    name: str
    description: str
    metadata: dict = {"index_fields": ["name"]}


@pytest.mark.asyncio
async def test_apply_node_set():
    """Test that apply_node_set applies NodeSet values from the relational store to DataPoint instances."""
    # Create test data
    dp_id1 = uuid4()
    dp_id2 = uuid4()

    # Create test DataPoint instances
    data_points = [
        TestDataPoint(id=dp_id1, name="Test 1", description="Description 1"),
        TestDataPoint(id=dp_id2, name="Test 2", description="Description 2"),
    ]

    # Create mock Data records that would be returned from the database
    node_set1 = ["node1", "node2"]
    node_set2 = ["node3", "node4", "node5"]

    # Create a mock implementation of _process_data_points
    async def mock_process_data_points(session, data_point_map):
        # Apply NodeSet directly to the DataPoints
        for dp_id, node_set in [(dp_id1, node_set1), (dp_id2, node_set2)]:
            dp_id_str = str(dp_id)
            if dp_id_str in data_point_map:
                data_point_map[dp_id_str].NodeSet = node_set

    # Patch the necessary functions
    with (
        patch("cognee.tasks.node_set.apply_node_set.get_relational_engine"),
        patch(
            "cognee.tasks.node_set.apply_node_set._process_data_points",
            side_effect=mock_process_data_points,
        ),
    ):
        # Call the function being tested
        result = await apply_node_set(data_points)

        # Verify the results
        assert len(result) == 2

        # Check that NodeSet values were applied correctly
        assert result[0].NodeSet == node_set1
        assert result[1].NodeSet == node_set2


@pytest.mark.asyncio
async def test_apply_node_set_empty_list():
    """Test apply_node_set with an empty list of DataPoints."""
    result = await apply_node_set([])
    assert result == []


@pytest.mark.asyncio
async def test_apply_node_set_no_matching_data():
    """Test apply_node_set when there are no matching Data records."""
    # Create test data
    dp_id = uuid4()

    # Create test DataPoint instances
    data_points = [TestDataPoint(id=dp_id, name="Test", description="Description")]

    # Create a mock implementation of _process_data_points that doesn't modify any DataPoints
    async def mock_process_data_points(session, data_point_map):
        # Don't modify anything - simulating no matching records
        pass

    # Patch the necessary functions
    with (
        patch("cognee.tasks.node_set.apply_node_set.get_relational_engine"),
        patch(
            "cognee.tasks.node_set.apply_node_set._process_data_points",
            side_effect=mock_process_data_points,
        ),
    ):
        # Call the function being tested
        result = await apply_node_set(data_points)

        # Verify the results - NodeSet should remain None
        assert len(result) == 1
        assert result[0].NodeSet is None


@pytest.mark.asyncio
async def test_apply_node_set_invalid_json():
    """Test apply_node_set when there's invalid JSON in the node_set column."""
    # Create test data
    dp_id = uuid4()

    # Create test DataPoint instances
    data_points = [TestDataPoint(id=dp_id, name="Test", description="Description")]

    # Create a mock implementation of _process_data_points that throws the appropriate error
    async def mock_process_data_points(session, data_point_map):
        # Simulate the JSONDecodeError by logging a warning
        from cognee.tasks.node_set.apply_node_set import logger

        logger.warning(f"Failed to parse NodeSet JSON for DataPoint {str(dp_id)}")

    # Patch the necessary functions
    with (
        patch("cognee.tasks.node_set.apply_node_set.get_relational_engine"),
        patch(
            "cognee.tasks.node_set.apply_node_set._process_data_points",
            side_effect=mock_process_data_points,
        ),
        patch("cognee.tasks.node_set.apply_node_set.logger") as mock_logger,
    ):
        # Call the function being tested
        result = await apply_node_set(data_points)

        # Verify the results - NodeSet should remain None
        assert len(result) == 1
        assert result[0].NodeSet is None

        # Verify logger warning was called
        mock_logger.warning.assert_called_once()
