import os
import json
import asyncio
import pytest
from uuid import uuid4
from unittest.mock import patch, AsyncMock, MagicMock
from contextlib import asynccontextmanager

import cognee
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.pipelines.tasks.Task import Task
from cognee.modules.pipelines import run_tasks
from cognee.tasks.node_set import apply_node_set
from cognee.infrastructure.databases.relational import create_db_and_tables


class TestDocument(DataPoint):
    """Test document model for NodeSet testing."""

    content: str
    metadata: dict = {"index_fields": ["content"]}


@pytest.mark.asyncio
async def test_node_set_add_to_cognify_workflow():
    """
    Test the full NodeSet workflow from add to cognify.

    This test verifies that:
    1. NodeSet data can be added using cognee.add
    2. The NodeSet data is stored in the relational database
    3. The apply_node_set task can retrieve the NodeSet and apply it to DataPoints
    """
    # Clean up any existing data
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Create test data
    test_content = "This is test content"
    test_node_set = ["node1", "node2", "node3"]
    dataset_name = f"test_dataset_{uuid4()}"

    # Create database tables
    await create_db_and_tables()

    # Mock functions to avoid external dependencies
    mock_add_data_points = AsyncMock()
    mock_add_data_points.return_value = []

    # Create a temporary file for the test
    temp_file_path = "/tmp/test_node_set.txt"
    with open(temp_file_path, "w") as f:
        f.write(test_content)

    try:
        # Mock ingest_data to capture and verify NodeSet
        original_ingest_data = cognee.tasks.ingestion.ingest_data

        async def mock_ingest_data(*args, **kwargs):
            # Call the original function but capture the NodeSet parameter
            assert len(args) >= 3, "Expected at least 3 arguments"
            assert args[1] == dataset_name, f"Expected dataset name {dataset_name}"
            assert kwargs.get("NodeSet") == test_node_set or args[3] == test_node_set, (
                "NodeSet not passed correctly"
            )
            return await original_ingest_data(*args, **kwargs)

        # Replace the ingest_data function temporarily
        with patch("cognee.tasks.ingestion.ingest_data", side_effect=mock_ingest_data):
            # Call the add function with NodeSet
            await cognee.add(temp_file_path, dataset_name, NodeSet=test_node_set)

            # Create test DataPoint for apply_node_set to process
            test_document = TestDocument(content=test_content)

            # Test the apply_node_set task
            with patch(
                "cognee.tasks.node_set.apply_node_set.get_relational_engine"
            ) as mock_get_engine:
                # Setup mock engine and session
                mock_session = AsyncMock()
                mock_engine = AsyncMock()

                # Properly mock the async context manager
                @asynccontextmanager
                async def mock_get_session():
                    try:
                        yield mock_session
                    finally:
                        pass

                mock_engine.get_async_session.return_value = mock_get_session()
                mock_get_engine.return_value = mock_engine

                # Create a mock Data object with our NodeSet
                class MockData:
                    def __init__(self, id, node_set):
                        self.id = id
                        self.node_set = node_set

                mock_data = MockData(test_document.id, json.dumps(test_node_set))

                # Setup the mock result
                mock_result = MagicMock()
                mock_result.scalars.return_value.all.return_value = [mock_data]
                mock_session.execute.return_value = mock_result

                # Run the apply_node_set task
                result = await apply_node_set([test_document])

                # Verify the NodeSet was applied
                assert len(result) == 1
                assert result[0].NodeSet == test_node_set

                # Verify the mock interactions
                mock_get_engine.assert_called_once()
                mock_engine.get_async_session.assert_called_once()
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Clean up after the test
        await cognee.prune.prune_data()


@pytest.mark.asyncio
async def test_node_set_in_cognify_pipeline():
    """
    Test the integration of apply_node_set task in the cognify pipeline.

    This test verifies that the apply_node_set task works correctly
    when run as part of a pipeline with other tasks.
    """
    # Create test data
    test_documents = [TestDocument(content="Document 1"), TestDocument(content="Document 2")]

    # Create a simple mock task that just passes data through
    async def mock_task(data):
        for item in data:
            yield item

    # Mock the apply_node_set function to verify it's called with the right data
    original_apply_node_set = apply_node_set

    apply_node_set_called = False

    async def mock_apply_node_set(data_points):
        nonlocal apply_node_set_called
        apply_node_set_called = True

        # Verify the input
        assert len(data_points) == 2
        assert all(isinstance(dp, TestDocument) for dp in data_points)

        # Apply NodeSet to demonstrate it worked
        for dp in data_points:
            dp.NodeSet = ["test_node"]

        return data_points

    # Create a pipeline with our tasks
    with patch("cognee.tasks.node_set.apply_node_set", side_effect=mock_apply_node_set):
        pipeline = run_tasks(
            tasks=[
                Task(mock_task),  # First task passes data through
                Task(apply_node_set),  # Second task applies NodeSet
            ],
            data=test_documents,
        )

        # Process all results from the pipeline
        results = []
        async for result in pipeline:
            results.extend(result)

        # Verify results
        assert apply_node_set_called, "apply_node_set was not called"
        assert len(results) == 2
        assert all(dp.NodeSet == ["test_node"] for dp in results)
