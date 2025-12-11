"""Tests for DataItem custom label functionality"""

import os
import pathlib
import asyncio
import pytest
from uuid import UUID

import cognee
from cognee.modules.data.models import DataItem
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.data.methods import get_dataset_data, get_datasets

logger = get_logger()


@pytest.fixture
async def setup_cognee():
    """Setup Cognee for testing"""
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_data_item")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_data_item")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    yield

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.mark.asyncio
async def test_data_item_with_label(setup_cognee):
    """Test that DataItem with custom label is properly handled during ingestion"""
    user = await get_default_user()

    # Create a DataItem with custom label
    test_text = "This is test data for DataItem label functionality."
    custom_label = "My Custom Test Data"
    data_item = DataItem(data=test_text, label=custom_label)

    # Add the data to Cognee
    from cognee.tasks.ingestion import ingest_data

    dataset_name = "test_dataset_with_labels"
    result = await ingest_data(
        data=data_item,
        dataset_name=dataset_name,
        user=user,
    )

    # Verify that data was ingested
    assert len(result) > 0, "Data should have been ingested"

    # Get the datasets
    datasets = await get_datasets(user.id)
    test_dataset = next((d for d in datasets if d.name == dataset_name), None)
    assert test_dataset is not None, f"Dataset '{dataset_name}' should exist"

    # Get the dataset data
    dataset_data = await get_dataset_data(test_dataset.id)
    assert len(dataset_data) > 0, "Dataset should contain data"

    # Check that the label was properly set
    data_item_record = dataset_data[0]
    assert data_item_record.label == custom_label, (
        f"Label should be '{custom_label}', got '{data_item_record.label}'"
    )
    assert data_item_record.name is not None, "Name should not be None"


@pytest.mark.asyncio
async def test_data_item_without_label(setup_cognee):
    """Test that DataItem without label works correctly (label is None)"""
    user = await get_default_user()

    # Create a DataItem without a label (label will be None)
    test_text = "This is test data without a label."
    data_item = DataItem(data=test_text)

    # Add the data to Cognee
    from cognee.tasks.ingestion import ingest_data

    dataset_name = "test_dataset_without_labels"
    result = await ingest_data(
        data=data_item,
        dataset_name=dataset_name,
        user=user,
    )

    # Verify that data was ingested
    assert len(result) > 0, "Data should have been ingested"

    # Get the datasets
    datasets = await get_datasets(user.id)
    test_dataset = next((d for d in datasets if d.name == dataset_name), None)
    assert test_dataset is not None, f"Dataset '{dataset_name}' should exist"

    # Get the dataset data
    dataset_data = await get_dataset_data(test_dataset.id)
    assert len(dataset_data) > 0, "Dataset should contain data"

    # Check that the label is None
    data_item_record = dataset_data[0]
    assert data_item_record.label is None, "Label should be None when not provided"


@pytest.mark.asyncio
async def test_plain_string_still_works(setup_cognee):
    """Test that plain strings still work without DataItem wrapper"""
    user = await get_default_user()

    # Add plain string data (without DataItem wrapper)
    test_text = "Plain text data without DataItem wrapper"

    from cognee.tasks.ingestion import ingest_data

    dataset_name = "test_dataset_plain_string"
    result = await ingest_data(
        data=test_text,
        dataset_name=dataset_name,
        user=user,
    )

    # Verify that data was ingested
    assert len(result) > 0, "Data should have been ingested"

    # Get the datasets
    datasets = await get_datasets(user.id)
    test_dataset = next((d for d in datasets if d.name == dataset_name), None)
    assert test_dataset is not None, f"Dataset '{dataset_name}' should exist"

    # Get the dataset data
    dataset_data = await get_dataset_data(test_dataset.id)
    assert len(dataset_data) > 0, "Dataset should contain data"

    # Check that label is None for plain string
    data_item_record = dataset_data[0]
    assert data_item_record.label is None, "Label should be None for plain string data"


@pytest.mark.asyncio
async def test_multiple_data_items_with_labels(setup_cognee):
    """Test that multiple DataItems with different labels are properly handled"""
    user = await get_default_user()

    # Create multiple DataItems with different labels
    items = [
        DataItem(data="First text item", label="Item 1"),
        DataItem(data="Second text item", label="Item 2"),
        DataItem(data="Third text item", label="Item 3"),
    ]

    from cognee.tasks.ingestion import ingest_data

    dataset_name = "test_dataset_multiple_labels"
    result = await ingest_data(
        data=items,
        dataset_name=dataset_name,
        user=user,
    )

    # Verify that all items were ingested
    assert len(result) >= 3, "All 3 items should have been ingested"

    # Get the datasets
    datasets = await get_datasets(user.id)
    test_dataset = next((d for d in datasets if d.name == dataset_name), None)
    assert test_dataset is not None, f"Dataset '{dataset_name}' should exist"

    # Get the dataset data
    dataset_data = await get_dataset_data(test_dataset.id)
    assert len(dataset_data) >= 3, "Dataset should contain at least 3 items"

    # Verify labels are properly set for each item
    labels = {item.label for item in dataset_data if item.label is not None}
    assert len(labels) >= 3, "Should have at least 3 unique labels"


@pytest.mark.asyncio
async def test_data_dto_includes_label():
    """Test that DataDTO in API responses includes label field"""
    from cognee.api.v1.datasets.routers.get_datasets_router import DataDTO
    from datetime import datetime

    # Create a DataDTO instance with label
    data_dto = DataDTO(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        name="Test Data",
        label="Custom Label",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        extension="txt",
        mime_type="text/plain",
        raw_data_location="/path/to/data",
        dataset_id=UUID("87654321-4321-8765-4321-876543218765"),
    )

    # Verify that label is accessible
    assert data_dto.label == "Custom Label", "Label should be accessible in DataDTO"

    # Create a DataDTO without label (should be None)
    data_dto_no_label = DataDTO(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        name="Test Data",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        extension="txt",
        mime_type="text/plain",
        raw_data_location="/path/to/data",
        dataset_id=UUID("87654321-4321-8765-4321-876543218765"),
    )

    # Verify that label is None when not provided
    assert data_dto_no_label.label is None, "Label should be None when not provided in DataDTO"


if __name__ == "__main__":
    # For running the tests manually
    pytest.main([__file__, "-v"])
