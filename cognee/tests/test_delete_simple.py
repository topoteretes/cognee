"""
Simple test for the new delete by ID functionality.
This test focuses on the core functionality without complex permission testing.
"""

import asyncio
import os
import pathlib
import cognee
from cognee.modules.users.methods import get_default_user
from cognee.modules.data.methods import get_dataset_data, get_datasets_by_name


async def test_delete_by_id():
    """Test the basic delete by ID functionality."""
    
    print("🧪 Simple Delete by ID Test")
    print("=" * 40)
    
    # Setup test environment
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_simple_delete")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_simple_delete")
        ).resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    # Clean up
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    
    # Initialize database by adding some data first (this triggers setup)
    test_text = "This is a test document for deletion by ID."
    dataset_name = "test_dataset"
    
    print(f"📝 Adding test data to dataset: {dataset_name}")
    await cognee.add([test_text], dataset_name=dataset_name)
    await cognee.cognify([dataset_name])
    
    # Now get default user (database should be initialized)
    user = await get_default_user()
    
    # Get the dataset and data
    datasets = await get_datasets_by_name([dataset_name], user.id)
    assert len(datasets) == 1, "Dataset should be created"
    
    dataset = datasets[0]
    dataset_data = await get_dataset_data(dataset.id)
    assert len(dataset_data) > 0, "Dataset should have data"
    
    data_id = dataset_data[0].id
    print(f"✅ Data created with ID: {data_id}")
    
    # Test the delete function
    print(f"🗑️ Deleting data with ID: {data_id}")
    
    try:
        result = await cognee.delete(
            data_id=data_id,
            dataset_id=dataset.id,
            user=user
        )
        print("✅ Delete operation completed successfully")
        print(f"Result: {result}")
        
        # Verify data is deleted
        remaining_data = await get_dataset_data(dataset.id)
        print(f"📊 Remaining data count: {len(remaining_data)}")
        
        assert len(remaining_data) == 0, "Data should be deleted"
        print("✅ Data successfully removed from dataset")
        
    except Exception as e:
        print(f"❌ Delete operation failed: {e}")
        raise
    
    print("\n🎉 Simple delete by ID test passed!")


if __name__ == "__main__":
    asyncio.run(test_delete_by_id()) 