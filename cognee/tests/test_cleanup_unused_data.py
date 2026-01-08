import os
import pathlib
import cognee
from datetime import datetime, timezone, timedelta
from uuid import UUID
from sqlalchemy import select, update
from cognee.modules.data.models import Data, DatasetData
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.search.types import SearchType

logger = get_logger()


async def test_textdocument_cleanup_with_sql():
    """
    End-to-end test for TextDocument cleanup based on last_accessed timestamps.
    """
    # Enable last accessed tracking BEFORE any cognee operations
    os.environ["ENABLE_LAST_ACCESSED"] = "true"

    # Setup test directories
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_cleanup")
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_cleanup")
        ).resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    # Initialize database
    from cognee.modules.engine.operations.setup import setup

    # Clean slate
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    logger.info("🧪 Testing TextDocument cleanup based on last_accessed")

    # Step 1: Add and cognify a test document
    dataset_name = "test_cleanup_dataset"
    test_text = """
    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. Deep learning uses
    neural networks with multiple layers to process data.
    """

    await setup()
    user = await get_default_user()
    await cognee.add([test_text], dataset_name=dataset_name, user=user)

    cognify_result = await cognee.cognify([dataset_name], user=user)

    # Extract dataset_id from cognify result
    dataset_id = None
    for ds_id, pipeline_result in cognify_result.items():
        dataset_id = ds_id
        break

    assert dataset_id is not None, "Failed to get dataset_id from cognify result"
    logger.info(f"✅ Document added and cognified. Dataset ID: {dataset_id}")

    # Step 2: Perform search to trigger last_accessed update
    logger.info("Triggering search to update last_accessed...")
    search_results = await cognee.search(
        query_type=SearchType.CHUNKS,
        query_text="machine learning",
        datasets=[dataset_name],
        user=user,
    )
    logger.info(f"✅ Search completed, found {len(search_results)} results")
    assert len(search_results) > 0, "Search should return results"

    # Step 3: Verify last_accessed was set and get data_id
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(Data, DatasetData)
            .join(DatasetData, Data.id == DatasetData.data_id)
            .where(DatasetData.dataset_id == dataset_id)
        )
        data_records = result.all()
        assert len(data_records) > 0, "No Data records found for the dataset"
        data_record = data_records[0][0]
        data_id = data_record.id

        # Verify last_accessed is set
        assert data_record.last_accessed is not None, (
            "last_accessed should be set after search operation"
        )

        original_last_accessed = data_record.last_accessed
        logger.info(f"✅ last_accessed verified: {original_last_accessed}")

    # Step 4: Manually age the timestamp
    minutes_threshold = 30
    aged_timestamp = datetime.now(timezone.utc) - timedelta(minutes=minutes_threshold + 10)

    async with db_engine.get_async_session() as session:
        stmt = update(Data).where(Data.id == data_id).values(last_accessed=aged_timestamp)
        await session.execute(stmt)
        await session.commit()

    # Verify timestamp was updated
    async with db_engine.get_async_session() as session:
        result = await session.execute(select(Data).where(Data.id == data_id))
        updated_data = result.scalar_one_or_none()
        assert updated_data is not None, "Data record should exist"
        retrieved_timestamp = updated_data.last_accessed
        if retrieved_timestamp.tzinfo is None:
            retrieved_timestamp = retrieved_timestamp.replace(tzinfo=timezone.utc)
        assert retrieved_timestamp == aged_timestamp, "Timestamp should be updated to aged value"

    # Step 5: Test cleanup (document-level is now the default)
    from cognee.tasks.cleanup.cleanup_unused_data import cleanup_unused_data

    # First do a dry run
    logger.info("Testing dry run...")
    dry_run_result = await cleanup_unused_data(minutes_threshold=10, dry_run=True, user_id=user.id)

    # Debug: Print the actual result
    logger.info(f"Dry run result: {dry_run_result}")

    assert dry_run_result["status"] == "dry_run", (
        f"Status should be 'dry_run', got: {dry_run_result['status']}"
    )
    assert dry_run_result["unused_count"] > 0, "Should find at least one unused document"
    logger.info(f"✅ Dry run found {dry_run_result['unused_count']} unused documents")

    # Now run actual cleanup
    logger.info("Executing cleanup...")
    cleanup_result = await cleanup_unused_data(minutes_threshold=30, dry_run=False, user_id=user.id)

    assert cleanup_result["status"] == "completed", "Cleanup should complete successfully"
    assert cleanup_result["deleted_count"]["documents"] > 0, (
        "At least one document should be deleted"
    )
    logger.info(
        f"✅ Cleanup completed. Deleted {cleanup_result['deleted_count']['documents']} documents"
    )

    # Step 6: Verify deletion
    async with db_engine.get_async_session() as session:
        deleted_data = (
            await session.execute(select(Data).where(Data.id == data_id))
        ).scalar_one_or_none()
        assert deleted_data is None, "Data record should be deleted"
        logger.info("✅ Confirmed: Data record was deleted")

    logger.info("🎉 All cleanup tests passed!")
    return True


if __name__ == "__main__":
    import asyncio

    success = asyncio.run(test_textdocument_cleanup_with_sql())
    exit(0 if success else 1)
