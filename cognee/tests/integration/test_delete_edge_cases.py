"""
Integration Tests: Delete Edge Cases

Tests for edge cases in the delete feature:
1. last_accessed field updates when retriever accesses data
2. cleanup_unused_data dry_run behavior
3. cleanup_unused_data actual deletion behavior

Test Coverage:
- test_last_accessed_updates_on_search: Verify last_accessed is updated during retrieval
- test_cleanup_unused_data_dry_run: Verify dry_run doesn't delete but reports correctly
- test_cleanup_actual_deletion: Verify actual cleanup deletes old documents
"""

import os
import pathlib
import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, update

import cognee
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data, DatasetData
from cognee.modules.engine.operations.setup import setup
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.tasks.cleanup.cleanup_unused_data import cleanup_unused_data

logger = get_logger()


@pytest.mark.asyncio
async def test_last_accessed_updates_on_search():
    """
    Test that last_accessed field is updated when retriever accesses data.

    Setup:
    - Enable ENABLE_LAST_ACCESSED
    - Add and cognify a document
    - Verify last_accessed is initially None or recent

    Operation:
    - Perform search query

    Expected:
    - last_accessed is updated to recent timestamp
    """
    # Enable last accessed tracking
    os.environ["ENABLE_LAST_ACCESSED"] = "true"

    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent.parent, ".data_storage/test_last_accessed_updates"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent.parent, ".cognee_system/test_last_accessed_updates"
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    user = await get_default_user()

    # Add and cognify a document
    dataset_name = "test_last_accessed"
    doc_text = "Apple is a technology company that makes smartphones and computers."

    await cognee.add([doc_text], dataset_name=dataset_name, user=user)
    cognify_result = await cognee.cognify([dataset_name], user=user)

    dataset_id = list(cognify_result.keys())[0]

    # Get the data_id
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(Data, DatasetData)
            .join(DatasetData, Data.id == DatasetData.data_id)
            .where(DatasetData.dataset_id == dataset_id)
        )
        data_records = result.all()
        assert len(data_records) > 0, "Should have at least one data record"

        data_before = data_records[0][0]
        data_id = data_before.id

        # Record timestamp before search
        last_accessed_before = data_before.last_accessed
        logger.info(f"last_accessed before search: {last_accessed_before}")

    # Wait a moment to ensure timestamp difference
    import asyncio

    await asyncio.sleep(0.1)

    # Perform search to trigger last_accessed update
    logger.info("Performing search to trigger last_accessed update...")
    search_results = await cognee.search(
        query_type=SearchType.CHUNKS,
        query_text="Apple technology",
        datasets=[dataset_name],
        user=user,
    )

    logger.info(f"Search returned {len(search_results)} results")

    # Check last_accessed after search
    async with db_engine.get_async_session() as session:
        result = await session.execute(select(Data).where(Data.id == data_id))
        data_after = result.scalar_one_or_none()

        assert data_after is not None, "Data should still exist"

        last_accessed_after = data_after.last_accessed
        logger.info(f"last_accessed after search: {last_accessed_after}")

        # Verify last_accessed was updated
        assert last_accessed_after is not None, "last_accessed should be set after search operation"

        # Verify timestamp is recent (within last 5 seconds)
        if last_accessed_after.tzinfo is None:
            last_accessed_after = last_accessed_after.replace(tzinfo=timezone.utc)

        time_diff = (datetime.now(timezone.utc) - last_accessed_after).total_seconds()
        assert time_diff < 5, (
            f"last_accessed should be recent (within 5 seconds), but was {time_diff}s ago"
        )

        logger.info(f"✅ last_accessed updated successfully (time_diff={time_diff:.2f}s)")

    logger.info("✅ test_last_accessed_updates_on_search PASSED")


@pytest.mark.asyncio
async def test_cleanup_unused_data_dry_run():
    """
    Test that cleanup_unused_data dry_run doesn't delete but reports correctly.

    Setup:
    - Enable ENABLE_LAST_ACCESSED
    - Create 5 documents
    - Age 3 documents to be "old" (last_accessed > threshold)
    - Keep 2 documents "new" (last_accessed < threshold)

    Operation:
    - Run cleanup_unused_data with dry_run=True

    Expected:
    - Status: "dry_run"
    - unused_count: 3
    - deleted_count: 0
    - All 5 documents still exist
    """
    # Enable last accessed tracking
    os.environ["ENABLE_LAST_ACCESSED"] = "true"

    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent.parent, ".data_storage/test_cleanup_dry_run"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent.parent, ".cognee_system/test_cleanup_dry_run"
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    user = await get_default_user()

    # Create dataset with multiple documents
    dataset_name = "test_cleanup_dry_run"

    docs = [
        "Document 1: Old document about cats",
        "Document 2: Old document about dogs",
        "Document 3: Old document about birds",
        "Document 4: New document about fish",
        "Document 5: New document about rabbits",
    ]

    data_ids = []
    for doc in docs:
        add_result = await cognee.add([doc], dataset_name=dataset_name, user=user)
        data_ids.append(add_result.data_ingestion_info[0]["data_id"])

    cognify_result = await cognee.cognify([dataset_name], user=user)
    dataset_id = list(cognify_result.keys())[0]

    # Age the first 3 documents to be "old"
    db_engine = get_relational_engine()
    threshold_minutes = 30
    aged_timestamp = datetime.now(timezone.utc) - timedelta(minutes=threshold_minutes + 10)
    recent_timestamp = datetime.now(timezone.utc) - timedelta(minutes=10)

    async with db_engine.get_async_session() as session:
        # Age first 3 documents
        for i in range(3):
            stmt = update(Data).where(Data.id == data_ids[i]).values(last_accessed=aged_timestamp)
            await session.execute(stmt)

        # Set last 2 documents as recently accessed
        for i in range(3, 5):
            stmt = update(Data).where(Data.id == data_ids[i]).values(last_accessed=recent_timestamp)
            await session.execute(stmt)

        await session.commit()

    logger.info("Aged 3 documents to be old, kept 2 as recent")

    # Run dry run cleanup
    logger.info("Running cleanup with dry_run=True...")
    result = await cleanup_unused_data(
        minutes_threshold=threshold_minutes, dry_run=True, user_id=user.id
    )

    logger.info(f"Cleanup result: {result}")

    # Assertions for dry run
    assert result["status"] == "dry_run", f"Status should be 'dry_run', got {result['status']}"
    assert result["unused_count"] == 3, (
        f"Should find 3 unused documents, found {result['unused_count']}"
    )
    assert result["deleted_count"]["documents"] == 0, (
        f"Dry run should not delete anything, deleted {result['deleted_count']['documents']}"
    )

    # Verify all documents still exist
    async with db_engine.get_async_session() as session:
        result = await session.execute(select(Data).where(Data.id.in_(data_ids)))
        remaining_data = result.all()
        assert len(remaining_data) == 5, (
            f"All 5 documents should still exist, found {len(remaining_data)}"
        )

    logger.info("✅ test_cleanup_unused_data_dry_run PASSED")


@pytest.mark.asyncio
async def test_cleanup_actual_deletion():
    """
    Test that cleanup_unused_data actually deletes old documents.

    Setup:
    - Enable ENABLE_LAST_ACCESSED
    - Create 5 documents
    - Age 3 documents to be "old"
    - Keep 2 documents "new"

    Operation:
    - Run cleanup_unused_data with dry_run=False

    Expected:
    - Status: "completed"
    - deleted_count: 3
    - Only 2 new documents remain
    """
    # Enable last accessed tracking
    os.environ["ENABLE_LAST_ACCESSED"] = "true"

    data_directory_path = os.path.join(
        pathlib.Path(__file__).parent.parent, ".data_storage/test_cleanup_actual"
    )
    cognee.config.data_root_directory(data_directory_path)

    cognee_directory_path = os.path.join(
        pathlib.Path(__file__).parent.parent, ".cognee_system/test_cleanup_actual"
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    user = await get_default_user()

    # Create dataset with multiple documents
    dataset_name = "test_cleanup_actual"

    docs = [
        "Document 1: Old document about mathematics",
        "Document 2: Old document about physics",
        "Document 3: Old document about chemistry",
        "Document 4: New document about biology",
        "Document 5: New document about geology",
    ]

    data_ids = []
    for doc in docs:
        add_result = await cognee.add([doc], dataset_name=dataset_name, user=user)
        data_ids.append(add_result.data_ingestion_info[0]["data_id"])

    cognify_result = await cognee.cognify([dataset_name], user=user)
    dataset_id = list(cognify_result.keys())[0]

    # Age the first 3 documents to be "old"
    db_engine = get_relational_engine()
    threshold_minutes = 30
    aged_timestamp = datetime.now(timezone.utc) - timedelta(minutes=threshold_minutes + 10)
    recent_timestamp = datetime.now(timezone.utc) - timedelta(minutes=10)

    async with db_engine.get_async_session() as session:
        # Age first 3 documents
        for i in range(3):
            stmt = update(Data).where(Data.id == data_ids[i]).values(last_accessed=aged_timestamp)
            await session.execute(stmt)

        # Set last 2 documents as recently accessed
        for i in range(3, 5):
            stmt = update(Data).where(Data.id == data_ids[i]).values(last_accessed=recent_timestamp)
            await session.execute(stmt)

        await session.commit()

    logger.info("Aged 3 documents to be old, kept 2 as recent")

    # Run actual cleanup
    logger.info("Running cleanup with dry_run=False...")
    result = await cleanup_unused_data(
        minutes_threshold=threshold_minutes, dry_run=False, user_id=user.id
    )

    logger.info(f"Cleanup result: {result}")

    # Assertions for actual deletion
    assert result["status"] == "completed", f"Status should be 'completed', got {result['status']}"
    assert result["deleted_count"]["documents"] == 3, (
        f"Should delete 3 documents, deleted {result['deleted_count']['documents']}"
    )

    # Verify only 2 documents remain
    async with db_engine.get_async_session() as session:
        result = await session.execute(select(Data).where(Data.id.in_(data_ids)))
        remaining_data = result.all()
        assert len(remaining_data) == 2, (
            f"Should have 2 remaining documents, found {len(remaining_data)}"
        )

        # Verify the remaining documents are the recent ones (data_ids[3] and data_ids[4])
        remaining_ids = [data[0].id for data in remaining_data]
        assert data_ids[3] in remaining_ids, "Document 4 should remain"
        assert data_ids[4] in remaining_ids, "Document 5 should remain"

        # Verify old documents are deleted
        assert data_ids[0] not in remaining_ids, "Document 1 should be deleted"
        assert data_ids[1] not in remaining_ids, "Document 2 should be deleted"
        assert data_ids[2] not in remaining_ids, "Document 3 should be deleted"

    logger.info("✅ test_cleanup_actual_deletion PASSED")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_last_accessed_updates_on_search())
    asyncio.run(test_cleanup_unused_data_dry_run())
    asyncio.run(test_cleanup_actual_deletion())
