"""Integration tests for cleanup_unused_data with real database interactions."""
import pytest
import asyncio
from datetime import datetime, timedelta
from cognee.tasks.cleanup_unused_data import cleanup_unused_data
from cognee.modules.data.access_tracking import AccessTracker
from cognee.infrastructure.databases.relational import get_relational_engine


@pytest.fixture
async def test_db():
    """Set up test database with sample data."""
    # Initialize test database connection
    engine = await get_relational_engine()
    
    # Create test data
    test_datasets = [
        {
            'id': 'test_dataset_1',
            'name': 'Old Test Dataset 1',
            'created_at': datetime.now() - timedelta(days=60),
            'last_modified': datetime.now() - timedelta(days=60)
        },
        {
            'id': 'test_dataset_2',
            'name': 'Recent Test Dataset',
            'created_at': datetime.now() - timedelta(days=5),
            'last_modified': datetime.now() - timedelta(days=5)
        },
        {
            'id': 'test_dataset_3',
            'name': 'Old But Accessed Dataset',
            'created_at': datetime.now() - timedelta(days=60),
            'last_modified': datetime.now() - timedelta(days=60)
        }
    ]
    
    # Insert test data
    async with engine.begin() as conn:
        for dataset in test_datasets:
            await conn.execute(
                "INSERT INTO datasets (id, name, created_at, last_modified) VALUES (?, ?, ?, ?)",
                (dataset['id'], dataset['name'], dataset['created_at'], dataset['last_modified'])
            )
    
    yield engine
    
    # Cleanup test data
    async with engine.begin() as conn:
        await conn.execute("DELETE FROM datasets WHERE id LIKE 'test_dataset_%'")
        await conn.execute("DELETE FROM access_logs WHERE resource_id LIKE 'test_dataset_%'")


class TestCleanupIntegration:
    """Integration test suite for cleanup functionality."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_cleanup_with_database(self, test_db):
        """Test complete cleanup workflow with real database."""
        # Run cleanup with 30 day threshold
        result = await cleanup_unused_data(days_threshold=30)
        
        assert result['status'] == 'success'
        assert result['datasets_deleted'] >= 1  # Should delete test_dataset_1
        assert 'test_dataset_2' not in [d['id'] for d in result.get('deleted_datasets', [])]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_access_tracking_prevents_deletion(self, test_db):
        """Test that access tracking prevents deletion of recently accessed old data."""
        # Mark old dataset as recently accessed
        tracker = AccessTracker()
        await tracker.mark_accessed('test_dataset_3', 'dataset')
        
        # Run cleanup
        result = await cleanup_unused_data(days_threshold=30)
        
        # Verify old but accessed dataset was not deleted
        deleted_ids = [d['id'] for d in result.get('deleted_datasets', [])]
        assert 'test_dataset_3' not in deleted_ids

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cleanup_with_related_documents(self, test_db):
        """Test cleanup properly handles documents related to datasets."""
        # Create dataset with documents
        async with test_db.begin() as conn:
            await conn.execute(
                "INSERT INTO datasets (id, name, created_at, last_modified) VALUES (?, ?, ?, ?)",
                ('test_dataset_4', 'Dataset with Docs', datetime.now() - timedelta(days=60),
                 datetime.now() - timedelta(days=60))
            )
            await conn.execute(
                "INSERT INTO documents (id, dataset_id, name, created_at) VALUES (?, ?, ?, ?)",
                ('test_doc_1', 'test_dataset_4', 'Test Document', datetime.now() - timedelta(days=60))
            )
        
        # Run cleanup
        result = await cleanup_unused_data(days_threshold=30)
        
        # Verify both dataset and its documents were deleted
        assert result['datasets_deleted'] >= 1
        assert result['documents_deleted'] >= 1
        
        # Verify data is actually deleted from database
        async with test_db.begin() as conn:
            dataset_result = await conn.execute(
                "SELECT * FROM datasets WHERE id = 'test_dataset_4'"
            )
            assert len(dataset_result.fetchall()) == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cleanup_rollback_on_error(self, test_db):
        """Test that cleanup rolls back changes on error."""
        # Get initial count of datasets
        async with test_db.begin() as conn:
            result = await conn.execute("SELECT COUNT(*) FROM datasets WHERE id LIKE 'test_dataset_%'")
            initial_count = result.scalar()
        
        # Try cleanup with invalid parameters (should fail gracefully)
        try:
            result = await cleanup_unused_data(days_threshold=-1)
        except ValueError:
            pass
        
        # Verify data is unchanged
        async with test_db.begin() as conn:
            result = await conn.execute("SELECT COUNT(*) FROM datasets WHERE id LIKE 'test_dataset_%'")
            final_count = result.scalar()
        
        assert initial_count == final_count

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_cleanup_operations(self, test_db):
        """Test that multiple cleanup operations can run safely."""
        # Run two cleanup operations concurrently
        results = await asyncio.gather(
            cleanup_unused_data(days_threshold=30, dry_run=True),
            cleanup_unused_data(days_threshold=30, dry_run=True),
            return_exceptions=True
        )
        
        # Both should succeed without errors
        assert all(isinstance(r, dict) for r in results)
        assert all(r['status'] == 'success' for r in results if isinstance(r, dict))

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cleanup_performance_with_large_dataset(self, test_db):
        """Test cleanup performance with many records."""
        # Create 100 old datasets
        async with test_db.begin() as conn:
            for i in range(100):
                await conn.execute(
                    "INSERT INTO datasets (id, name, created_at, last_modified) VALUES (?, ?, ?, ?)",
                    (f'perf_test_{i}', f'Performance Test {i}',
                     datetime.now() - timedelta(days=60),
                     datetime.now() - timedelta(days=60))
                )
        
        # Measure cleanup time
        start_time = datetime.now()
        result = await cleanup_unused_data(days_threshold=30)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Cleanup should complete within reasonable time (< 30 seconds for 100 records)
        assert duration < 30
        assert result['status'] == 'success'
        assert result['datasets_deleted'] >= 100

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_dry_run_does_not_modify_database(self, test_db):
        """Test that dry-run mode doesn't actually modify the database."""
        # Get initial count
        async with test_db.begin() as conn:
            result = await conn.execute("SELECT COUNT(*) FROM datasets WHERE id LIKE 'test_dataset_%'")
            initial_count = result.scalar()
        
        # Run cleanup in dry-run mode
        result = await cleanup_unused_data(days_threshold=30, dry_run=True)
        
        # Verify database is unchanged
        async with test_db.begin() as conn:
            result_after = await conn.execute("SELECT COUNT(*) FROM datasets WHERE id LIKE 'test_dataset_%'")
            final_count = result_after.scalar()
        
        assert initial_count == final_count
        assert result['datasets_to_delete'] >= 1
