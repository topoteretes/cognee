"""Performance tests for cleanup_unused_data functionality."""
import pytest
import time
import asyncio
from datetime import datetime, timedelta
from cognee.tasks.cleanup_unused_data import cleanup_unused_data
from cognee.modules.data.access_tracking import AccessTracker
from cognee.infrastructure.databases.relational import get_relational_engine


class TestCleanupPerformance:
    """Performance test suite for cleanup operations."""

    @pytest.fixture
    async def large_dataset_db(self):
        """Create a test database with large number of records."""
        engine = await get_relational_engine()
        
        # Create 1000 datasets
        async with engine.begin() as conn:
            for i in range(1000):
                await conn.execute(
                    "INSERT INTO datasets (id, name, created_at, last_modified) VALUES (?, ?, ?, ?)",
                    (f'perf_dataset_{i}', f'Performance Dataset {i}',
                     datetime.now() - timedelta(days=60),
                     datetime.now() - timedelta(days=60))
                )
        
        yield engine
        
        # Cleanup
        async with engine.begin() as conn:
            await conn.execute("DELETE FROM datasets WHERE id LIKE 'perf_dataset_%'")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cleanup_performance_1000_records(self, large_dataset_db):
        """Test cleanup performance with 1000 old datasets."""
        start_time = time.time()
        
        result = await cleanup_unused_data(days_threshold=30)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within 60 seconds for 1000 records
        assert duration < 60, f"Cleanup took {duration:.2f}s, expected < 60s"
        assert result['status'] == 'success'
        assert result['datasets_deleted'] >= 1000
        
        # Log performance metrics
        print(f"\nPerformance metrics for 1000 records:")
        print(f"  Total time: {duration:.2f}s")
        print(f"  Records deleted: {result['datasets_deleted']}")
        print(f"  Rate: {result['datasets_deleted'] / duration:.2f} records/second")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cleanup_performance_with_access_checking(self, large_dataset_db):
        """Test cleanup performance when access checking is enabled."""
        # Mark half of the datasets as recently accessed
        tracker = AccessTracker()
        for i in range(500):
            await tracker.mark_accessed(f'perf_dataset_{i}', 'dataset')
        
        start_time = time.time()
        
        result = await cleanup_unused_data(days_threshold=30)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within 90 seconds with access checking
        assert duration < 90, f"Cleanup with access checking took {duration:.2f}s, expected < 90s"
        assert result['status'] == 'success'
        
        # Should only delete datasets that weren't recently accessed (~500)
        assert 450 <= result['datasets_deleted'] <= 550
        
        print(f"\nPerformance metrics with access checking:")
        print(f"  Total time: {duration:.2f}s")
        print(f"  Records checked: 1000")
        print(f"  Records deleted: {result['datasets_deleted']}")
        print(f"  Rate: {1000 / duration:.2f} records checked/second")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_dry_run_performance(self, large_dataset_db):
        """Test dry-run mode performance."""
        start_time = time.time()
        
        result = await cleanup_unused_data(days_threshold=30, dry_run=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Dry-run should be faster as it doesn't actually delete
        assert duration < 30, f"Dry-run took {duration:.2f}s, expected < 30s"
        assert result['datasets_to_delete'] >= 1000
        
        print(f"\nDry-run performance metrics:")
        print(f"  Total time: {duration:.2f}s")
        print(f"  Records scanned: {result['datasets_to_delete']}")
        print(f"  Rate: {result['datasets_to_delete'] / duration:.2f} records/second")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_cleanup_performance(self):
        """Test performance of concurrent cleanup operations."""
        start_time = time.time()
        
        # Run 5 dry-run cleanup operations concurrently
        tasks = [cleanup_unused_data(days_threshold=30, dry_run=True) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time even with concurrent operations
        assert duration < 120, f"Concurrent operations took {duration:.2f}s, expected < 120s"
        assert all(isinstance(r, dict) for r in results)
        
        print(f"\nConcurrent operation performance:")
        print(f"  Total time for 5 concurrent operations: {duration:.2f}s")
        print(f"  Average time per operation: {duration / 5:.2f}s")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_efficiency(self, large_dataset_db):
        """Test memory efficiency during cleanup operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        result = await cleanup_unused_data(days_threshold=30)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for 1000 records)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB, expected < 100MB"
        assert result['status'] == 'success'
        
        print(f"\nMemory efficiency metrics:")
        print(f"  Initial memory: {initial_memory:.2f}MB")
        print(f"  Final memory: {final_memory:.2f}MB")
        print(f"  Memory increase: {memory_increase:.2f}MB")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_batch_processing_performance(self, large_dataset_db):
        """Test performance of batch processing vs individual deletions."""
        # Test with batch size configuration
        start_time = time.time()
        
        result = await cleanup_unused_data(days_threshold=30, batch_size=100)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Batch processing should be efficient
        assert duration < 45, f"Batch processing took {duration:.2f}s, expected < 45s"
        assert result['status'] == 'success'
        
        print(f"\nBatch processing performance:")
        print(f"  Total time: {duration:.2f}s")
        print(f"  Batch size: 100")
        print(f"  Records deleted: {result['datasets_deleted']}")
        print(f"  Batches processed: {result['datasets_deleted'] // 100}")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_access_tracker_performance(self):
        """Test performance of access tracking operations."""
        tracker = AccessTracker()
        
        # Test marking 1000 resources as accessed
        start_time = time.time()
        
        for i in range(1000):
            await tracker.mark_accessed(f'resource_{i}', 'dataset')
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within 30 seconds
        assert duration < 30, f"Marking 1000 accesses took {duration:.2f}s, expected < 30s"
        
        print(f"\nAccess tracker performance:")
        print(f"  Total time to mark 1000 accesses: {duration:.2f}s")
        print(f"  Rate: {1000 / duration:.2f} marks/second")
        
        # Test getting last access for 1000 resources
        start_time = time.time()
        
        for i in range(1000):
            await tracker.get_last_access(f'resource_{i}', 'dataset')
        
        end_time = time.time()
        query_duration = end_time - start_time
        
        # Should complete within 20 seconds
        assert query_duration < 20, f"Querying 1000 accesses took {query_duration:.2f}s, expected < 20s"
        
        print(f"  Total time to query 1000 accesses: {query_duration:.2f}s")
        print(f"  Rate: {1000 / query_duration:.2f} queries/second")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_scalability_10k_records(self):
        """Test scalability with 10,000 records."""
        engine = await get_relational_engine()
        
        # Create 10,000 datasets
        print("\nCreating 10,000 test datasets...")
        async with engine.begin() as conn:
            for i in range(10000):
                await conn.execute(
                    "INSERT INTO datasets (id, name, created_at, last_modified) VALUES (?, ?, ?, ?)",
                    (f'scale_test_{i}', f'Scale Test {i}',
                     datetime.now() - timedelta(days=60),
                     datetime.now() - timedelta(days=60))
                )
        
        start_time = time.time()
        
        result = await cleanup_unused_data(days_threshold=30)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should scale reasonably (< 10 minutes for 10k records)
        assert duration < 600, f"Cleanup of 10k records took {duration:.2f}s, expected < 600s"
        assert result['status'] == 'success'
        assert result['datasets_deleted'] >= 10000
        
        print(f"\nScalability test (10k records):")
        print(f"  Total time: {duration:.2f}s ({duration / 60:.2f} minutes)")
        print(f"  Records deleted: {result['datasets_deleted']}")
        print(f"  Rate: {result['datasets_deleted'] / duration:.2f} records/second")
        
        # Cleanup
        async with engine.begin() as conn:
            await conn.execute("DELETE FROM datasets WHERE id LIKE 'scale_test_%'")
