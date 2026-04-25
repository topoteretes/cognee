"""Tests for the Dataset Queue system.

The Dataset Queue is a semaphore-based concurrency limiter that restricts
the number of parallel dataset operations (search, run_pipeline_per_dataset).

Configuration via environment variables:
- DATASET_QUEUE_ENABLED: Enable/disable the queue (default: False)
- DATABASE_MAX_LRU_CACHE_SIZE: Maximum concurrent dataset operations
  (default: 128, sourced from cognee.shared.lru_cache)
"""

import asyncio
import pytest
from unittest.mock import patch


# Module path for mocking the settings seam.
DATASET_QUEUE_MODULE = "cognee.infrastructure.databases.dataset_queue"
GET_DATASET_QUEUE_SETTINGS = f"{DATASET_QUEUE_MODULE}.queue.get_dataset_queue_settings"


class TestDatasetQueueInitialization:
    """Tests for DatasetQueue initialization and singleton behavior."""

    @pytest.fixture(autouse=True)
    def reset_queue_singleton(self):
        """Reset the singleton instance between tests."""
        try:
            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            dataset_queue._instance = None
        except (ImportError, AttributeError):
            pass
        yield

    def test_queue_initialization_when_enabled(self):
        """Test that queue initializes properly when enabled."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            assert queue._enabled is True
            assert queue._max_concurrent == 5

    def test_queue_initialization_when_disabled(self):
        """Test that queue initializes in disabled state correctly."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = False
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            assert queue._enabled is False

    def test_queue_is_singleton(self):
        """Test that the queue maintains singleton pattern."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 10

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue1 = dataset_queue()
            queue2 = dataset_queue()

            assert queue1 is queue2

    def test_queue_creates_semaphore_with_correct_size(self):
        """Test that internal semaphore has correct limit."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 3

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            # The semaphore should allow max_concurrent acquisitions
            assert queue._semaphore._value == 3


class TestDatasetQueueDisabledBehavior:
    """Tests for queue behavior when disabled."""

    @pytest.fixture(autouse=True)
    def reset_queue_singleton(self):
        """Reset the singleton instance between tests."""
        try:
            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            dataset_queue._instance = None
        except (ImportError, AttributeError):
            pass
        yield

    @pytest.mark.asyncio
    async def test_disabled_queue_allows_unlimited_concurrency(self):
        """Disabled queue's acquire() is a no-op and imposes no limit."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = False
            mock_settings.return_value.max_concurrent = 1  # would limit if enabled

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            concurrent_count = 0
            max_concurrent = 0

            async def operation(op_id: int):
                nonlocal concurrent_count, max_concurrent
                async with queue.acquire():
                    concurrent_count += 1
                    max_concurrent = max(max_concurrent, concurrent_count)
                    await asyncio.sleep(0.05)
                    concurrent_count -= 1
                    return op_id

            results = await asyncio.gather(*[operation(i) for i in range(10)])

            assert len(results) == 10
            # When disabled, all should run concurrently.
            assert max_concurrent == 10


class TestDatasetQueueContextManager:
    """Tests for async context manager usage."""

    @pytest.fixture(autouse=True)
    def reset_queue_singleton(self):
        """Reset the singleton instance between tests."""
        try:
            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            dataset_queue._instance = None
        except (ImportError, AttributeError):
            pass
        yield

    @pytest.mark.asyncio
    async def test_context_manager_acquires_and_releases(self):
        """Test that context manager properly acquires and releases slot."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 1

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            result = None
            async with queue.acquire():
                result = "inside"

            assert result == "inside"

            # Should be able to acquire again after release
            async with queue.acquire():
                result = "second"

            assert result == "second"

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self):
        """Test that context manager releases slot on exception."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 1

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            with pytest.raises(ValueError):
                async with queue.acquire():
                    raise ValueError("Test error")

            # Should be able to acquire after exception
            result = None
            async with queue.acquire():
                result = "after_exception"

            assert result == "after_exception"


class TestDatasetQueueIntegrationPatterns:
    """Tests demonstrating integration patterns with pipeline operations."""

    @pytest.fixture(autouse=True)
    def reset_queue_singleton(self):
        """Reset the singleton instance between tests."""
        try:
            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            dataset_queue._instance = None
        except (ImportError, AttributeError):
            pass
        yield

    @pytest.mark.asyncio
    async def test_pipeline_pattern_with_queue(self):
        """Test pattern for using queue with pipeline-style async generator."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            async def run_pipeline_per_dataset_mock(dataset_id: str):
                async with queue.acquire():
                    await asyncio.sleep(0.02)
                    yield {"status": "started", "dataset": dataset_id}
                    await asyncio.sleep(0.02)
                    yield {"status": "completed", "dataset": dataset_id}

            async def collect_results(dataset_id):
                results = []
                async for result in run_pipeline_per_dataset_mock(dataset_id):
                    results.append(result)
                return results

            pipeline_results = await asyncio.gather(
                collect_results("dataset1"),
                collect_results("dataset2"),
                collect_results("dataset3"),
            )

            assert len(pipeline_results) == 3
            for results in pipeline_results:
                assert len(results) == 2
                assert results[0]["status"] == "started"
                assert results[1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_mixed_operations_with_shared_queue(self):
        """Test that different operations share the same queue."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            concurrent_count = 0
            max_concurrent = 0

            async def search_op():
                nonlocal concurrent_count, max_concurrent
                async with queue.acquire():
                    concurrent_count += 1
                    max_concurrent = max(max_concurrent, concurrent_count)
                    await asyncio.sleep(0.03)
                    concurrent_count -= 1
                    return "search"

            async def pipeline_op():
                nonlocal concurrent_count, max_concurrent
                async with queue.acquire():
                    concurrent_count += 1
                    max_concurrent = max(max_concurrent, concurrent_count)
                    await asyncio.sleep(0.03)
                    concurrent_count -= 1
                    return "pipeline"

            results = await asyncio.gather(
                search_op(),
                pipeline_op(),
                search_op(),
                pipeline_op(),
            )

            assert len(results) == 4
            assert results.count("search") == 2
            assert results.count("pipeline") == 2
            assert max_concurrent <= 2


class TestDatasetQueueEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture(autouse=True)
    def reset_queue_singleton(self):
        """Reset the singleton instance between tests."""
        try:
            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            dataset_queue._instance = None
        except (ImportError, AttributeError):
            pass
        yield

    @pytest.mark.asyncio
    async def test_cancelled_task_releases_slot(self):
        """Test that cancelled tasks release their slots."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 1

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            async def long_operation():
                async with queue.acquire():
                    await asyncio.sleep(10.0)  # Long operation

            task = asyncio.create_task(long_operation())
            await asyncio.sleep(0.01)  # Let it acquire

            # Cancel the task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Slot should be released, new operation should work
            result = None
            async with queue.acquire():
                result = "success"

            assert result == "success"

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self):
        """Stress test with many concurrent operations."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            concurrent_count = 0
            max_concurrent = 0
            completed = 0

            async def operation(op_id: int):
                nonlocal concurrent_count, max_concurrent, completed
                async with queue.acquire():
                    concurrent_count += 1
                    max_concurrent = max(max_concurrent, concurrent_count)
                    await asyncio.sleep(0.01)
                    concurrent_count -= 1
                    completed += 1
                    return op_id

            # Run 50 operations with limit of 5
            results = await asyncio.gather(*[operation(i) for i in range(50)])

            assert len(results) == 50
            assert completed == 50
            assert max_concurrent <= 5

    @pytest.mark.asyncio
    async def test_reentrant_acquire_behavior(self):
        """Test behavior when same coroutine acquires multiple times."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            # Nested acquire - works because semaphore allows up to max_concurrent
            result = None
            async with queue.acquire():
                async with queue.acquire():
                    result = "nested"

            assert result == "nested"

    @pytest.mark.asyncio
    async def test_queue_works_with_sync_callbacks(self):
        """Test that queue works with sync callback wrapped operations."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            def sync_callback():
                return "sync_result"

            async def async_wrapper():
                async with queue.acquire():
                    return sync_callback()

            result = await async_wrapper()
            assert result == "sync_result"
