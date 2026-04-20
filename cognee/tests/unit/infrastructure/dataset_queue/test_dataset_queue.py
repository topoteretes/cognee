"""Tests for the Dataset Queue system.

The Dataset Queue is a semaphore-based concurrency limiter that restricts
the number of parallel dataset operations (search, run_pipeline_per_dataset).

Configuration via environment variables:
- DATASET_QUEUE_ENABLED: Enable/disable the queue (default: False)
- DATABASE_MAX_LRU_CACHE_SIZE: Maximum concurrent dataset operations (default: 128, sourced from cognee.shared.lru_cache)
"""

import asyncio
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# Module paths for mocking - these will need to be updated once implementation exists
DATASET_QUEUE_MODULE = "cognee.infrastructure.databases.dataset_queue"
GET_DATASET_QUEUE_SETTINGS = f"{DATASET_QUEUE_MODULE}.queue.get_dataset_queue_settings"
DATASET_QUEUE_SINGLETON = f"{DATASET_QUEUE_MODULE}.dataset_queue"


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


class TestDatasetQueueConcurrencyLimiting:
    """Tests for the core concurrency limiting behavior."""

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
    async def test_single_request_executes_immediately(self):
        """Test that a single request executes without waiting."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            executed = False

            async def operation():
                nonlocal executed
                executed = True
                return "result"

            result = await queue.execute(operation)

            assert executed is True
            assert result == "result"

    @pytest.mark.asyncio
    async def test_requests_within_limit_execute_concurrently(self):
        """Test that requests within the limit execute in parallel."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 3

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            execution_times = []

            async def operation(op_id: int):
                start = time.monotonic()
                await asyncio.sleep(0.1)
                end = time.monotonic()
                execution_times.append((op_id, start, end))
                return f"result_{op_id}"

            # Run 3 operations concurrently (within limit)
            results = await asyncio.gather(
                queue.execute(lambda: operation(1)),
                queue.execute(lambda: operation(2)),
                queue.execute(lambda: operation(3)),
            )

            assert len(results) == 3
            assert all(r.startswith("result_") for r in results)

            # All operations should have started nearly simultaneously
            start_times = [t[1] for t in execution_times]
            max_start_diff = max(start_times) - min(start_times)
            assert max_start_diff < 0.05  # Within 50ms of each other

    @pytest.mark.asyncio
    async def test_requests_exceeding_limit_wait(self):
        """Test that requests exceeding the limit wait for available slots."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            execution_order = []
            start_event = asyncio.Event()

            async def operation(op_id: int, delay: float = 0.1):
                execution_order.append(f"start_{op_id}")
                if op_id <= 2:
                    start_event.set()
                await asyncio.sleep(delay)
                execution_order.append(f"end_{op_id}")
                return f"result_{op_id}"

            # Start 4 operations with limit of 2
            # First 2 should start immediately, 3rd and 4th should wait
            task1 = asyncio.create_task(queue.execute(lambda: operation(1)))
            task2 = asyncio.create_task(queue.execute(lambda: operation(2)))
            await start_event.wait()  # Wait for first two to start
            await asyncio.sleep(0.01)  # Small delay to ensure order

            task3 = asyncio.create_task(queue.execute(lambda: operation(3)))
            task4 = asyncio.create_task(queue.execute(lambda: operation(4)))

            results = await asyncio.gather(task1, task2, task3, task4)

            assert len(results) == 4
            # Operations 1 and 2 should have started before 3 and 4
            assert execution_order.index("start_1") < execution_order.index("start_3")
            assert execution_order.index("start_2") < execution_order.index("start_4")

    @pytest.mark.asyncio
    async def test_queue_releases_slot_after_completion(self):
        """Test that completed operations release their slots."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 1

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            results = []

            async def operation(op_id: int):
                await asyncio.sleep(0.05)
                results.append(op_id)
                return op_id

            # With limit of 1, operations must run sequentially
            await asyncio.gather(
                queue.execute(lambda: operation(1)),
                queue.execute(lambda: operation(2)),
                queue.execute(lambda: operation(3)),
            )

            # All should complete
            assert len(results) == 3
            assert sorted(results) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_queue_releases_slot_on_exception(self):
        """Test that slots are released even when operations fail."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 1

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            async def failing_operation():
                await asyncio.sleep(0.01)
                raise ValueError("Test error")

            async def success_operation():
                return "success"

            # First operation fails
            with pytest.raises(ValueError, match="Test error"):
                await queue.execute(failing_operation)

            # Second operation should still be able to acquire the slot
            result = await queue.execute(success_operation)
            assert result == "success"


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
        """Test that disabled queue doesn't limit concurrency."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = False
            mock_settings.return_value.max_concurrent = 1  # Would limit if enabled

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            concurrent_count = 0
            max_concurrent = 0

            async def operation(op_id: int):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.05)
                concurrent_count -= 1
                return op_id

            # Run 10 operations "concurrently"
            results = await asyncio.gather(
                *[queue.execute(lambda i=i: operation(i)) for i in range(10)]
            )

            assert len(results) == 10
            # When disabled, all should run at once
            assert max_concurrent == 10

    @pytest.mark.asyncio
    async def test_disabled_queue_executes_operation_directly(self):
        """Test that disabled queue executes operations without queuing."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = False
            mock_settings.return_value.max_concurrent = 1

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            async def operation():
                return "direct_result"

            result = await queue.execute(operation)
            assert result == "direct_result"


class TestDatasetQueueDecorator:
    """Tests for the @dataset_queue_limit decorator."""

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
    async def test_decorator_wraps_async_function(self):
        """Test that decorator works with async functions."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue_limit

            @dataset_queue_limit
            async def test_function(arg1, arg2=None):
                return f"{arg1}_{arg2}"

            result = await test_function("hello", arg2="world")
            assert result == "hello_world"

    @pytest.mark.asyncio
    async def test_decorator_respects_queue_limit(self):
        """Test that decorated functions respect concurrency limit."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue_limit

            concurrent_count = 0
            max_concurrent = 0

            @dataset_queue_limit
            async def limited_operation(op_id: int):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.05)
                concurrent_count -= 1
                return op_id

            results = await asyncio.gather(*[limited_operation(i) for i in range(5)])

            assert len(results) == 5
            assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue_limit

            @dataset_queue_limit
            async def my_documented_function():
                """This is the docstring."""
                return "result"

            assert my_documented_function.__name__ == "my_documented_function"
            assert "This is the docstring" in my_documented_function.__doc__

    @pytest.mark.asyncio
    async def test_decorator_propagates_exceptions(self):
        """Test that decorator properly propagates exceptions."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue_limit

            @dataset_queue_limit
            async def failing_function():
                raise RuntimeError("Expected error")

            with pytest.raises(RuntimeError, match="Expected error"):
                await failing_function()


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


class TestDatasetQueueStatus:
    """Tests for queue status reporting."""

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
    async def test_available_slots_reporting(self):
        """Test that queue reports available slots correctly."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 3

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            assert queue.available_slots == 3

            hold_event = asyncio.Event()
            release_event = asyncio.Event()

            async def hold_slot():
                async with queue.acquire():
                    hold_event.set()
                    await release_event.wait()

            task = asyncio.create_task(hold_slot())
            await hold_event.wait()

            assert queue.available_slots == 2

            release_event.set()
            await task

            assert queue.available_slots == 3

    def test_queue_max_concurrent_property(self):
        """Test that max_concurrent property returns configured value."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 7

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            assert queue.max_concurrent == 7

    def test_queue_enabled_property(self):
        """Test that enabled property returns current state."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            assert queue.enabled is True


class TestDatasetQueueIntegrationPatterns:
    """Tests demonstrating integration patterns with search and pipeline."""

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
    async def test_search_pattern_with_queue(self):
        """Test pattern for using queue with search operations."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue_limit

            # Simulated search function with queue limiting
            @dataset_queue_limit
            async def mock_search(query_text: str, dataset_ids=None):
                await asyncio.sleep(0.02)  # Simulate DB access
                return [{"result": query_text}]

            # Multiple concurrent searches
            results = await asyncio.gather(
                mock_search("query1"),
                mock_search("query2"),
                mock_search("query3"),
            )

            assert len(results) == 3
            assert results[0] == [{"result": "query1"}]
            assert results[1] == [{"result": "query2"}]
            assert results[2] == [{"result": "query3"}]

    @pytest.mark.asyncio
    async def test_pipeline_pattern_with_queue(self):
        """Test pattern for using queue with pipeline operations."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            # Simulated pipeline generator with queue limiting
            async def run_pipeline_per_dataset_mock(dataset_id: str):
                async with queue.acquire():
                    await asyncio.sleep(0.02)  # Simulate processing
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
        """Test that search and pipeline operations share the same queue."""
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


class TestDatasetQueueTimeout:
    """Tests for queue timeout behavior (optional feature)."""

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
    async def test_execute_with_timeout_succeeds_within_limit(self):
        """Test that operations completing within timeout succeed."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 1
            mock_settings.return_value.dataset_queue_timeout = 5.0

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            async def quick_operation():
                await asyncio.sleep(0.01)
                return "quick_result"

            result = await queue.execute(quick_operation, timeout=1.0)
            assert result == "quick_result"

    @pytest.mark.asyncio
    async def test_acquire_with_timeout_raises_on_timeout(self):
        """Test that timeout raises appropriate exception when exceeded."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 1

            from cognee.infrastructure.databases.dataset_queue import (
                dataset_queue,
                DatasetQueueTimeoutError,
            )

            queue = dataset_queue()

            # Hold the only slot
            async def hold_slot():
                async with queue.acquire():
                    await asyncio.sleep(1.0)

            hold_task = asyncio.create_task(hold_slot())
            await asyncio.sleep(0.01)  # Let it acquire

            # Try to acquire with short timeout
            with pytest.raises(DatasetQueueTimeoutError):
                await queue.acquire_with_timeout(timeout=0.05)

            hold_task.cancel()
            try:
                await hold_task
            except asyncio.CancelledError:
                pass


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
        """Test behavior when same coroutine tries to acquire multiple times."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            # This tests nested acquire - should work since semaphore allows it
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
