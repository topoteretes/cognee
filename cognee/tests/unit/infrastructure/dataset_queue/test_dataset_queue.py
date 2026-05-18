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


class TestReleaseSlotFor:
    """Tests for the async release_slot_for with optional on_last_release callback."""

    @pytest.fixture(autouse=True)
    def reset_queue_singleton(self):
        try:
            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            dataset_queue._instance = None
        except (ImportError, AttributeError):
            pass
        yield

    @pytest.mark.asyncio
    async def test_cleanup_fires_for_single_holder(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        ds = "dataset-A"
        cleaned = False

        async def cleanup():
            nonlocal cleaned
            cleaned = True

        await queue.ensure_slot(ds)
        await queue.release_slot_for(ds, on_last_release=cleanup)
        assert cleaned is True

    @pytest.mark.asyncio
    async def test_cleanup_skipped_for_nested_depth(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        ds = "dataset-B"
        call_count = 0

        async def cleanup():
            nonlocal call_count
            call_count += 1

        await queue.ensure_slot(ds)
        await queue.ensure_slot(ds)  # depth = 2

        await queue.release_slot_for(ds, on_last_release=cleanup)
        assert call_count == 0  # inner exit — skipped

        await queue.release_slot_for(ds, on_last_release=cleanup)
        assert call_count == 1  # outer exit — fires

    @pytest.mark.asyncio
    async def test_cleanup_skipped_when_cross_task_holder_exists(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        ds = "dataset-C"
        cleaned = False

        async def cleanup():
            nonlocal cleaned
            cleaned = True

        other_ready = asyncio.Event()
        check_done = asyncio.Event()

        async def other_task():
            await queue.ensure_slot(ds)
            other_ready.set()
            await check_done.wait()
            await queue.release_slot_for(ds)

        task = asyncio.create_task(other_task())
        await other_ready.wait()

        await queue.ensure_slot(ds)
        await queue.release_slot_for(ds, on_last_release=cleanup)
        assert cleaned is False  # other task still holds the dataset

        check_done.set()
        await task

    @pytest.mark.asyncio
    async def test_cleanup_fires_after_last_cross_task_holder_releases(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        ds = "dataset-D"
        call_count = 0

        async def cleanup():
            nonlocal call_count
            call_count += 1

        other_ready = asyncio.Event()
        main_released = asyncio.Event()

        async def other_task():
            await queue.ensure_slot(ds)
            other_ready.set()
            await main_released.wait()
            await queue.release_slot_for(ds, on_last_release=cleanup)

        task = asyncio.create_task(other_task())
        await other_ready.wait()

        await queue.ensure_slot(ds)
        await queue.release_slot_for(ds, on_last_release=cleanup)
        assert call_count == 0  # not last

        main_released.set()
        await task
        assert call_count == 1  # other task was last, cleanup fired

    @pytest.mark.asyncio
    async def test_different_dataset_does_not_block_cleanup(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        cleaned = False

        async def cleanup():
            nonlocal cleaned
            cleaned = True

        other_ready = asyncio.Event()
        check_done = asyncio.Event()

        async def other_task():
            await queue.ensure_slot("dataset-OTHER")
            other_ready.set()
            await check_done.wait()
            await queue.release_slot_for("dataset-OTHER")

        task = asyncio.create_task(other_task())
        await other_ready.wait()

        await queue.ensure_slot("dataset-OURS")
        await queue.release_slot_for("dataset-OURS", on_last_release=cleanup)
        assert cleaned is True  # different dataset — we're still last for ours

        check_done.set()
        await task

    @pytest.mark.asyncio
    async def test_disabled_queue_always_runs_cleanup(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=False, max_concurrent=5)
        cleaned = False

        async def cleanup():
            nonlocal cleaned
            cleaned = True

        await queue.release_slot_for("any-dataset", on_last_release=cleanup)
        assert cleaned is True

    @pytest.mark.asyncio
    async def test_cleanup_exception_still_releases_slot(self):
        """Slot must be freed even if the cleanup callback raises."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=1)
        ds = "dataset-E"

        async def failing_cleanup():
            raise ValueError("engine teardown failed")

        await queue.ensure_slot(ds)

        with pytest.raises(ValueError, match="engine teardown failed"):
            await queue.release_slot_for(ds, on_last_release=failing_cleanup)

        # Semaphore must have been released — acquiring again should not block.
        await queue.ensure_slot(ds)
        await queue.release_slot_for(ds)

    @pytest.mark.asyncio
    async def test_no_callback_behaves_as_plain_release(self):
        """Without on_last_release the method is a plain depth-aware release."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=1)
        ds = "dataset-F"

        await queue.ensure_slot(ds)
        await queue.release_slot_for(ds)

        # Slot freed — re-acquire should succeed immediately.
        await queue.ensure_slot(ds)
        await queue.release_slot_for(ds)

    @pytest.mark.asyncio
    async def test_release_without_ensure_is_noop(self):
        """Releasing a slot that was never acquired must not crash."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        cleaned = False

        async def cleanup():
            nonlocal cleaned
            cleaned = True

        # No ensure_slot — should silently no-op.
        await queue.release_slot_for("never-acquired", on_last_release=cleanup)
        assert cleaned is False
        assert queue._semaphore._value == 5  # nothing consumed

    @pytest.mark.asyncio
    async def test_double_release_is_idempotent(self):
        """Calling release twice for the same slot must not crash or over-release."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=2)
        ds = "dataset-G"

        await queue.ensure_slot(ds)
        assert queue._semaphore._value == 1  # one slot consumed

        await queue.release_slot_for(ds)
        assert queue._semaphore._value == 2  # back to full

        # Second release — entry already popped, should be a no-op.
        await queue.release_slot_for(ds)
        assert queue._semaphore._value == 2  # still full, not over-released

    @pytest.mark.asyncio
    async def test_semaphore_accounting_after_mixed_operations(self):
        """Semaphore value must be exactly right after acquires and releases."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=3)

        await queue.ensure_slot("ds1")
        await queue.ensure_slot("ds2")
        assert queue._semaphore._value == 1  # 2 consumed

        await queue.ensure_slot("ds1")  # re-entrant, no new acquire
        assert queue._semaphore._value == 1  # still 2 consumed

        await queue.release_slot_for("ds1")  # depth 2 → 1
        assert queue._semaphore._value == 1  # not freed yet

        await queue.release_slot_for("ds1")  # depth 1 → 0, freed
        assert queue._semaphore._value == 2

        await queue.release_slot_for("ds2")
        assert queue._semaphore._value == 3  # all back

    @pytest.mark.asyncio
    async def test_three_tasks_cleanup_fires_only_on_last(self):
        """With three tasks on the same dataset, cleanup fires once on the last exit."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        ds = "shared-ds"
        cleanup_count = 0

        async def cleanup():
            nonlocal cleanup_count
            cleanup_count += 1

        gate_1 = asyncio.Event()
        gate_2 = asyncio.Event()
        ready_1 = asyncio.Event()
        ready_2 = asyncio.Event()

        async def task_a():
            await queue.ensure_slot(ds)
            ready_1.set()
            await gate_1.wait()
            await queue.release_slot_for(ds, on_last_release=cleanup)

        async def task_b():
            await queue.ensure_slot(ds)
            ready_2.set()
            await gate_2.wait()
            await queue.release_slot_for(ds, on_last_release=cleanup)

        t1 = asyncio.create_task(task_a())
        t2 = asyncio.create_task(task_b())
        await ready_1.wait()
        await ready_2.wait()

        # Main task also holds the dataset.
        await queue.ensure_slot(ds)

        # Release from main — two others still hold it.
        await queue.release_slot_for(ds, on_last_release=cleanup)
        assert cleanup_count == 0

        # Release task_a — task_b still holds it.
        gate_1.set()
        await t1
        assert cleanup_count == 0

        # Release task_b — now last holder.
        gate_2.set()
        await t2
        assert cleanup_count == 1

    @pytest.mark.asyncio
    async def test_cleanup_fires_exactly_once_under_stress(self):
        """Many tasks on the same dataset; cleanup fires exactly once total."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        n_tasks = 20
        queue = DatasetQueue(enabled=True, max_concurrent=n_tasks + 1)
        ds = "stress-ds"
        cleanup_count = 0

        async def cleanup():
            nonlocal cleanup_count
            cleanup_count += 1

        # Manual barrier for Python 3.10 compat (asyncio.Barrier is 3.11+).
        arrived = 0
        all_arrived = asyncio.Event()

        async def worker():
            nonlocal arrived
            await queue.ensure_slot(ds)
            arrived += 1
            if arrived == n_tasks:
                all_arrived.set()
            else:
                await all_arrived.wait()
            await queue.release_slot_for(ds, on_last_release=cleanup)

        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(n_tasks)])
        assert cleanup_count == 1

    @pytest.mark.asyncio
    async def test_backstop_frees_slot_then_remaining_task_fires_cleanup(self):
        """Task-end backstop releases a crashed task's slot without cleanup.
        The surviving task should then be the last holder and fire cleanup."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        ds = "backstop-ds"
        cleaned = False

        async def cleanup():
            nonlocal cleaned
            cleaned = True

        crashed_ready = asyncio.Event()

        async def crashing_task():
            await queue.ensure_slot(ds)
            crashed_ready.set()
            raise RuntimeError("boom")

        task = asyncio.create_task(crashing_task())
        await crashed_ready.wait()

        # Let the task finish and its backstop fire.
        try:
            await task
        except RuntimeError:
            pass

        # Done-callbacks are scheduled via call_soon; yield to the event
        # loop so the backstop actually removes the crashed task's entry.
        await asyncio.sleep(0)

        # Now the main task acquires and is the only holder.
        await queue.ensure_slot(ds)
        await queue.release_slot_for(ds, on_last_release=cleanup)
        assert cleaned is True

    @pytest.mark.asyncio
    async def test_nested_depth_plus_cross_task(self):
        """Depth > 1 with another task holding — inner exit skips,
        outer exit skips (other task present), other task fires."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        ds = "combo-ds"
        cleanup_count = 0

        async def cleanup():
            nonlocal cleanup_count
            cleanup_count += 1

        other_ready = asyncio.Event()
        main_done = asyncio.Event()

        async def other_task():
            await queue.ensure_slot(ds)
            other_ready.set()
            await main_done.wait()
            await queue.release_slot_for(ds, on_last_release=cleanup)

        task = asyncio.create_task(other_task())
        await other_ready.wait()

        # Main task: depth = 2.
        await queue.ensure_slot(ds)
        await queue.ensure_slot(ds)

        await queue.release_slot_for(ds, on_last_release=cleanup)
        assert cleanup_count == 0  # depth 2 → 1, inner exit

        await queue.release_slot_for(ds, on_last_release=cleanup)
        assert cleanup_count == 0  # depth 0, but other task holds it

        main_done.set()
        await task
        assert cleanup_count == 1  # other task was last

    @pytest.mark.asyncio
    async def test_two_datasets_release_one_keeps_other(self):
        """Releasing one dataset doesn't affect a slot held for another."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        cleaned_a = False
        cleaned_b = False

        async def cleanup_a():
            nonlocal cleaned_a
            cleaned_a = True

        async def cleanup_b():
            nonlocal cleaned_b
            cleaned_b = True

        await queue.ensure_slot("ds-A")
        await queue.ensure_slot("ds-B")
        assert queue._semaphore._value == 3  # 2 consumed

        await queue.release_slot_for("ds-A", on_last_release=cleanup_a)
        assert cleaned_a is True
        assert queue._semaphore._value == 4  # 1 consumed (ds-B)

        # ds-B still held, should succeed.
        await queue.release_slot_for("ds-B", on_last_release=cleanup_b)
        assert cleaned_b is True
        assert queue._semaphore._value == 5  # all back

    @pytest.mark.asyncio
    async def test_none_dataset_id(self):
        """dataset_id=None uses the ds:<none> key and works correctly."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=2)
        cleaned = False

        async def cleanup():
            nonlocal cleaned
            cleaned = True

        await queue.ensure_slot(None)
        assert queue._semaphore._value == 1

        await queue.release_slot_for(None, on_last_release=cleanup)
        assert cleaned is True
        assert queue._semaphore._value == 2

    @pytest.mark.asyncio
    async def test_cleanup_exception_does_not_affect_other_datasets(self):
        """If cleanup raises for one dataset, another dataset's slot is unaffected."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)

        async def failing_cleanup():
            raise RuntimeError("kaboom")

        await queue.ensure_slot("ds-ok")
        await queue.ensure_slot("ds-fail")

        with pytest.raises(RuntimeError, match="kaboom"):
            await queue.release_slot_for("ds-fail", on_last_release=failing_cleanup)

        # ds-ok is still held and can be released normally.
        cleaned = False

        async def ok_cleanup():
            nonlocal cleaned
            cleaned = True

        await queue.release_slot_for("ds-ok", on_last_release=ok_cleanup)
        assert cleaned is True
        assert queue._semaphore._value == 5
