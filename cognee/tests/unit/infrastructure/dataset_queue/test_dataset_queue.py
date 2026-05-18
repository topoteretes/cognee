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
    """Tests for release_slot_for — verifies that _teardown_subprocess_engines
    fires at the right time (last holder) and that the semaphore is always
    released regardless of teardown outcome."""

    @pytest.fixture(autouse=True)
    def reset_queue_singleton(self):
        try:
            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            dataset_queue._instance = None
        except (ImportError, AttributeError):
            pass
        yield

    @staticmethod
    def _mock_teardown(queue):
        """Replace _teardown_subprocess_engines with a counter."""
        call_count = 0

        async def fake_teardown():
            nonlocal call_count
            call_count += 1

        queue._teardown_subprocess_engines = fake_teardown

        class Counter:
            @property
            def value(self):
                return call_count

        return Counter()

    @pytest.mark.asyncio
    async def test_teardown_fires_for_single_holder(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_teardown(queue)

        await queue.ensure_slot("ds-A")
        await queue.release_slot_for("ds-A")
        assert counter.value == 1

    @pytest.mark.asyncio
    async def test_teardown_skipped_for_nested_depth(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_teardown(queue)

        await queue.ensure_slot("ds-B")
        await queue.ensure_slot("ds-B")  # depth = 2

        await queue.release_slot_for("ds-B")
        assert counter.value == 0  # inner exit — skipped

        await queue.release_slot_for("ds-B")
        assert counter.value == 1  # outer exit — fires

    @pytest.mark.asyncio
    async def test_teardown_skipped_when_cross_task_holder_exists(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_teardown(queue)
        ds = "ds-C"

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
        await queue.release_slot_for(ds)
        assert counter.value == 0  # other task still holds the dataset

        check_done.set()
        await task
        assert counter.value == 1  # other task was last

    @pytest.mark.asyncio
    async def test_teardown_fires_after_last_cross_task_holder_releases(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_teardown(queue)
        ds = "ds-D"

        other_ready = asyncio.Event()
        main_released = asyncio.Event()

        async def other_task():
            await queue.ensure_slot(ds)
            other_ready.set()
            await main_released.wait()
            await queue.release_slot_for(ds)

        task = asyncio.create_task(other_task())
        await other_ready.wait()

        await queue.ensure_slot(ds)
        await queue.release_slot_for(ds)
        assert counter.value == 0  # not last

        main_released.set()
        await task
        assert counter.value == 1  # other task was last, teardown fired

    @pytest.mark.asyncio
    async def test_different_dataset_does_not_block_teardown(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_teardown(queue)

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
        await queue.release_slot_for("dataset-OURS")
        assert counter.value == 1  # different dataset — we're last for ours

        check_done.set()
        await task

    @pytest.mark.asyncio
    async def test_disabled_queue_skips_teardown(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=False, max_concurrent=5)
        counter = self._mock_teardown(queue)

        await queue.release_slot_for("any-dataset")
        assert counter.value == 0

    @pytest.mark.asyncio
    async def test_teardown_exception_still_releases_slot(self):
        """Slot must be freed even if _teardown_subprocess_engines raises."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=1)
        ds = "ds-E"

        async def failing_teardown():
            raise ValueError("engine teardown failed")

        queue._teardown_subprocess_engines = failing_teardown

        await queue.ensure_slot(ds)

        with pytest.raises(ValueError, match="engine teardown failed"):
            await queue.release_slot_for(ds)

        # Replace the failing mock so the verification calls below don't blow up.
        self._mock_teardown(queue)

        # Semaphore must have been released — acquiring again should not block.
        await queue.ensure_slot(ds)
        await queue.release_slot_for(ds)

    @pytest.mark.asyncio
    async def test_release_without_ensure_is_noop(self):
        """Releasing a slot that was never acquired must not crash."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_teardown(queue)

        await queue.release_slot_for("never-acquired")
        assert counter.value == 0  # no entry — teardown not called
        assert queue._semaphore._value == 5  # nothing consumed

    @pytest.mark.asyncio
    async def test_double_release_is_idempotent(self):
        """Calling release twice for the same slot must not crash or over-release."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=2)
        counter = self._mock_teardown(queue)
        ds = "ds-G"

        await queue.ensure_slot(ds)
        assert queue._semaphore._value == 1

        await queue.release_slot_for(ds)
        assert queue._semaphore._value == 2
        assert counter.value == 1

        # Second release — entry already popped, should be a no-op.
        await queue.release_slot_for(ds)
        assert queue._semaphore._value == 2  # not over-released
        assert counter.value == 1  # not called again

    @pytest.mark.asyncio
    async def test_semaphore_accounting_after_mixed_operations(self):
        """Semaphore value must be exactly right after acquires and releases."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=3)
        self._mock_teardown(queue)

        await queue.ensure_slot("ds1")
        await queue.ensure_slot("ds2")
        assert queue._semaphore._value == 1  # 2 consumed

        await queue.ensure_slot("ds1")  # re-entrant, no new acquire
        assert queue._semaphore._value == 1

        await queue.release_slot_for("ds1")  # depth 2 → 1
        assert queue._semaphore._value == 1  # not freed yet

        await queue.release_slot_for("ds1")  # depth 1 → 0, freed
        assert queue._semaphore._value == 2

        await queue.release_slot_for("ds2")
        assert queue._semaphore._value == 3  # all back

    @pytest.mark.asyncio
    async def test_three_tasks_teardown_fires_only_on_last(self):
        """With three tasks on the same dataset, teardown fires once on the last exit."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_teardown(queue)
        ds = "shared-ds"

        gate_1 = asyncio.Event()
        gate_2 = asyncio.Event()
        ready_1 = asyncio.Event()
        ready_2 = asyncio.Event()

        async def task_a():
            await queue.ensure_slot(ds)
            ready_1.set()
            await gate_1.wait()
            await queue.release_slot_for(ds)

        async def task_b():
            await queue.ensure_slot(ds)
            ready_2.set()
            await gate_2.wait()
            await queue.release_slot_for(ds)

        t1 = asyncio.create_task(task_a())
        t2 = asyncio.create_task(task_b())
        await ready_1.wait()
        await ready_2.wait()

        await queue.ensure_slot(ds)

        await queue.release_slot_for(ds)
        assert counter.value == 0

        gate_1.set()
        await t1
        assert counter.value == 0

        gate_2.set()
        await t2
        assert counter.value == 1

    @pytest.mark.asyncio
    async def test_teardown_fires_exactly_once_under_stress(self):
        """Many tasks on the same dataset; teardown fires exactly once total."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        n_tasks = 20
        queue = DatasetQueue(enabled=True, max_concurrent=n_tasks + 1)
        counter = self._mock_teardown(queue)
        ds = "stress-ds"

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
            await queue.release_slot_for(ds)

        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(n_tasks)])
        assert counter.value == 1

    @pytest.mark.asyncio
    async def test_backstop_frees_slot_then_remaining_task_fires_teardown(self):
        """Task-end backstop releases a crashed task's slot without teardown.
        The surviving task should then be the last holder and fire teardown."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_teardown(queue)
        ds = "backstop-ds"

        crashed_ready = asyncio.Event()

        async def crashing_task():
            await queue.ensure_slot(ds)
            crashed_ready.set()
            raise RuntimeError("boom")

        task = asyncio.create_task(crashing_task())
        await crashed_ready.wait()

        try:
            await task
        except RuntimeError:
            pass

        # Done-callbacks are scheduled via call_soon; yield so the backstop fires.
        await asyncio.sleep(0)

        await queue.ensure_slot(ds)
        await queue.release_slot_for(ds)
        assert counter.value == 1

    @pytest.mark.asyncio
    async def test_nested_depth_plus_cross_task(self):
        """Depth > 1 with another task holding — inner exit skips,
        outer exit skips (other task present), other task fires."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_teardown(queue)
        ds = "combo-ds"

        other_ready = asyncio.Event()
        main_done = asyncio.Event()

        async def other_task():
            await queue.ensure_slot(ds)
            other_ready.set()
            await main_done.wait()
            await queue.release_slot_for(ds)

        task = asyncio.create_task(other_task())
        await other_ready.wait()

        await queue.ensure_slot(ds)
        await queue.ensure_slot(ds)  # depth = 2

        await queue.release_slot_for(ds)
        assert counter.value == 0  # depth 2 → 1

        await queue.release_slot_for(ds)
        assert counter.value == 0  # depth 0, but other task holds it

        main_done.set()
        await task
        assert counter.value == 1  # other task was last

    @pytest.mark.asyncio
    async def test_two_datasets_release_one_keeps_other(self):
        """Releasing one dataset doesn't affect a slot held for another."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_teardown(queue)

        await queue.ensure_slot("ds-A")
        await queue.ensure_slot("ds-B")
        assert queue._semaphore._value == 3

        await queue.release_slot_for("ds-A")
        assert counter.value == 1
        assert queue._semaphore._value == 4

        await queue.release_slot_for("ds-B")
        assert counter.value == 2
        assert queue._semaphore._value == 5

    @pytest.mark.asyncio
    async def test_none_dataset_id(self):
        """dataset_id=None uses the ds:<none> key and works correctly."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=2)
        counter = self._mock_teardown(queue)

        await queue.ensure_slot(None)
        assert queue._semaphore._value == 1

        await queue.release_slot_for(None)
        assert counter.value == 1
        assert queue._semaphore._value == 2

    @pytest.mark.asyncio
    async def test_teardown_exception_does_not_affect_other_datasets(self):
        """If teardown raises for one dataset, another dataset's slot is unaffected."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        call_count = 0

        async def teardown_fails_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("kaboom")

        queue._teardown_subprocess_engines = teardown_fails_once

        await queue.ensure_slot("ds-ok")
        await queue.ensure_slot("ds-fail")

        with pytest.raises(RuntimeError, match="kaboom"):
            await queue.release_slot_for("ds-fail")

        # ds-ok is still held and can be released normally.
        await queue.release_slot_for("ds-ok")
        assert call_count == 2
        assert queue._semaphore._value == 5
