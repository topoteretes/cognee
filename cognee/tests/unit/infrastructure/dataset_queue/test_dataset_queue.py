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
    """Tests for release_slot_for — verifies that _evict_subprocess_engines
    fires at the right time (last holder) and that the semaphore is always
    released regardless of eviction outcome."""

    @pytest.fixture(autouse=True)
    def reset_queue_singleton(self):
        try:
            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            dataset_queue._instance = None
        except (ImportError, AttributeError):
            pass
        yield

    @staticmethod
    def _mock_evict(queue):
        """Count eviction firings via the synchronous eviction seam.

        Eviction is the queue's entire teardown responsibility — the engine
        cache closes evicted engines on its own threads, so there is nothing
        asynchronous left to wait for at this layer.
        """
        call_count = 0

        def fake_evict():
            nonlocal call_count
            call_count += 1

        queue._evict_subprocess_engines = fake_evict

        class Counter:
            @property
            def value(self):
                return call_count

        return Counter()

    @pytest.mark.asyncio
    async def test_eviction_fires_for_single_holder(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_evict(queue)

        await queue.ensure_slot("ds-A")
        await queue.release_slot_for("ds-A")
        assert counter.value == 1

    @pytest.mark.asyncio
    async def test_eviction_skipped_for_nested_depth(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_evict(queue)

        await queue.ensure_slot("ds-B")
        await queue.ensure_slot("ds-B")  # depth = 2

        await queue.release_slot_for("ds-B")
        assert counter.value == 0  # inner exit — skipped

        await queue.release_slot_for("ds-B")
        assert counter.value == 1  # outer exit — fires

    @pytest.mark.asyncio
    async def test_eviction_skipped_when_cross_task_holder_exists(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_evict(queue)
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
    async def test_eviction_fires_after_last_cross_task_holder_releases(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_evict(queue)
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
        assert counter.value == 1  # other task was last, eviction fired

    @pytest.mark.asyncio
    async def test_different_dataset_does_not_block_eviction(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_evict(queue)

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
    async def test_disabled_queue_skips_eviction(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=False, max_concurrent=5)
        counter = self._mock_evict(queue)

        await queue.release_slot_for("any-dataset")
        assert counter.value == 0

    @pytest.mark.asyncio
    async def test_eviction_exception_still_releases_slot(self):
        """Slot must be freed even if _evict_subprocess_engines raises; the
        error propagates to the caller rather than being eaten."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=1)
        ds = "ds-E"

        def failing_evict():
            raise ValueError("eviction failed")

        queue._evict_subprocess_engines = failing_evict

        await queue.ensure_slot(ds)
        with pytest.raises(ValueError, match="eviction failed"):
            await queue.release_slot_for(ds)

        # Replace the failing mock so the verification calls below don't blow up.
        self._mock_evict(queue)

        # Semaphore must have been released — acquiring again should not block.
        await asyncio.wait_for(queue.ensure_slot(ds), timeout=1)
        await queue.release_slot_for(ds)

    @pytest.mark.asyncio
    async def test_release_without_ensure_is_noop(self):
        """Releasing a slot that was never acquired must not crash."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_evict(queue)

        await queue.release_slot_for("never-acquired")
        assert counter.value == 0  # no entry — eviction not called
        assert queue._semaphore._value == 5  # nothing consumed

    @pytest.mark.asyncio
    async def test_double_release_is_idempotent(self):
        """Calling release twice for the same slot must not crash or over-release."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=2)
        counter = self._mock_evict(queue)
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
        self._mock_evict(queue)

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
    async def test_three_tasks_eviction_fires_only_on_last(self):
        """With three tasks on the same dataset, eviction fires once on the last exit."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_evict(queue)
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
    async def test_eviction_fires_exactly_once_under_stress(self):
        """Many tasks on the same dataset; eviction fires exactly once total."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        n_tasks = 20
        queue = DatasetQueue(enabled=True, max_concurrent=n_tasks + 1)
        counter = self._mock_evict(queue)
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
    async def test_backstop_frees_slot_then_remaining_task_fires_eviction(self):
        """Task-end backstop releases a crashed task's slot without eviction.
        The surviving task should then be the last holder and fire eviction."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        counter = self._mock_evict(queue)
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
        counter = self._mock_evict(queue)
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
        counter = self._mock_evict(queue)

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
        counter = self._mock_evict(queue)

        await queue.ensure_slot(None)
        assert queue._semaphore._value == 1

        await queue.release_slot_for(None)
        assert counter.value == 1
        assert queue._semaphore._value == 2

    @pytest.mark.asyncio
    async def test_eviction_exception_does_not_affect_other_datasets(self):
        """If eviction raises for one dataset, another dataset's slot is unaffected."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        call_count = 0

        def evict_fails_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("kaboom")

        queue._evict_subprocess_engines = evict_fails_once

        await queue.ensure_slot("ds-ok")
        await queue.ensure_slot("ds-fail")

        with pytest.raises(RuntimeError, match="kaboom"):
            await queue.release_slot_for("ds-fail")

        # ds-ok is still held and can be released normally.
        await queue.release_slot_for("ds-ok")
        assert call_count == 2
        assert queue._semaphore._value == 5


class TestActiveDatasetIds:
    """Tests for the read-only active_dataset_ids snapshot used by engine-cache pinning."""

    @pytest.mark.asyncio
    async def test_tracks_slots_and_excludes_sentinels(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=4)
        assert queue.active_dataset_ids() == set()

        await queue.ensure_slot("dataset-1")
        await queue.ensure_slot("dataset-2")
        await queue.ensure_slot(None)  # "ds:<none>" must not be reported
        async with queue.acquire():  # re-entrant pass-through: no "acquire:" slot is added
            assert queue.active_dataset_ids() == {"dataset-1", "dataset-2"}

        await queue.release_slot_for("dataset-1")
        assert queue.active_dataset_ids() == {"dataset-2"}

    @pytest.mark.asyncio
    async def test_acquire_only_slots_are_not_dataset_ids(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=4)
        async with queue.acquire():  # scoped slot without a dataset id
            assert queue.active_dataset_ids() == set()

    def test_disabled_queue_reports_no_active_datasets(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=False, max_concurrent=4)
        assert queue.active_dataset_ids() == set()

    @pytest.mark.asyncio
    async def test_depth_does_not_duplicate_and_survives_until_last_release(self):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=4)
        await queue.ensure_slot("dataset-1")
        await queue.ensure_slot("dataset-1")  # re-entrant: depth bump, same slot
        assert queue.active_dataset_ids() == {"dataset-1"}

        await queue.release_slot_for("dataset-1")
        assert queue.active_dataset_ids() == {"dataset-1"}  # depth 1 remains
        await queue.release_slot_for("dataset-1")
        assert queue.active_dataset_ids() == set()


class TestEvictSubprocessEngines:
    """The queue's teardown is eviction only — the engine cache owns the close."""

    @pytest.mark.asyncio
    async def test_evict_is_synchronous_at_release(self):
        """Eviction must complete before release returns — a caller on the
        very next line must never fetch a dying engine from the cache.
        (Regression: deferring eviction let e2e flows grab an engine whose
        close then killed it under them: "LadybugAdapter is closed; a new
        adapter must be created".)"""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        evicted = False

        def evict():
            nonlocal evicted
            evicted = True

        queue._evict_subprocess_engines = evict
        await queue.ensure_slot("ds-S")
        await queue.release_slot_for("ds-S")
        assert evicted, "eviction must run inline at release"

    @staticmethod
    def _modules():
        """Resolve the patched modules via importlib: dotted-string patch
        targets like ``...graph.get_graph_engine`` can resolve to the
        same-named FUNCTION exported by the package __init__ instead of the
        submodule (broke on Python 3.10 in CI)."""
        import importlib

        return (
            importlib.import_module("cognee.infrastructure.databases.graph.config"),
            importlib.import_module("cognee.infrastructure.databases.vector.config"),
            importlib.import_module("cognee.infrastructure.databases.graph.get_graph_engine"),
            importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine"),
        )

    def test_eviction_routes_through_cache_evict(self):
        """_evict_subprocess_engines must evict via the cache helpers — it must
        NOT fetch engines or close them itself: a fetched lease proxy defers
        the real close behind the lease and hides it from the cache's
        pending-close registry, and a loop-scheduled close starves when a
        synchronous engine spawn blocks the loop (the "Lock is held by PID N"
        e2e failures)."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        g_conf_mod, v_conf_mod, g_engine_mod, v_engine_mod = self._modules()

        g_cfg = {"graph_database_subprocess_enabled": True, "graph_database_name": "g"}
        v_cfg = {"vector_db_subprocess_enabled": True, "vector_db_name": "v"}

        with (
            patch.object(g_conf_mod, "get_graph_context_config", return_value=g_cfg),
            patch.object(v_conf_mod, "get_vectordb_context_config", return_value=v_cfg),
            patch.object(g_engine_mod, "evict_graph_engine") as evict_graph,
            patch.object(v_engine_mod, "evict_vector_engine") as evict_vector,
            patch.object(g_engine_mod, "create_graph_engine") as create_graph,
        ):
            queue._evict_subprocess_engines()

        # force_close: an idle holder pinning the lease proxy (e.g. a test
        # keeping a get_graph_engine() handle) must not defer the close and
        # keep the worker's file locks alive indefinitely.
        evict_graph.assert_called_once_with(force_close=True, **g_cfg)
        evict_vector.assert_called_once_with(force_close=True, **v_cfg)
        create_graph.assert_not_called()

    def test_eviction_skips_non_subprocess_engines(self):
        """Engines running in-process are not evicted at release."""
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5)
        g_conf_mod, v_conf_mod, g_engine_mod, v_engine_mod = self._modules()

        with (
            patch.object(
                g_conf_mod,
                "get_graph_context_config",
                return_value={"graph_database_subprocess_enabled": False},
            ),
            patch.object(
                v_conf_mod,
                "get_vectordb_context_config",
                return_value={"vector_db_subprocess_enabled": False},
            ),
            patch.object(g_engine_mod, "evict_graph_engine") as evict_graph,
            patch.object(v_engine_mod, "evict_vector_engine") as evict_vector,
        ):
            queue._evict_subprocess_engines()

        evict_graph.assert_not_called()
        evict_vector.assert_not_called()


class TestIdleKeepAlive:
    """SUBPROCESS_IDLE_TTL_SECONDS keeps engines warm across releases."""

    def _counting_queue(self, ttl):
        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5, idle_ttl_seconds=ttl)
        calls = {"touch": 0, "evict": 0, "reaper": 0}
        queue._touch_subprocess_engines = lambda: calls.__setitem__("touch", calls["touch"] + 1)
        queue._evict_subprocess_engines = lambda: calls.__setitem__("evict", calls["evict"] + 1)
        queue._ensure_reaper = lambda: calls.__setitem__("reaper", calls["reaper"] + 1)
        return queue, calls

    @pytest.mark.asyncio
    async def test_ttl_release_touches_instead_of_evicting(self):
        queue, calls = self._counting_queue(ttl=900)

        await queue.ensure_slot("ds-K")
        await queue.release_slot_for("ds-K")
        assert calls == {"touch": 1, "evict": 0, "reaper": 1}

    @pytest.mark.asyncio
    async def test_zero_ttl_evicts_at_release(self):
        queue, calls = self._counting_queue(ttl=0)

        await queue.ensure_slot("ds-K")
        await queue.release_slot_for("ds-K")
        assert calls == {"touch": 0, "evict": 1, "reaper": 0}

    def test_reaper_starts_exactly_once(self):
        from unittest.mock import MagicMock

        from cognee.infrastructure.databases.dataset_queue.queue import DatasetQueue

        queue = DatasetQueue(enabled=True, max_concurrent=5, idle_ttl_seconds=900)
        thread = MagicMock()
        with patch(
            "cognee.infrastructure.databases.dataset_queue.queue.threading.Thread",
            return_value=thread,
        ) as thread_cls:
            queue._ensure_reaper()
            queue._ensure_reaper()
        thread_cls.assert_called_once()
        thread.start.assert_called_once()

    def test_settings_read_ttl_from_env(self):
        import os
        from unittest.mock import patch as env_patch

        from cognee.infrastructure.databases.dataset_queue.queue import (
            get_dataset_queue_settings,
        )

        with env_patch.dict(os.environ, {"SUBPROCESS_IDLE_TTL_SECONDS": "42.5"}):
            assert get_dataset_queue_settings().idle_ttl_seconds == 42.5
        with env_patch.dict(os.environ, {"SUBPROCESS_IDLE_TTL_SECONDS": "-3"}):
            assert get_dataset_queue_settings().idle_ttl_seconds == 0.0
        with env_patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SUBPROCESS_IDLE_TTL_SECONDS", None)
            assert get_dataset_queue_settings().idle_ttl_seconds == 900.0
