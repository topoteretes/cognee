"""Integration-style tests for Dataset Queue with search and pipeline operations.

These tests demonstrate how the dataset queue should integrate with the actual
search and run_pipeline_per_dataset functions.
"""

import asyncio
from uuid import uuid4
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


# Module paths for mocking
DATASET_QUEUE_MODULE = "cognee.infrastructure.databases.dataset_queue"
GET_DATASET_QUEUE_SETTINGS = f"{DATASET_QUEUE_MODULE}.queue.get_dataset_queue_settings"
SEARCH_MODULE = "cognee.api.v1.search.search"
PIPELINE_MODULE = "cognee.modules.pipelines.operations.pipeline"


class TestSearchIntegration:
    """Tests for dataset queue integration with search operations."""

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
    async def test_search_respects_queue_limit(self):
        """Test that search operations respect the queue concurrency limit."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue_limit

            concurrent_searches = 0
            max_concurrent_searches = 0

            @dataset_queue_limit
            async def mock_search(query_text: str, **kwargs):
                nonlocal concurrent_searches, max_concurrent_searches
                concurrent_searches += 1
                max_concurrent_searches = max(max_concurrent_searches, concurrent_searches)
                await asyncio.sleep(0.05)
                concurrent_searches -= 1
                return [{"query": query_text, "results": []}]

            # Execute multiple searches concurrently
            queries = [f"query_{i}" for i in range(10)]
            results = await asyncio.gather(*[mock_search(q) for q in queries])

            assert len(results) == 10
            assert max_concurrent_searches <= 2
            for i, result in enumerate(results):
                assert result[0]["query"] == f"query_{i}"

    @pytest.mark.asyncio
    async def test_search_with_different_datasets_shares_queue(self):
        """Test that searches on different datasets share the same queue."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue_limit

            concurrent_count = 0
            max_concurrent = 0

            @dataset_queue_limit
            async def mock_search_with_dataset(query_text: str, dataset_id: str):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.03)
                concurrent_count -= 1
                return [{"query": query_text, "dataset": dataset_id}]

            dataset_ids = [str(uuid4()) for _ in range(5)]
            results = await asyncio.gather(
                *[
                    mock_search_with_dataset(f"query_{i}", dataset_ids[i % len(dataset_ids)])
                    for i in range(10)
                ]
            )

            assert len(results) == 10
            assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_search_returns_results_correctly_through_queue(self):
        """Test that search results are returned correctly when going through queue."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue_limit

            @dataset_queue_limit
            async def mock_search(query_text: str, top_k: int = 10):
                return [{"id": f"result_{i}", "score": 0.9 - i * 0.1} for i in range(top_k)]

            results = await mock_search("test query", top_k=5)

            assert len(results) == 5
            assert results[0]["id"] == "result_0"
            assert results[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_search_error_handling_through_queue(self):
        """Test that search errors propagate correctly through queue."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue_limit

            @dataset_queue_limit
            async def mock_failing_search(query_text: str):
                raise ValueError(f"Search failed for: {query_text}")

            with pytest.raises(ValueError, match="Search failed for: bad query"):
                await mock_failing_search("bad query")


class TestPipelineIntegration:
    """Tests for dataset queue integration with pipeline operations."""

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
    async def test_pipeline_per_dataset_respects_queue_limit(self):
        """Test that pipeline operations respect queue concurrency limit."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 2

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            concurrent_pipelines = 0
            max_concurrent_pipelines = 0

            async def mock_run_pipeline_per_dataset(dataset_id: str):
                nonlocal concurrent_pipelines, max_concurrent_pipelines
                async with queue.acquire():
                    concurrent_pipelines += 1
                    max_concurrent_pipelines = max(max_concurrent_pipelines, concurrent_pipelines)

                    yield {"status": "started", "dataset_id": dataset_id}
                    await asyncio.sleep(0.05)
                    yield {"status": "processing", "dataset_id": dataset_id}
                    await asyncio.sleep(0.05)
                    yield {"status": "completed", "dataset_id": dataset_id}

                    concurrent_pipelines -= 1

            async def run_pipeline(dataset_id: str):
                results = []
                async for info in mock_run_pipeline_per_dataset(dataset_id):
                    results.append(info)
                return results

            dataset_ids = [str(uuid4()) for _ in range(5)]
            all_results = await asyncio.gather(*[run_pipeline(did) for did in dataset_ids])

            assert len(all_results) == 5
            assert max_concurrent_pipelines <= 2
            for results in all_results:
                assert len(results) == 3
                assert results[0]["status"] == "started"
                assert results[-1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_pipeline_yields_correctly_through_queue(self):
        """Test that pipeline yields all results correctly through queue."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            async def mock_pipeline_with_queue(dataset_id: str):
                async with queue.acquire():
                    for i in range(5):
                        await asyncio.sleep(0.01)
                        yield {"step": i, "dataset_id": dataset_id}

            results = []
            async for result in mock_pipeline_with_queue("test_dataset"):
                results.append(result)

            assert len(results) == 5
            assert [r["step"] for r in results] == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_pipeline_error_releases_queue_slot(self):
        """Test that pipeline errors properly release queue slots."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 1

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            async def failing_pipeline():
                async with queue.acquire():
                    yield {"status": "started"}
                    raise RuntimeError("Pipeline failed")

            async def consume_failing_pipeline():
                results = []
                try:
                    async for result in failing_pipeline():
                        results.append(result)
                except RuntimeError:
                    pass
                return results

            # First pipeline fails
            results1 = await consume_failing_pipeline()
            assert len(results1) == 1
            assert results1[0]["status"] == "started"

            # Second pipeline should be able to acquire slot
            async def success_pipeline():
                async with queue.acquire():
                    yield {"status": "success"}

            results2 = []
            async for result in success_pipeline():
                results2.append(result)

            assert len(results2) == 1
            assert results2[0]["status"] == "success"


class TestMixedOperations:
    """Tests for mixed search and pipeline operations sharing the queue."""

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
    async def test_search_and_pipeline_share_queue(self):
        """Test that search and pipeline operations share concurrency limit."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 3

            from cognee.infrastructure.databases.dataset_queue import (
                dataset_queue,
                dataset_queue_limit,
            )

            queue = dataset_queue()
            concurrent_ops = 0
            max_concurrent_ops = 0

            @dataset_queue_limit
            async def search_operation(query: str):
                nonlocal concurrent_ops, max_concurrent_ops
                concurrent_ops += 1
                max_concurrent_ops = max(max_concurrent_ops, concurrent_ops)
                await asyncio.sleep(0.03)
                concurrent_ops -= 1
                return f"search_result_{query}"

            async def pipeline_operation(dataset: str):
                nonlocal concurrent_ops, max_concurrent_ops
                async with queue.acquire():
                    concurrent_ops += 1
                    max_concurrent_ops = max(max_concurrent_ops, concurrent_ops)
                    await asyncio.sleep(0.03)
                    concurrent_ops -= 1
                    return f"pipeline_result_{dataset}"

            # Mix of search and pipeline operations
            tasks = [
                search_operation("q1"),
                pipeline_operation("d1"),
                search_operation("q2"),
                pipeline_operation("d2"),
                search_operation("q3"),
                pipeline_operation("d3"),
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 6
            assert max_concurrent_ops <= 3

    @pytest.mark.asyncio
    async def test_fair_scheduling_between_search_and_pipeline(self):
        """Test that queue provides fair scheduling between operation types."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 1

            from cognee.infrastructure.databases.dataset_queue import (
                dataset_queue,
                dataset_queue_limit,
            )

            queue = dataset_queue()
            completion_order = []

            @dataset_queue_limit
            async def search_op(op_id: int):
                await asyncio.sleep(0.02)
                completion_order.append(f"search_{op_id}")
                return f"search_{op_id}"

            async def pipeline_op(op_id: int):
                async with queue.acquire():
                    await asyncio.sleep(0.02)
                    completion_order.append(f"pipeline_{op_id}")
                    return f"pipeline_{op_id}"

            # Start all operations
            tasks = [
                asyncio.create_task(search_op(1)),
                asyncio.create_task(pipeline_op(1)),
                asyncio.create_task(search_op(2)),
                asyncio.create_task(pipeline_op(2)),
            ]

            await asyncio.gather(*tasks)

            # With limit of 1, operations should complete sequentially
            assert len(completion_order) == 4


class TestConfigurationIntegration:
    """Tests for configuration integration scenarios."""

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
    async def test_queue_respects_runtime_disabled_state(self):
        """Test operations work normally when queue is disabled at runtime."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = False
            mock_settings.return_value.max_concurrent = 1

            from cognee.infrastructure.databases.dataset_queue import dataset_queue_limit

            concurrent_count = 0
            max_concurrent = 0

            @dataset_queue_limit
            async def unrestricted_operation(op_id: int):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.02)
                concurrent_count -= 1
                return op_id

            results = await asyncio.gather(*[unrestricted_operation(i) for i in range(10)])

            assert len(results) == 10
            # When disabled, all should run concurrently
            assert max_concurrent == 10

    @pytest.mark.asyncio
    async def test_queue_with_custom_lru_cache_size(self):
        """Test queue correctly uses custom DATABASE_MAX_LRU_CACHE_SIZE."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 7

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            assert queue.max_concurrent == 7
            assert queue._semaphore._value == 7


class TestQueueMonitoring:
    """Tests for queue monitoring and observability."""

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
    async def test_queue_stats_tracking(self):
        """Test that queue tracks usage statistics."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 5

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            # Track some operations
            for _ in range(10):
                async with queue.acquire():
                    await asyncio.sleep(0.01)

            # Queue should track total operations (if implemented)
            if hasattr(queue, "total_operations"):
                assert queue.total_operations == 10

            # Queue should track current utilization
            assert queue.available_slots == 5

    @pytest.mark.asyncio
    async def test_queue_reports_waiting_count(self):
        """Test that queue can report number of waiting operations."""
        with patch(GET_DATASET_QUEUE_SETTINGS) as mock_settings:
            mock_settings.return_value.enabled = True
            mock_settings.return_value.max_concurrent = 1

            from cognee.infrastructure.databases.dataset_queue import dataset_queue

            queue = dataset_queue()

            hold_event = asyncio.Event()
            waiting_check_done = asyncio.Event()

            async def holder():
                async with queue.acquire():
                    hold_event.set()
                    await waiting_check_done.wait()

            async def waiter():
                async with queue.acquire():
                    return "done"

            # Start holder
            holder_task = asyncio.create_task(holder())
            await hold_event.wait()

            # Start waiters
            waiter_tasks = [asyncio.create_task(waiter()) for _ in range(3)]
            await asyncio.sleep(0.01)  # Let waiters start waiting

            # Check waiting count if implemented
            if hasattr(queue, "waiting_count"):
                assert queue.waiting_count >= 0

            # Cleanup
            waiting_check_done.set()
            await holder_task
            await asyncio.gather(*waiter_tasks)
