"""Integration-style tests for Dataset Queue with pipeline-shaped operations.

These tests demonstrate how the dataset queue should integrate with the
async-generator shape used by ``run_pipeline_per_dataset``.
"""

import asyncio
from uuid import uuid4
import pytest
from unittest.mock import patch


DATASET_QUEUE_MODULE = "cognee.infrastructure.databases.dataset_queue"
GET_DATASET_QUEUE_SETTINGS = f"{DATASET_QUEUE_MODULE}.queue.get_dataset_queue_settings"


class TestPipelineIntegration:
    """Tests for dataset queue integration with pipeline-style operations."""

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
