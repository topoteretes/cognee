import asyncio
import pytest
from uuid import uuid4
from cognee.infrastructure.ingestion.execution import (
    IngestionJob,
    PriorityIngestionQueue,
    BackgroundWorkerPool,
    metrics_tracker,
)
from cognee.modules.users.models import User
from cognee.modules.pipelines.tasks.task import Task

class MockUser(User):
    pass

@pytest.mark.asyncio
async def test_priority_scheduling():
    """Verify that jobs are processed in priority order."""
    queue = PriorityIngestionQueue()
    user = MockUser(id=uuid4())
    
    # 0 = High, 1 = Normal, 2 = Low
    job_low = IngestionJob(dataset_id=uuid4(), tasks=[], data=[], user=user, priority=2)
    job_high = IngestionJob(dataset_id=uuid4(), tasks=[], data=[], user=user, priority=0)
    job_normal = IngestionJob(dataset_id=uuid4(), tasks=[], data=[], user=user, priority=1)
    
    queue.submit_job(job_low)
    queue.submit_job(job_high)
    queue.submit_job(job_normal)
    
    first = await queue.get_job()
    second = await queue.get_job()
    third = await queue.get_job()
    
    assert first.job_id == job_high.job_id
    assert second.job_id == job_normal.job_id
    assert third.job_id == job_low.job_id


@pytest.mark.asyncio
async def test_worker_pool_concurrency():
    """Verify that the worker pool processes jobs concurrently."""
    queue = PriorityIngestionQueue()
    pool = BackgroundWorkerPool(queue=queue, num_workers=2)
    user = MockUser(id=uuid4())
    
    execution_order = []
    
    async def mock_task(*args, **kwargs):
        execution_order.append("start")
        await asyncio.sleep(0.2)
        execution_order.append("end")
        yield "result"

    # Override pipeline runner to execute tasks directly, avoiding DB and LLM check requirements.
    async def mock_run_job_pipeline(job):
        for task in job.tasks:
            async for _ in task.execute(job.data, {}):
                pass
    pool._run_job_pipeline = mock_run_job_pipeline

    job1 = IngestionJob(dataset_id=uuid4(), tasks=[Task(mock_task)], data=["data1"], user=user)
    job2 = IngestionJob(dataset_id=uuid4(), tasks=[Task(mock_task)], data=["data2"], user=user)
    
    queue.submit_job(job1)
    queue.submit_job(job2)
    
    pool.start()
    await asyncio.sleep(0.4)
    await pool.stop()
    
    # Since we have 2 workers, they should start concurrently
    # The list should start with two "start" markers before any "end" markers
    assert execution_order[:2] == ["start", "start"]
    assert execution_order[2:] == ["end", "end"]


@pytest.mark.asyncio
async def test_worker_pool_retries():
    """Verify that failing jobs are retried up to max_retries."""
    queue = PriorityIngestionQueue()
    pool = BackgroundWorkerPool(queue=queue, num_workers=1)
    user = MockUser(id=uuid4())
    
    fail_count = 0
    
    async def mock_failing_task(*args, **kwargs):
        nonlocal fail_count
        fail_count += 1
        raise RuntimeError("Mock task failure")
        yield None

    async def mock_run_job_pipeline(job):
        for task in job.tasks:
            async for _ in task.execute(job.data, {}):
                pass
    pool._run_job_pipeline = mock_run_job_pipeline

    # Max retries = 2 (so total attempts = 3)
    job = IngestionJob(dataset_id=uuid4(), tasks=[Task(mock_failing_task)], data=["data"], user=user, retries=2)
    
    queue.submit_job(job)
    pool.start()
    
    # Wait for execution and retries (backoff: 2s, 4s...)
    await asyncio.sleep(7.0)
    await pool.stop()
    
    assert fail_count == 3
    assert job.status == "failed"
    assert "Mock task failure" in job.error


@pytest.mark.asyncio
async def test_worker_pool_timeout():
    """Verify that jobs that exceed timeout are cancelled."""
    queue = PriorityIngestionQueue()
    pool = BackgroundWorkerPool(queue=queue, num_workers=1)
    user = MockUser(id=uuid4())
    
    task_completed = False
    
    async def long_running_task(*args, **kwargs):
        nonlocal task_completed
        await asyncio.sleep(2.0)
        task_completed = True
        yield "done"

    async def mock_run_job_pipeline(job):
        for task in job.tasks:
            async for _ in task.execute(job.data, {}):
                pass
    pool._run_job_pipeline = mock_run_job_pipeline

    # Timeout = 0.5 seconds
    job = IngestionJob(dataset_id=uuid4(), tasks=[Task(long_running_task)], data=["data"], user=user, timeout=0.5, retries=0)
    
    queue.submit_job(job)
    pool.start()
    await asyncio.sleep(1.0)
    await pool.stop()
    
    assert not task_completed
    assert job.status == "failed"
    assert "timed out" in job.error.lower()
