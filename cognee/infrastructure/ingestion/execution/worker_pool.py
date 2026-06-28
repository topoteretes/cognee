import asyncio
import time
import os
from typing import List, Optional
from cognee.shared.logging_utils import get_logger
from .priority_queue import IngestionJob, PriorityIngestionQueue, ingestion_queue
from .metrics import metrics_tracker

logger = get_logger("worker_pool")


class BackgroundWorkerPool:
    """Manages a pool of concurrent asyncio worker tasks that process ingestion jobs."""

    def __init__(self, queue: PriorityIngestionQueue = ingestion_queue, num_workers: int = None):
        self.queue: PriorityIngestionQueue = queue

        # Concurrency limit configuration
        if num_workers is None:
            try:
                num_workers = int(os.getenv("COGNEE_INGESTION_CONCURRENCY", "2"))
            except ValueError:
                num_workers = 2
        self.num_workers: int = max(1, num_workers)

        self._workers: List[asyncio.Task] = []
        self._running: bool = False

    def start(self) -> None:
        """Start the background worker tasks."""
        if self._running:
            logger.warning("BackgroundWorkerPool is already running.")
            return

        self._running = True
        for i in range(self.num_workers):
            task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(task)
        logger.info("BackgroundWorkerPool started with %d workers", self.num_workers)

    async def stop(self) -> None:
        """Stop the background worker tasks gracefully."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping BackgroundWorkerPool...")
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("BackgroundWorkerPool stopped.")

    async def _worker_loop(self, worker_id: int) -> None:
        """Main loop for a single worker task."""
        logger.debug("Worker %d loop started", worker_id)

        while self._running:
            try:
                job = await self.queue.get_job()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker %d failed to get job: %s", worker_id, e)
                await asyncio.sleep(1)
                continue

            job.status = "processing"
            metrics_tracker.job_started()
            start_time = time.time()

            logger.info("Worker %d starting job %s", worker_id, job.job_id)

            try:
                # Execute pipeline with timeout enforcement
                await asyncio.wait_for(self._run_job_pipeline(job), timeout=job.timeout)

                # Successful execution
                duration = time.time() - start_time
                job.status = "completed"
                job.completed_at = time.time()
                metrics_tracker.job_completed_successfully(duration)
                logger.info(
                    "Worker %d completed job %s successfully in %.2fs",
                    worker_id,
                    job.job_id,
                    duration,
                )

            except asyncio.TimeoutError:
                logger.error("Job %s timed out after %.1fs", job.job_id, job.timeout)
                await self._handle_job_failure(job, "Job timed out")

            except Exception as error:
                logger.error("Job %s failed with error: %s", job.job_id, error, exc_info=True)
                await self._handle_job_failure(job, str(error))

            finally:
                self.queue.task_done()

    async def _run_job_pipeline(self, job: IngestionJob) -> None:
        """Run the actual pipeline tasks for the job."""
        from cognee.modules.pipelines import run_pipeline

        # Consume the generator to drive the pipeline tasks to completion
        generator = run_pipeline(
            tasks=job.tasks, data=job.data, datasets=[job.dataset_id], user=job.user, **job.kwargs
        )

        async for _ in generator:
            pass

    async def _handle_job_failure(self, job: IngestionJob, error_message: str) -> None:
        """Handle retry logic or permanent failure for a job."""
        job.error = error_message

        if job.retries_left > 0:
            job.retries_left -= 1
            job.status = "retrying"
            metrics_tracker.job_retried_attempt()

            # Exponential backoff: 2s, 4s, 8s...
            backoff = 2 ** (job.max_retries - job.retries_left)
            logger.warning(
                "Job %s failed. Retries left: %d. Retrying in %ds...",
                job.job_id,
                job.retries_left,
                backoff,
            )

            # We schedule re-submission after backoff
            async def re_enqueue():
                await asyncio.sleep(backoff)
                job.status = "queued"
                self.queue.submit_job(job)

            asyncio.create_task(re_enqueue())

        else:
            job.status = "failed"
            job.completed_at = time.time()
            metrics_tracker.job_execution_failed()
            logger.error(
                "Job %s failed permanently after all retries. Error: %s", job.job_id, error_message
            )


# Global worker pool singleton
worker_pool = BackgroundWorkerPool()
