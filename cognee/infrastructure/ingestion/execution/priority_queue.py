import asyncio
import time
from uuid import UUID, uuid4
from typing import Any, List, Optional
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from .metrics import metrics_tracker

logger = get_logger("priority_queue")


class IngestionJob:
    """Represents a single document ingestion job in the system."""

    def __init__(
        self,
        dataset_id: UUID,
        tasks: List[Any],
        data: List[Any],
        user: User,
        priority: int = 1,  # 0 = High, 1 = Normal, 2 = Low
        retries: int = 3,
        timeout: float = 600.0,
        job_id: Optional[UUID] = None,
        **kwargs,
    ):
        self.job_id: UUID = job_id or uuid4()
        self.dataset_id: UUID = dataset_id
        self.tasks: List[Any] = tasks
        self.data: List[Any] = data
        self.user: User = user
        self.priority: int = priority
        self.max_retries: int = retries
        self.retries_left: int = retries
        self.timeout: float = timeout
        self.kwargs: dict = kwargs

        self.status: str = "queued"  # queued, processing, completed, failed
        self.error: Optional[str] = None
        self.created_at: float = time.time()
        self.completed_at: Optional[float] = None

    def __repr__(self) -> str:
        return f"IngestionJob(id={self.job_id}, dataset={self.dataset_id}, priority={self.priority}, status={self.status})"


class PriorityIngestionQueue:
    """Priority queue for scheduling document ingestion tasks."""

    def __init__(self):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._counter: int = 0
        self._jobs: dict[UUID, IngestionJob] = {}

    def submit_job(self, job: IngestionJob) -> None:
        """Add a job to the queue."""
        self._counter += 1
        self._jobs[job.job_id] = job
        # We push a tuple (priority, FIFO_counter, job)
        self._queue.put_nowait((job.priority, self._counter, job))
        metrics_tracker.job_submitted()
        logger.info("Job %s (priority %d) submitted to ingestion queue", job.job_id, job.priority)

    async def get_job(self) -> IngestionJob:
        """Retrieve the next highest priority job from the queue."""
        priority, _, job = await self._queue.get()
        return job

    def task_done(self) -> None:
        """Acknowledge that a previously retrieved job is processed."""
        self._queue.task_done()

    def get_job_by_id(self, job_id: UUID) -> Optional[IngestionJob]:
        """Fetch job details by ID."""
        return self._jobs.get(job_id)

    def size(self) -> int:
        """Returns the number of items in the queue."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Returns True if the queue is empty."""
        return self._queue.empty()


# Global ingestion queue
ingestion_queue = PriorityIngestionQueue()
