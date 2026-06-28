import time
from typing import Dict, Any, List


class IngestionMetrics:
    def __init__(self):
        self.jobs_queued: int = 0
        self.jobs_completed: int = 0
        self.jobs_failed: int = 0
        self.jobs_retried: int = 0
        self.active_workers: int = 0
        self.processing_times: List[float] = []

    def job_submitted(self) -> None:
        self.jobs_queued += 1

    def job_started(self) -> None:
        self.active_workers += 1

    def job_completed_successfully(self, duration: float) -> None:
        self.active_workers = max(0, self.active_workers - 1)
        self.jobs_completed += 1
        self.processing_times.append(duration)
        # Keep only the last 1000 processing times to avoid memory leaks
        if len(self.processing_times) > 1000:
            self.processing_times.pop(0)

    def job_execution_failed(self) -> None:
        self.active_workers = max(0, self.active_workers - 1)
        self.jobs_failed += 1

    def job_retried_attempt(self) -> None:
        self.jobs_retried += 1

    def get_avg_processing_time(self) -> float:
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return a snapshot of current metrics."""
        return {
            "jobs_queued": self.jobs_queued,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "jobs_retried": self.jobs_retried,
            "active_workers": self.active_workers,
            "avg_processing_time_seconds": self.get_avg_processing_time(),
        }


# Global metrics tracker
metrics_tracker = IngestionMetrics()
