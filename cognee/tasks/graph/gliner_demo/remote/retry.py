"""Retry schedule for transient worker failures.

Inference is a pure function of its request, so resending one is always safe.
Only :class:`GlinerWorkerUnavailableError` is retried. The policy is shared with
the embedding worker client (:mod:`cognee.infrastructure.remote_worker.retry`).
"""

from cognee.infrastructure.remote_worker.retry import RetryPolicy

__all__ = ["RetryPolicy"]
