"""Retry schedule for transient worker failures.

Inference is a pure function of its request, so resending one is always safe.
Only :class:`GlinerWorkerUnavailableError` is retried.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 4
    base_delay: float = 0.25
    max_delay: float = 8.0

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait after failed attempt ``attempt`` (0-based).

        Exponential backoff with jitter in its upper half, so replicas restarting
        together are not hit in lockstep. A server's ``Retry-After`` is honoured
        when it asks for longer, up to ``max_delay``.
        """
        backoff = min(self.max_delay, self.base_delay * (2**attempt))
        delay = random.uniform(backoff / 2, backoff)
        if retry_after is not None and retry_after > delay:
            delay = min(self.max_delay, retry_after)
        return delay
