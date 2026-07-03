"""Configuration for the Slack + cognee bot (issue #3609).

Commit 3 scope: only the ingestion/trigger thresholds live here. Slack transport
settings (bot/app tokens, opted-in channels) are added in commit 4 when the Bolt
layer lands — this module is intentionally extended incrementally.

Thresholds are environment-driven so the batch/timer behaviour can be tuned
without code changes, following cognee's env-var configuration convention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Default number of buffered messages that triggers a cognify for a channel.
DEFAULT_COGNIFY_BATCH_SIZE = 10


@dataclass(frozen=True)
class IngestionSettings:
    """Batch/timer thresholds for the ingestion buffer.

    Attributes
    ----------
    cognify_batch_size:
        Flush (cognify) a channel once this many messages have been buffered.
    flush_interval_seconds:
        Optional time-based trigger. ``None`` disables the timer (size-only).
    """

    cognify_batch_size: int = DEFAULT_COGNIFY_BATCH_SIZE
    flush_interval_seconds: float | None = None


def load_ingestion_settings() -> IngestionSettings:
    """Build :class:`IngestionSettings` from the environment.

    * ``COGNEE_SLACK_COGNIFY_BATCH`` — int, batch size (default 10).
    * ``COGNEE_SLACK_FLUSH_INTERVAL_SECONDS`` — float, timer (default: disabled).
    """
    batch = os.getenv("COGNEE_SLACK_COGNIFY_BATCH")
    interval = os.getenv("COGNEE_SLACK_FLUSH_INTERVAL_SECONDS")
    return IngestionSettings(
        cognify_batch_size=int(batch) if batch else DEFAULT_COGNIFY_BATCH_SIZE,
        flush_interval_seconds=float(interval) if interval else None,
    )
