"""Progress heartbeat for in-flight pipeline runs.

``PipelineRun`` is an append-only event log: ``log_pipeline_run_start`` /
``_complete`` / ``_error`` each INSERT a row sharing one ``pipeline_run_id``,
and a run's current status is its newest row. That records *transitions* but
nothing that moves while a run is working, so a run wedged for six hours and a
run healthily chewing through a large document look identical to anything
outside the task. That ambiguity is what blocks cleaning up abandoned runs:
with a local LLM a run can legitimately take days, so age alone can never
separate "slow" from "dead" (see ``cognee.modules.cognify.recovery``).

The heartbeat closes that gap. Every time the pipeline finishes a task it
stamps ``last_heartbeat_at`` on the run's STARTED row. The signal is
*progress*, not wall clock: nothing advances the heartbeat on its own, so a
days-long local-LLM run keeps looking alive as long as it keeps completing
tasks, while a stalled run's heartbeat freezes no matter how much time passes.

Writes are throttled so the signal stays cheap under load. The cost scales
with data items, not with streamed batches: a measured add + cognify of a
single 20KB document produced 7 task completions in 86s, with every task
firing exactly once, so a small ingestion is nowhere near expensive enough to
need throttling. ``run_tasks`` runs up to ``data_per_batch`` items
concurrently though, so the same 7 completions per document become thousands
of UPDATEs against one row for a large multi-document ingestion, and on the
default SQLite backend thousands of write-lock acquisitions with them.
Throttling bounds that at one UPDATE per
``COGNEE_PIPELINE_HEARTBEAT_INTERVAL_SECONDS`` per run without turning the
heartbeat back into a timer: it can only ever *suppress* a write that progress
already asked for, never manufacture one.

Granularity is coarse, and that is the real limitation. Progress is only
observable at task boundaries, so the LLM-bound stage of that measured run
left a 79 second window with no heartbeat at all. Under a local LLM the same
stage can run far longer, so the staleness threshold that consumes this signal
has to exceed the longest single *task*, not merely be larger than some
comfortable multiple of the heartbeat interval.
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import update

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.shared.logging_utils import get_logger

logger = get_logger("heartbeat_pipeline_run")

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0

# Monotonic timestamp of the last heartbeat write per run, used to throttle.
# Keyed by str(pipeline_run_id). Bounded by _prune_write_slots.
_last_write_monotonic: dict[str, float] = {}

# Runs are short-lived relative to process lifetime, but a long-running server
# processes many of them, so the throttle map is pruned rather than left to
# grow for the life of the process.
_MAX_TRACKED_RUNS = 1024


def get_heartbeat_interval_seconds() -> float:
    """Minimum seconds between two heartbeat writes for one run.

    ``0`` writes on every task completion (useful in tests); a negative value
    disables heartbeat writes entirely.
    """
    raw = os.getenv("COGNEE_PIPELINE_HEARTBEAT_INTERVAL_SECONDS")
    if raw is None:
        return DEFAULT_HEARTBEAT_INTERVAL_SECONDS

    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid COGNEE_PIPELINE_HEARTBEAT_INTERVAL_SECONDS=%r, falling back to %s.",
            raw,
            DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        )
        return DEFAULT_HEARTBEAT_INTERVAL_SECONDS


def _prune_write_slots(now: float, interval: float) -> None:
    """Drop throttle entries for runs that have not written in a long while."""
    cutoff = max(interval * 10, 600.0)
    stale_keys = [
        key for key, written_at in _last_write_monotonic.items() if now - written_at > cutoff
    ]
    for key in stale_keys:
        _last_write_monotonic.pop(key, None)


def _claim_write_slot(key: str, interval: float) -> bool:
    """Reserve the right to write a heartbeat for ``key``, or return False.

    The reservation is recorded *before* the caller awaits the database. A
    single run has many concurrent data items in flight (``run_tasks`` gathers
    up to ``data_per_batch`` of them), and a check-then-await-then-record
    ordering would let every one of them pass the check while the first write
    is still on the wire.
    """
    now = time.monotonic()
    previous = _last_write_monotonic.get(key)

    if previous is not None and (now - previous) < interval:
        return False

    if len(_last_write_monotonic) >= _MAX_TRACKED_RUNS:
        _prune_write_slots(now, interval)

    _last_write_monotonic[key] = now
    return True


def reset_heartbeat_throttle() -> None:
    """Clear all throttle state. Intended for tests."""
    _last_write_monotonic.clear()


async def heartbeat_pipeline_run(pipeline_run_id: Optional[UUID]) -> bool:
    """Record that ``pipeline_run_id`` just made progress.

    Returns True when a row was actually written, False when the call was
    throttled, disabled, unaddressed, or failed. Never raises: a heartbeat is
    an observability signal, and losing one must not fail the pipeline task
    that produced it.
    """
    if pipeline_run_id is None:
        return False

    interval = get_heartbeat_interval_seconds()
    if interval < 0:
        return False

    if not _claim_write_slot(str(pipeline_run_id), interval):
        return False

    try:
        db_engine = get_relational_engine()

        async with db_engine.get_async_session() as session:
            # Scoped to the STARTED row: it is the only row of a run that
            # represents "in flight", and it is unique because
            # generate_pipeline_run_id returns a fresh uuid4 per execution.
            await session.execute(
                update(PipelineRun)
                .where(PipelineRun.pipeline_run_id == pipeline_run_id)
                .where(PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_STARTED)
                .values(last_heartbeat_at=datetime.now(timezone.utc))
            )
            await session.commit()

        return True
    except Exception as error:
        # The throttle reservation is deliberately left in place so a database
        # that is failing persistently is not retried at every task boundary.
        logger.debug(
            "Pipeline heartbeat write failed for run %s: %s",
            pipeline_run_id,
            error,
        )
        return False
