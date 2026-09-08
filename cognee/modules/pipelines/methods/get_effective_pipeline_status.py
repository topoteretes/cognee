import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus

# 30 minutes by default; overridable via env var for tests. Mirrors
# SESSION_ABANDON_AFTER_SECONDS in cognee/modules/session_lifecycle/metrics.py.
#
# A non-positive value would make the threshold now-or-later, flagging nearly
# every in-flight STARTED row as ABANDONED, and an absurdly large value
# overflows the timedelta() call below (OverflowError, uncaught, 500s the
# whole endpoint) — both parse fine as a plain int, so they are rejected here
# rather than left for the caller to hit.
_MAX_ABANDON_AFTER_SECONDS = 10**9  # ~31 years; timedelta stays well inside range


def _pipeline_run_abandon_after_seconds() -> int:
    raw = os.environ.get("PIPELINE_RUN_ABANDON_AFTER_SECONDS", "")
    try:
        value = int(raw) if raw else 1800
    except ValueError:
        return 1800
    if value <= 0 or value > _MAX_ABANDON_AFTER_SECONDS:
        return 1800
    return value


def get_effective_pipeline_status(run: PipelineRun) -> Optional[str]:
    """Stored status, with a stale STARTED row reported as "ABANDONED".

    A worker crash leaves the row at DATASET_PROCESSING_STARTED forever —
    there is no worker-side transition to a terminal status for that case
    (that's SDK-591 part 3, not implemented here). Rather than add an
    ABANDONED member to PipelineRunStatus, which is a native Postgres enum
    column and would need an ALTER TYPE migration, the override is computed
    here at read time, same approach as get_effective_status_sql() for
    session records.

    Only applies to pipeline rows (pipeline_name set) — SDK-399 operation
    rows have no status column at all and are untouched.
    """
    if run.status is None:
        return None
    if run.pipeline_name is None:
        return run.status.value
    if run.status != PipelineRunStatus.DATASET_PROCESSING_STARTED:
        return run.status.value
    if run.created_at is None:
        return run.status.value
    threshold = datetime.now(timezone.utc) - timedelta(
        seconds=_pipeline_run_abandon_after_seconds()
    )
    created_at = run.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if created_at < threshold:
        return "ABANDONED"
    return run.status.value
