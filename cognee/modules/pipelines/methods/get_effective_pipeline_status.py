import enum
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus


class EffectivePipelineRunStatus(str, enum.Enum):
    """Read-time/reporting status of a pipeline run.

    This is the value clients see: the stored PipelineRunStatus, plus
    ABANDONED for a STARTED row that has gone stale (see
    get_effective_pipeline_status below). PipelineRunStatus itself stays the
    DB-stored enum — a native Postgres column type — and is never given an
    ABANDONED member, so no migration is needed for this.

    Member values are identical to PipelineRunStatus's for the four shared
    states, so the wire format for anything already terminal or freshly
    started is unchanged.
    """

    DATASET_PROCESSING_INITIATED = "DATASET_PROCESSING_INITIATED"
    DATASET_PROCESSING_STARTED = "DATASET_PROCESSING_STARTED"
    DATASET_PROCESSING_COMPLETED = "DATASET_PROCESSING_COMPLETED"
    DATASET_PROCESSING_ERRORED = "DATASET_PROCESSING_ERRORED"
    ABANDONED = "ABANDONED"


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


def get_effective_pipeline_status(
    run: PipelineRun, *, run_has_terminal_row: bool
) -> Optional[EffectivePipelineRunStatus]:
    """Stored status, with a stale STARTED row reported as ABANDONED.

    This is the read-time/reporting status (EffectivePipelineRunStatus) —
    for control flow (deciding whether a pipeline is already running or
    already done) always use the raw stored PipelineRunStatus instead, e.g.
    via get_pipeline_status(). Mixing the two up silently breaks that
    control flow: see check_pipeline_run_qualification.

    A worker crash leaves the row at DATASET_PROCESSING_STARTED forever —
    there is no worker-side transition to a terminal status for that case
    (that's SDK-591 part 3, not implemented here). Rather than add an
    ABANDONED member to PipelineRunStatus, which is a native Postgres enum
    column and would need an ALTER TYPE migration, the override is computed
    here at read time, same approach as get_effective_status_sql() for
    session records.

    Staleness is decided per run, not per row. log_pipeline_run_start /
    _complete / _error each INSERT a new row sharing one pipeline_run_id
    rather than UPDATE-ing an existing one, so a finished run still has its
    original STARTED row sitting in the table next to its terminal row. That
    STARTED row is a historical marker, not a stuck worker — it must report
    its own raw status, never ABANDONED. run_has_terminal_row says whether
    the caller already knows a COMPLETED/ERRORED row exists for this run's
    pipeline_run_id; it is keyword-only with no default on purpose; a
    default would silently pick the dangerous direction (labelling a
    finished run's start row as failed).

    Only applies to pipeline rows (pipeline_name set) — SDK-399 operation
    rows have no status column at all and are untouched.
    """
    if run.status is None:
        return None
    if run.pipeline_name is None:
        return EffectivePipelineRunStatus(run.status.value)
    if run.status != PipelineRunStatus.DATASET_PROCESSING_STARTED:
        return EffectivePipelineRunStatus(run.status.value)
    if run_has_terminal_row:
        return EffectivePipelineRunStatus(run.status.value)
    if run.created_at is None:
        return EffectivePipelineRunStatus(run.status.value)
    threshold = datetime.now(timezone.utc) - timedelta(
        seconds=_pipeline_run_abandon_after_seconds()
    )
    created_at = run.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if created_at < threshold:
        return EffectivePipelineRunStatus.ABANDONED
    return EffectivePipelineRunStatus(run.status.value)
