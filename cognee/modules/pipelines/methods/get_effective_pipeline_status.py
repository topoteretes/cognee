import enum
import os
from datetime import datetime, timedelta, timezone
from functools import cache

from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


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

    ABANDONED is uppercase on purpose, even though the equivalent on the
    sessions side is SessionStatus.ABANDONED = "abandoned". The other four
    values here are the stored enum's and cannot change without breaking
    the wire format, and a lowercase value sitting next to
    "DATASET_PROCESSING_STARTED" in the same field is worse than differing
    from /v1/sessions: a client reads this one field and meets both
    spellings at once, while the two endpoints are consumed separately.
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


_DEFAULT_ABANDON_AFTER_SECONDS = 1800
_ABANDON_AFTER_ENV = "PIPELINE_RUN_ABANDON_AFTER_SECONDS"


@cache
def _abandon_after_seconds_for(raw: str) -> int:
    """Validate one raw env value, warning once per distinct bad value.

    Cached on the raw string so a misconfigured deployment gets one warning
    rather than one per request: the reporting endpoints are polled, and
    this is read on every call, so warning unconditionally would repeat the
    same line for as long as the process runs. Distinct values are distinct
    cache keys, so changing the variable is still picked up.
    """
    if not raw:
        return _DEFAULT_ABANDON_AFTER_SECONDS

    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: not an integer number of seconds. Using %ds.",
            _ABANDON_AFTER_ENV,
            raw,
            _DEFAULT_ABANDON_AFTER_SECONDS,
        )
        return _DEFAULT_ABANDON_AFTER_SECONDS

    if value <= 0:
        logger.warning(
            "Ignoring %s=%d: must be positive, a threshold at or after now would "
            "mark every in-flight run abandoned. Using %ds.",
            _ABANDON_AFTER_ENV,
            value,
            _DEFAULT_ABANDON_AFTER_SECONDS,
        )
        return _DEFAULT_ABANDON_AFTER_SECONDS

    if value > _MAX_ABANDON_AFTER_SECONDS:
        logger.warning(
            "Ignoring %s=%d: above the %ds ceiling, timedelta() overflows past it. Using %ds.",
            _ABANDON_AFTER_ENV,
            value,
            _MAX_ABANDON_AFTER_SECONDS,
            _DEFAULT_ABANDON_AFTER_SECONDS,
        )
        return _DEFAULT_ABANDON_AFTER_SECONDS

    return value


def _pipeline_run_abandon_after_seconds() -> int:
    return _abandon_after_seconds_for(os.environ.get(_ABANDON_AFTER_ENV, ""))


def get_abandon_cutoff() -> datetime:
    """The instant a STARTED row has to predate to count as abandoned.

    Computed once per request and handed to get_effective_pipeline_status
    rather than recomputed inside it. A page holds up to 500 rows, and
    taking a fresh now() per row would judge the first and last rows of one
    response against slightly different clocks, so the boundary would not be
    consistent within a single response. It also keeps the env var read to
    once per request, and lets a frozen-clock test cover a whole page.
    """
    return datetime.now(timezone.utc) - timedelta(seconds=_pipeline_run_abandon_after_seconds())


def get_effective_pipeline_status(
    run: PipelineRun, *, run_has_terminal_row: bool, abandon_cutoff: datetime
) -> EffectivePipelineRunStatus | None:
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

    abandon_cutoff comes from get_abandon_cutoff() and is also required, so
    that one value covers every row of a response instead of each row
    picking up its own clock.

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
    created_at = run.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if created_at < abandon_cutoff:
        return EffectivePipelineRunStatus.ABANDONED
    return EffectivePipelineRunStatus(run.status.value)
