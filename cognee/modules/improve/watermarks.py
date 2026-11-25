"""Stage-owned improve watermarks, read from succeeded operation rows.

A stage stamps its own START time, not the row's ``ended_at``: stages take the
dataset pipeline lock separately, so a write can land between a stage finishing
and its run record closing — against ``ended_at`` that write would be invisible
forever. Only ``succeeded`` operations count; a failed run may not have done its
work, and a ``noop`` row (a lost lock claim, an all-skipped run) did none.
Stamp filtering happens in SQL, so a stage that rarely runs survives a long
history of runs in which it was disabled. Missing or malformed stamps mean
rerun: being conservative costs extra work, never lost work.

TODO(SDK-416 follow-up): a watermark is stage-owned state parked in the run
record because cognee has no per-dataset state store. ``run_info`` was a dormant
column improve itself has no use for. A dedicated dataset-keyed state row would
remove the scan and give the operation record back — at the cost of a migration
(this PR needs none) and of rebuilding the invalidation the row's outcome
provides for free.
"""

from datetime import datetime, timezone
from typing import NamedTuple
from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("improve.watermarks")

# Operations that only read a dataset.
READ_OPERATION_NAMES = ("search", "recall")

IMPROVE_OPERATION_NAME = "improve"

REVIEW_WATERMARK_KEY = "review_conflicts"
# Renaming it makes every dataset rebuild its index once, e.g. after the index text changes.
GLOBAL_CONTEXT_WATERMARK_KEY = "global_context_index"


class StageWatermark(NamedTuple):
    """The newest succeeded improve row carrying one stage's stamp."""

    operation_id: UUID
    started_at: datetime
    stamp: dict


def watermark_stamp(key: str, status: str, started_at: datetime, **extra) -> dict:
    """Record the stage start, before any writes that a later run must notice."""
    return {key: {"status": status, "started_at": started_at.isoformat(), **extra}}


def _stamped_stage_start(run_info, key: str) -> datetime | None:
    if not isinstance(run_info, dict):
        return None
    stamp = run_info.get(key)
    if not isinstance(stamp, dict):
        return None
    try:
        parsed = datetime.fromisoformat(str(stamp.get("started_at")))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


async def last_improve_watermark(
    dataset_id: UUID, key: str, *, exclude_operation_id: UUID | None = None
) -> StageWatermark | None:
    """Read the newest succeeded operation carrying this stage's stamp."""
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.pipelines.models import PipelineRun

    conditions = [
        PipelineRun.dataset_id == dataset_id,
        PipelineRun.operation_name == IMPROVE_OPERATION_NAME,
        PipelineRun.outcome == "succeeded",
        PipelineRun.status.is_(None),
        PipelineRun.run_info[key].as_string().is_not(None),
    ]
    if exclude_operation_id is not None:
        conditions.append(PipelineRun.pipeline_run_id != exclude_operation_id)
    async with get_relational_engine().get_async_session() as session:
        row = (
            await session.execute(
                select(PipelineRun.pipeline_run_id, PipelineRun.run_info)
                .where(*conditions)
                .order_by(PipelineRun.ended_at.desc())
                .limit(1)
            )
        ).first()
    if row is None:
        return None
    started_at = _stamped_stage_start(row[1], key)
    # _stamped_stage_start accepted it, so row[1][key] is a dict carrying a date.
    return StageWatermark(row[0], started_at, row[1][key]) if started_at is not None else None


async def has_dataset_changed_since_last_improve(
    dataset_id: UUID,
    key: str,
    *,
    exclude_operation_id: UUID | None = None,
    any_dataset: bool = False,
) -> bool:
    """True unless the ``key`` stage's stamp exists and only reads ran after it.

    For a stage whose output must not lag the graph: every run row counts,
    whatever its pipeline or outcome, except a read, a ``noop`` improve and the
    stamped run's own. A failed run may have written part of its data, a
    deletion runs no pipeline, add() can drop a source's stale documents, and a
    writer this module does not know of counts too. ``any_dataset`` counts every
    dataset's rows, for datasets sharing a graph.
    """
    try:
        from sqlalchemy import func, or_, select

        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.pipelines.models import PipelineRun

        watermark = await last_improve_watermark(
            dataset_id, key, exclude_operation_id=exclude_operation_id
        )
        if watermark is None:
            return True

        # The stamped run's own row and the runs it started close after the stamp.
        # A run's start row carries no parent, so its runs are matched by run id.
        own_runs = select(PipelineRun.pipeline_run_id).where(
            PipelineRun.parent_operation_id == watermark.operation_id,
            PipelineRun.pipeline_run_id.is_not(None),  # NOT IN with a NULL matches nothing
        )
        conditions = [
            PipelineRun.created_at > watermark.started_at,
            PipelineRun.pipeline_run_id != watermark.operation_id,
            PipelineRun.pipeline_run_id.not_in(own_runs),
            or_(
                PipelineRun.operation_name.is_(None),
                PipelineRun.operation_name.not_in(READ_OPERATION_NAMES),
            ),
            # A noop improve ran no stage; a start row's outcome is NULL, which != alone drops.
            or_(PipelineRun.outcome.is_(None), PipelineRun.outcome != "noop"),
        ]
        if not any_dataset:
            conditions.append(PipelineRun.dataset_id == dataset_id)
        async with get_relational_engine().get_async_session() as session:
            changes = await session.execute(select(func.count(PipelineRun.id)).where(*conditions))
            return bool(changes.scalar_one())
    except Exception as error:
        logger.debug(
            "improve: change check could not decide, running the stage: %s", error, exc_info=True
        )
        return True
