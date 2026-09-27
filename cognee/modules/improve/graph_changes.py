"""Has anything written to this dataset's graph since its last enrichment?

The triplet-enrichment stage's ``already_completed`` gate (plan Part 5.10,
stage 8). It reads ``pipeline_runs`` — indexed on ``dataset_id`` and
``created_at`` — and never a graph-wide node or edge count, which would put
the cost back into the gate.

The watermark is stage-8-scoped, not run-scoped: an improve row counts only
when it carries the enrichment stamp (``run_info["triplet_enrichment"]``),
written only when the stage actually completed a full, unscoped enrichment —
a run whose stage 8 was skipped (triplet_embedding off, disabled by config)
never gates a later run. Writes are compared against the STAMPED STAGE START
TIME, not the row's ``ended_at``: stages 8 and 9 take the dataset pipeline
lock separately, so a cognify can land between stage 8 and the row close —
against ``ended_at`` its write would be invisible forever.

Conservative by construction: when there is no stamped improve for the
dataset, or the query cannot decide, the answer is "changed", so the stage runs.

TODO(SDK-416 follow-up): the watermark is stage-owned state parked in the run
record because cognee has no per-dataset state store. ``run_info`` was a
dormant column improve itself has no use for, and the read side derives the
current watermark from a bounded scan of run history instead of a key lookup.
A dedicated dataset-keyed state row would turn the scan into a lookup and give
the operation record back — at the cost of a migration (this PR needs none)
and of rebuilding the invalidation the row's outcome provides for free.
"""

from collections.abc import Iterable
from datetime import datetime, timezone
from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("improve.graph_changes")

# Pipelines whose completed runs write nodes or edges into a dataset's graph.
# ``add_pipeline`` is absent on purpose: add() writes relational rows and files.
WRITE_PIPELINE_NAMES = (
    "cognify_pipeline",  # cognify() and update() (incremental attributes to it)
    "code_graph_pipeline",
    "memify_pipeline",  # every memify writer, the session-persist stages included
    "custom_pipeline",  # run_custom_pipeline's default name; add_data_points writes graph data
    "presort_graph_pipeline",
    "skills_ingest_pipeline",
    "skill_improvement_pipeline",
    "skill_runs_pipeline",
    "agentic_skill_runs_pipeline",
    "migration_import_pipeline",
)

IMPROVE_OPERATION_NAME = "improve"

# run_info key stamped on an improve row whose enrichment stage completed.
ENRICHMENT_WATERMARK_KEY = "triplet_enrichment"

# Newest succeeded improve rows inspected for the enrichment stamp. Bounded so
# a dataset with a long history of stamp-less rows (skipped stage 8, pre-stamp
# releases) costs one small indexed read; running out conservatively means
# "changed".
_RECENT_IMPROVE_ROWS_SCANNED = 100


def enrichment_watermark_stamp(status: str, started_at: datetime) -> dict:
    """The ``run_info`` stamp for an improve row whose stage 8 completed.

    ``started_at`` is the stage's start: any write after it — including one
    that raced the row close — stays visible to the next run's gate.
    """
    return {ENRICHMENT_WATERMARK_KEY: {"status": status, "started_at": started_at.isoformat()}}


def _stamped_stage_start(run_info) -> datetime | None:
    if not isinstance(run_info, dict):
        return None
    stamp = run_info.get(ENRICHMENT_WATERMARK_KEY)
    if not isinstance(stamp, dict):
        return None
    try:
        parsed = datetime.fromisoformat(str(stamp.get("started_at")))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


async def has_graph_changed_since_last_improve(
    dataset_id: UUID,
    write_pipeline_names: Iterable[str] = WRITE_PIPELINE_NAMES,
    exclude_operation_id: UUID | None = None,
) -> bool:
    """True unless a stamped enrichment exists and no write pipeline completed after it.

    The watermark is the newest succeeded improve row carrying the enrichment
    stamp (see module docstring); the comparison point is that stamp's stage
    start time. ``exclude_operation_id`` is the calling run's own
    operation-record id: its row must never serve as its own watermark. The
    orchestrator closes that record only after the run finishes, so the row
    normally does not exist yet when this runs — the exclusion is insurance
    against the close moving earlier again.
    """
    try:
        from sqlalchemy import func, select

        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus

        last_improve_conditions = [
            PipelineRun.dataset_id == dataset_id,
            PipelineRun.operation_name == IMPROVE_OPERATION_NAME,
            # "succeeded" excludes failed runs (their work may not have
            # happened) and "noop" rows (a lost lock claim, an all-skipped
            # run — nothing ran, so nothing to watermark).
            PipelineRun.outcome == "succeeded",
            PipelineRun.status.is_(None),
        ]
        if exclude_operation_id is not None:
            last_improve_conditions.append(PipelineRun.pipeline_run_id != exclude_operation_id)

        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            recent_improves = (
                await session.execute(
                    select(PipelineRun.pipeline_run_id, PipelineRun.run_info)
                    .where(*last_improve_conditions)
                    .order_by(PipelineRun.ended_at.desc())
                    .limit(_RECENT_IMPROVE_ROWS_SCANNED)
                )
            ).all()

            stamped_operation_id = None
            last_enrichment_started_at = None
            for operation_id, run_info in recent_improves:
                stamped = _stamped_stage_start(run_info)
                if stamped is not None:
                    stamped_operation_id = operation_id
                    last_enrichment_started_at = stamped
                    break
            if last_enrichment_started_at is None:
                return True

            from sqlalchemy import or_

            writes_since = (
                await session.execute(
                    select(func.count(PipelineRun.id)).where(
                        PipelineRun.dataset_id == dataset_id,
                        PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
                        PipelineRun.pipeline_name.in_(list(write_pipeline_names)),
                        PipelineRun.created_at > last_enrichment_started_at,
                        # The stamped run's own later pipelines (stage 8's memify,
                        # stage 9) start after the stamp; they are the enrichment,
                        # not writes it missed. A deeper-nested child slipping past
                        # this errs toward "changed" — extra work, never lost work.
                        or_(
                            PipelineRun.parent_operation_id.is_(None),
                            PipelineRun.parent_operation_id != stamped_operation_id,
                        ),
                    )
                )
            ).scalar_one()

            return bool(writes_since)
    except Exception as error:
        logger.debug(
            "improve: change check could not decide, running the stage: %s", error, exc_info=True
        )
        return True
