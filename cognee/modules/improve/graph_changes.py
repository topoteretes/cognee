"""Has anything written to this dataset's graph since its last improve?

The triplet-enrichment stage's ``already_completed`` gate (plan Part 5.10,
stage 8). It reads ``pipeline_runs`` — indexed on ``dataset_id`` and
``created_at`` — and never a graph-wide node or edge count, which would put
the cost back into the gate.

Conservative by construction: when there is no prior completed improve for the
dataset, or the query cannot decide, the answer is "changed", so the stage runs.
"""

from typing import Iterable, Optional
from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("improve.graph_changes")

# Pipelines whose completed runs write nodes or edges into a dataset's graph.
# ``add_pipeline`` is absent on purpose: add() writes relational rows and files.
WRITE_PIPELINE_NAMES = (
    "cognify_pipeline",  # cognify() and update() (incremental attributes to it)
    "code_graph_pipeline",
    "memify_pipeline",  # every memify writer, the session-persist stages included
    "skills_ingest_pipeline",
    "skill_improvement_pipeline",
    "skill_runs_pipeline",
    "agentic_skill_runs_pipeline",
    "migration_import_pipeline",
)

IMPROVE_OPERATION_NAME = "improve"


async def has_graph_changed_since_last_improve(
    dataset_id: UUID,
    write_pipeline_names: Iterable[str] = WRITE_PIPELINE_NAMES,
) -> bool:
    """True unless a completed improve exists and no write pipeline completed after it."""
    try:
        from sqlalchemy import func, select

        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus

        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            last_improve_ended_at = (
                await session.execute(
                    select(func.max(PipelineRun.ended_at)).where(
                        PipelineRun.dataset_id == dataset_id,
                        PipelineRun.operation_name == IMPROVE_OPERATION_NAME,
                        PipelineRun.outcome == "succeeded",
                        PipelineRun.status.is_(None),
                    )
                )
            ).scalar_one_or_none()

            if last_improve_ended_at is None:
                return True

            writes_since = (
                await session.execute(
                    select(func.count(PipelineRun.id)).where(
                        PipelineRun.dataset_id == dataset_id,
                        PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
                        PipelineRun.pipeline_name.in_(list(write_pipeline_names)),
                        PipelineRun.created_at > last_improve_ended_at,
                    )
                )
            ).scalar_one()

            return bool(writes_since)
    except Exception as error:
        logger.debug("improve: change check could not decide, running the stage: %s", error)
        return True


def describe_change_check(changed: bool) -> Optional[str]:
    """Reason text for the stage result (``None`` when the stage runs)."""
    return None if changed else "no_writes_since_last_improve"
