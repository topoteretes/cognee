from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.operations import get_operation_origin, scrub_error_message
from cognee.modules.operations.usage_accumulator import get_parent_run_id
from cognee.modules.pipelines.methods import get_terminal_pipeline_run
from cognee.modules.pipelines.models import OperationOutcome, PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.utils import summarize_run_info_data
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("pipelines.log_pipeline_run_error")


async def log_pipeline_run_error(
    pipeline_run_id: UUID,
    pipeline_id: UUID,
    pipeline_name: str,
    dataset_id: UUID,
    data: Any,
    e: Exception,
    *,
    user: User | None = None,
    started_at: datetime | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    data_info: Any | None = None,
):
    # For a caller that already holds a summarized value. Startup recovery
    # closes a run whose STARTED row carries one, and summarizing a summary
    # stringifies the list of data ids and re-truncates an already-truncated
    # preview against the wrong character count.
    if data_info is None:
        data_info = summarize_run_info_data(data)

    existing_terminal_run = await get_terminal_pipeline_run(pipeline_run_id)
    if existing_terminal_run is not None:
        # A terminal row for this run already exists — most often startup
        # recovery closed it as ERRORED while the process that started it was
        # still running and has now reached its own error path. Writing a
        # second terminal row would not correct the first one, it would just
        # leave two contradictory events for the same run; keep the one that
        # is already there.
        # The alternative framing ("two contradictory events") undersells
        # what is lost here: if this run genuinely failed on its own after
        # recovery pre-closed it, the real cause never reaches the row.
        # Logging it is the only trace that survives.
        logger.warning(
            "Skipping duplicate terminal write for pipeline run %s: already %s. "
            "Discarded error: %s: %s",
            pipeline_run_id,
            existing_terminal_run.status,
            type(e).__name__,
            scrub_error_message(e),
        )
        return existing_terminal_run

    pipeline_run = PipelineRun(
        pipeline_run_id=pipeline_run_id,
        pipeline_name=pipeline_name,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_ERRORED,
        dataset_id=dataset_id,
        run_info={
            "data": data_info,
            # Scrubbed like error_message — persisting the raw text here would
            # defeat the redaction (and run_info growth is capped, COG-5359).
            "error": scrub_error_message(e),
        },
        user_id=user.id if user else None,
        tenant_id=getattr(user, "tenant_id", None) if user else None,
        operation_name=pipeline_name,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
        outcome=OperationOutcome.FAILED.value,
        error_class=type(e).__name__,
        error_message=scrub_error_message(e),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        origin=get_operation_origin(),
        # This writer runs inside the pipeline's own parent_run_scope —
        # exclude it so the row parents to the next enclosing run.
        parent_operation_id=get_parent_run_id(excluding=pipeline_run_id),
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
