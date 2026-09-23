from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.operations import get_operation_origin, scrub_error_message
from cognee.modules.operations.usage_accumulator import get_parent_run_id
from cognee.modules.pipelines.models import OperationOutcome, PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.utils import summarize_run_info_data
from cognee.modules.users.models import User


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
    data_info: Any = None,
    origin: str | None = None,
    parent_operation_id: UUID | None = None,
):
    """Append the ERRORED row for a run.

    ``data_info``, ``origin`` and ``parent_operation_id`` default to what this
    call's own context provides. A writer closing a run on behalf of a process
    that is gone (startup recovery) passes the STARTED row's values instead, so
    the ERRORED row describes the run that died, not the process closing it.
    """
    if data_info is None:
        data_info = summarize_run_info_data(data)

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
        origin=origin if origin is not None else get_operation_origin(),
        # This writer runs inside the pipeline's own parent_run_scope —
        # exclude it so the row parents to the next enclosing run.
        parent_operation_id=(
            parent_operation_id
            if parent_operation_id is not None
            else get_parent_run_id(excluding=pipeline_run_id)
        ),
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
