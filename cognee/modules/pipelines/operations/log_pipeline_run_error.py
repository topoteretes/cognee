from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.operations import get_operation_origin, scrub_error_message
from cognee.modules.pipelines.operations.classify_pipeline_error import classify_pipeline_error
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
    user: Optional[User] = None,
    started_at: Optional[datetime] = None,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
):
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
            # Semantic bucket (auth/provider_quota/schema/config/loader/
            # db_init/unknown) — the taxonomy activation analytics groups by.
            "error_category": classify_pipeline_error(e).category,
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
