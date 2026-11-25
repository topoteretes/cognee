from datetime import datetime, timezone
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.operations import get_current_session_ids, get_operation_origin
from cognee.modules.operations.usage_accumulator import get_parent_run_id
from cognee.modules.pipelines.models import OperationOutcome, PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.operations.run_sessions import add_run_sessions
from cognee.modules.users.models import User


async def log_pipeline_run_complete(
    pipeline_run_id: UUID,
    pipeline_id: UUID,
    pipeline_name: str,
    dataset_id: UUID,
    *,
    user: User | None = None,
    started_at: datetime | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    embedding_tokens: int | None = None,
):
    session_ids = get_current_session_ids()

    pipeline_run = PipelineRun(
        pipeline_run_id=pipeline_run_id,
        pipeline_name=pipeline_name,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        dataset_id=dataset_id,
        run_info={},
        user_id=user.id if user else None,
        tenant_id=getattr(user, "tenant_id", None) if user else None,
        operation_name=pipeline_name,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
        outcome=OperationOutcome.SUCCEEDED.value,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        embedding_tokens=embedding_tokens,
        origin=get_operation_origin(),
        # This writer runs inside the pipeline's own parent_run_scope —
        # exclude it so the row parents to the next enclosing run.
        parent_operation_id=get_parent_run_id(excluding=pipeline_run_id),
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        add_run_sessions(session, pipeline_run, session_ids)
        await session.commit()

    return pipeline_run
