from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.operations import get_current_operation, get_operation_origin
from cognee.modules.pipelines.models import OperationOutcome, PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.utils import summarize_run_info_data
from cognee.modules.users.models import User


async def log_pipeline_run_complete(
    pipeline_run_id: UUID,
    pipeline_id: UUID,
    pipeline_name: str,
    dataset_id: UUID,
    data: Any,
    *,
    user: Optional[User] = None,
    started_at: Optional[datetime] = None,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
):
    data_info = summarize_run_info_data(data)
    enclosing_operation = get_current_operation()

    pipeline_run = PipelineRun(
        pipeline_run_id=pipeline_run_id,
        pipeline_name=pipeline_name,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
        dataset_id=dataset_id,
        run_info={
            "data": data_info,
        },
        user_id=user.id if user else None,
        tenant_id=getattr(user, "tenant_id", None) if user else None,
        operation_name=pipeline_name,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
        outcome=OperationOutcome.SUCCEEDED.value,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        origin=get_operation_origin(),
        parent_operation_id=enclosing_operation.operation_id if enclosing_operation else None,
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
