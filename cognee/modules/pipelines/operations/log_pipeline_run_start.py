from datetime import datetime, timezone
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.operations import get_current_session_ids, get_operation_origin
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.operations.run_sessions import add_run_sessions
from cognee.modules.pipelines.utils import generate_pipeline_run_id
from cognee.modules.users.models import User


async def log_pipeline_run_start(
    pipeline_id: UUID,
    pipeline_name: str,
    dataset_id: UUID,
    *,
    user: User | None = None,
    recovery_token: str | None = None,
):
    pipeline_run_id = generate_pipeline_run_id(pipeline_id, dataset_id)
    # Same reason as origin below: a session-scoped reader must see the run
    # while it is still in flight, not only once it has finished.
    session_ids = get_current_session_ids()

    pipeline_run = PipelineRun(
        pipeline_run_id=pipeline_run_id,
        pipeline_name=pipeline_name,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        dataset_id=dataset_id,
        run_info={"recovery_lock": recovery_token} if recovery_token else {},
        user_id=user.id if user else None,
        tenant_id=getattr(user, "tenant_id", None) if user else None,
        operation_name=pipeline_name,
        started_at=datetime.now(timezone.utc),
        # Stamped here too, not just on the terminal row: without it every
        # in-flight run is invisible to a query filtering pipeline_runs by
        # originating surface.
        origin=get_operation_origin(),
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        add_run_sessions(session, pipeline_run, session_ids)
        await session.commit()

    return pipeline_run
