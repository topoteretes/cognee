from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import delete

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus
from cognee.modules.pipelines.utils import generate_pipeline_run_id, summarize_run_info_data
from cognee.modules.users.models import User


async def log_pipeline_run_start(
    pipeline_id: UUID,
    pipeline_name: str,
    dataset_id: UUID,
    data: Any,
    *,
    user: User | None = None,
):
    data_info = summarize_run_info_data(data)

    pipeline_run_id = generate_pipeline_run_id(pipeline_id, dataset_id)

    pipeline_run = PipelineRun(
        pipeline_run_id=pipeline_run_id,
        pipeline_name=pipeline_name,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        dataset_id=dataset_id,
        run_info={
            "data": data_info,
        },
        user_id=user.id if user else None,
        tenant_id=getattr(user, "tenant_id", None) if user else None,
        operation_name=pipeline_name,
        started_at=datetime.now(timezone.utc),
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        # An INITIATED row is a reset marker saying "this pipeline may run
        # again" (see log_pipeline_run_initiated), not a record of work. The run
        # it unblocked is starting now, so the marker has served its purpose;
        # leaving it behind is what grew pipeline_runs by a permanent row that
        # no later row superseded, one or two per add.
        #
        # Keyed on dataset_id + pipeline_name, deliberately not on pipeline_id.
        # That pair is the unit every status reader works in (they partition on
        # exactly it, see get_pipeline_runs_by_dataset), so it is the unit the
        # marker speaks for. pipeline_id folds in the acting user, and a marker
        # written by one user for a dataset another user then cognifies would
        # never be matched, leaving behind the phantom this clears. It also
        # rides the existing (dataset_id, pipeline_name, created_at) index.
        await session.execute(
            delete(PipelineRun).where(
                PipelineRun.dataset_id == dataset_id,
                PipelineRun.pipeline_name == pipeline_name,
                PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_INITIATED,
            )
        )
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
