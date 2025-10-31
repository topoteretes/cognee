try:
    import modal
except ModuleNotFoundError:
    modal = None

from typing import Any, List, Optional
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Dataset
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.pipelines.models import (
    PipelineRunStarted,
    PipelineRunCompleted,
    PipelineRunErrored,
)
from cognee.modules.pipelines.operations import (
    log_pipeline_run_start,
    log_pipeline_run_complete,
    log_pipeline_run_error,
)
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.modules.pipelines.exceptions import PipelineRunFailedError
from cognee.tasks.ingestion import resolve_data_directories
from .run_tasks_data_item import run_tasks_data_item

logger = get_logger("run_tasks_distributed()")

if modal:
    import os
    from distributed.app import app
    from distributed.modal_image import image

    secret_name = os.environ.get("MODAL_SECRET_NAME", "distributed_cognee")

    @app.function(
        retries=3,
        image=image,
        timeout=86400,
        max_containers=50,
        secrets=[modal.Secret.from_name(secret_name)],
    )
    async def run_tasks_on_modal(
        data_item,
        dataset: Dataset,
        tasks: List[Task],
        pipeline_name: str,
        pipeline_id: str,
        pipeline_run_id: str,
        context: Optional[dict],
        user: User,
        incremental_loading: bool,
    ):
        """
        Wrapper that runs the run_tasks_data_item function.
        This is the function/code that runs on modal executor and produces the graph/vector db objects
        """
        from cognee.infrastructure.databases.relational import get_relational_engine

        result = await run_tasks_data_item(
            data_item=data_item,
            dataset=dataset,
            tasks=tasks,
            pipeline_name=pipeline_name,
            pipeline_id=pipeline_id,
            pipeline_run_id=pipeline_run_id,
            context={
                **(context or {}),
                "user": user,
                "data": data_item,
                "dataset": dataset,
            },
            user=user,
            incremental_loading=incremental_loading,
        )

        return result


async def run_tasks_distributed(
    tasks: List[Task],
    dataset_id: UUID,
    data: Optional[List[Any]] = None,
    user: Optional[User] = None,
    pipeline_name: str = "unknown_pipeline",
    context: Optional[dict] = None,
    incremental_loading: bool = False,
    data_per_batch: int = 20,
):
    if not user:
        user = await get_default_user()

    async with get_relational_engine().get_async_session() as session:
        from cognee.modules.data.models import Dataset

        dataset = await session.get(Dataset, dataset_id)

    pipeline_id: UUID = generate_pipeline_id(user.id, dataset.id, pipeline_name)
    pipeline_run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset.id, data)
    pipeline_run_id: UUID = pipeline_run.pipeline_run_id

    yield PipelineRunStarted(
        pipeline_run_id=pipeline_run_id,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        payload=data,
    )

    try:
        if not isinstance(data, list):
            data = [data]

        data = await resolve_data_directories(data)

        number_of_data_items = len(data) if isinstance(data, list) else 1

        data_item_tasks = [
            data,
            [dataset] * number_of_data_items,
            [tasks] * number_of_data_items,
            [pipeline_name] * number_of_data_items,
            [pipeline_id] * number_of_data_items,
            [pipeline_run_id] * number_of_data_items,
            [context] * number_of_data_items,
            [user] * number_of_data_items,
            [incremental_loading] * number_of_data_items,
        ]

        results = []
        async for result in run_tasks_on_modal.map.aio(*data_item_tasks):
            if not result:
                continue
            results.append(result)

        # Remove skipped results
        results = [r for r in results if r]

        # If any data item failed, raise PipelineRunFailedError
        errored = [
            r
            for r in results
            if r and r.get("run_info") and isinstance(r["run_info"], PipelineRunErrored)
        ]
        if errored:
            raise PipelineRunFailedError("Pipeline run failed. Data item could not be processed.")

        await log_pipeline_run_complete(
            pipeline_run_id, pipeline_id, pipeline_name, dataset.id, data
        )

        yield PipelineRunCompleted(
            pipeline_run_id=pipeline_run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            data_ingestion_info=results,
        )

    except Exception as error:
        await log_pipeline_run_error(
            pipeline_run_id, pipeline_id, pipeline_name, dataset.id, data, error
        )

        yield PipelineRunErrored(
            pipeline_run_id=pipeline_run_id,
            payload=repr(error),
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            data_ingestion_info=locals().get("results"),
        )

        if not isinstance(error, PipelineRunFailedError):
            raise
