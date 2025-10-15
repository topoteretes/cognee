import os

import asyncio
from uuid import UUID
from typing import Any, List
from functools import wraps

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.operations.run_tasks_distributed import run_tasks_distributed
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.modules.pipelines.exceptions import PipelineRunFailedError
from cognee.tasks.ingestion import resolve_data_directories
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
)
from cognee.modules.pipelines.operations import (
    log_pipeline_run_start,
    log_pipeline_run_complete,
    log_pipeline_run_error,
)
from .run_tasks_data_item import run_tasks_data_item
from ..tasks.task import Task


logger = get_logger("run_tasks(tasks: [Task], data)")


def override_run_tasks(new_gen):
    def decorator(original_gen):
        @wraps(original_gen)
        async def wrapper(*args, distributed=None, **kwargs):
            default_distributed_value = os.getenv("COGNEE_DISTRIBUTED", "False").lower() == "true"
            distributed = default_distributed_value if distributed is None else distributed

            if distributed:
                async for run_info in new_gen(*args, **kwargs):
                    yield run_info
            else:
                async for run_info in original_gen(*args, **kwargs):
                    yield run_info

        return wrapper

    return decorator


@override_run_tasks(run_tasks_distributed)
async def run_tasks(
    tasks: List[Task],
    dataset_id: UUID,
    data: List[Any] = None,
    user: User = None,
    pipeline_name: str = "unknown_pipeline",
    context: dict = None,
    incremental_loading: bool = False,
    data_per_batch: int = 20,
):
    if not user:
        user = await get_default_user()

    # Get Dataset object
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        from cognee.modules.data.models import Dataset

        dataset = await session.get(Dataset, dataset_id)

    pipeline_id = generate_pipeline_id(user.id, dataset.id, pipeline_name)
    pipeline_run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset_id, data)
    pipeline_run_id = pipeline_run.pipeline_run_id

    yield PipelineRunStarted(
        pipeline_run_id=pipeline_run_id,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        payload=data,
    )

    try:
        if not isinstance(data, list):
            data = [data]

        if incremental_loading:
            data = await resolve_data_directories(data)

        # Create and gather batches of async tasks of data items that will run the pipeline for the data item
        results = []
        for start in range(0, len(data), data_per_batch):
            data_batch = data[start : start + data_per_batch]

            data_item_tasks = [
                asyncio.create_task(
                    run_tasks_data_item(
                        data_item,
                        dataset,
                        tasks,
                        pipeline_name,
                        pipeline_id,
                        pipeline_run_id,
                        context,
                        user,
                        incremental_loading,
                    )
                )
                for data_item in data_batch
            ]

            results.extend(await asyncio.gather(*data_item_tasks))

        # Remove skipped data items from results
        results = [result for result in results if result]

        # If any data item could not be processed propagate error
        errored_results = [
            result for result in results if isinstance(result["run_info"], PipelineRunErrored)
        ]
        if errored_results:
            raise PipelineRunFailedError(
                message="Pipeline run failed. Data item could not be processed."
            )

        await log_pipeline_run_complete(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data
        )

        yield PipelineRunCompleted(
            pipeline_run_id=pipeline_run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            data_ingestion_info=results,
        )

        graph_engine = await get_graph_engine()
        if hasattr(graph_engine, "push_to_s3"):
            await graph_engine.push_to_s3()

        relational_engine = get_relational_engine()
        if hasattr(relational_engine, "push_to_s3"):
            await relational_engine.push_to_s3()

    except Exception as error:
        await log_pipeline_run_error(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data, error
        )

        yield PipelineRunErrored(
            pipeline_run_id=pipeline_run_id,
            payload=repr(error),
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            data_ingestion_info=locals().get(
                "results"
            ),  # Returns results if they exist or returns None
        )

        # In case of error during incremental loading of data just let the user know the pipeline Errored, don't raise error
        if not isinstance(error, PipelineRunFailedError):
            raise error
