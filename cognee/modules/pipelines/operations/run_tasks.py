import os

import asyncio
from functools import wraps
from typing import Any, List, Optional
from uuid import UUID

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.operations.run_tasks_distributed import run_tasks_distributed
from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.modules.pipelines.exceptions import PipelineRunFailedError
from cognee.tasks.ingestion import resolve_data_directories
from cognee.modules.pipelines.models import PipelineContext
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
    data: Optional[List[Any]] = None,
    user: Optional[User] = None,
    pipeline_name: str = "unknown_pipeline",
    incremental_loading: bool = False,
    data_per_batch: int = 20,
    extras: Optional[dict] = None,
):
    if not user:
        user = await get_default_user()

    async with get_relational_engine().get_async_session() as session:
        from cognee.modules.data.models import Dataset

        dataset = await session.get(Dataset, dataset_id)

    pipeline_id = generate_pipeline_id(user.id, dataset.id, pipeline_name)
    pipeline_run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset.id, data)
    pipeline_run_id = pipeline_run.pipeline_run_id

    yield PipelineRunStarted(
        pipeline_run_id=pipeline_run_id,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        payload=data,
    )

    # Note: Setting of global context has to be done after yielding PipelineRunStarted due to running in
    #       background mode requiring the pipeline run started yield.
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        try:
            if not isinstance(data, list):
                data = [data]

            if incremental_loading:
                data = await resolve_data_directories(data)

            # Semaphore-based concurrency: all items are scheduled at once,
            # but at most data_per_batch run concurrently at any time.
            semaphore = asyncio.Semaphore(data_per_batch)

            async def _run_item(data_item):
                async with semaphore:
                    return await run_tasks_data_item(
                        data_item,
                        dataset,
                        tasks,
                        pipeline_name,
                        pipeline_id,
                        pipeline_run_id,
                        PipelineContext(
                            user=user,
                            data_item=data_item,
                            dataset=dataset,
                            pipeline_name=pipeline_name,
                            extras=extras if isinstance(extras, dict) else {},
                        ),
                        user,
                        incremental_loading,
                    )

            gathered = await asyncio.gather(
                *[asyncio.create_task(_run_item(item)) for item in data],
                return_exceptions=True,
            )

            # Separate successes from unhandled exceptions
            results = []
            for i, result in enumerate(gathered):
                if isinstance(result, BaseException):
                    logger.error(f"Item {i} failed: {result}", exc_info=result)
                    results.append(
                        {
                            "run_info": PipelineRunErrored(
                                pipeline_run_id=pipeline_run_id,
                                payload=repr(result),
                                dataset_id=dataset.id,
                                dataset_name=dataset.name,
                            ),
                        }
                    )
                elif result:
                    results.append(result)

            # If any data item could not be processed propagate error
            errored_results = [
                result for result in results if isinstance(result["run_info"], PipelineRunErrored)
            ]
            if errored_results:
                raise PipelineRunFailedError(
                    message="Pipeline run failed. Data item could not be processed."
                )

            await log_pipeline_run_complete(
                pipeline_run_id, pipeline_id, pipeline_name, dataset.id, data
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
                pipeline_run_id, pipeline_id, pipeline_name, dataset.id, data, error
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
