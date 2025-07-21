import os
import cognee.modules.ingestion as ingestion

import asyncio
from uuid import UUID
from typing import Any
from functools import wraps
from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.operations.run_tasks_distributed import run_tasks_distributed
from cognee.modules.users.models import User
from cognee.modules.data.models import Data
from cognee.modules.ingestion.methods import get_s3_fs, open_data_file
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.tasks.ingestion import save_data_item_to_storage, resolve_data_directories
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
    PipelineRunYield,
)

from cognee.modules.pipelines.operations import (
    log_pipeline_run_start,
    log_pipeline_run_complete,
    log_pipeline_run_error,
)
from .run_tasks_with_telemetry import run_tasks_with_telemetry
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


# TODO: Check if we should split task_per_data_generator into two functions one for regular and one for incremental loading instead of if statements
async def run_tasks_per_data_generator(
    data_item,
    dataset,
    tasks,
    pipeline_name,
    pipeline_id,
    pipeline_run_id,
    context,
    fs,
    user,
    incremental_loading,
):
    db_engine = get_relational_engine()
    # If incremental_loading of data is set to True don't process documents already processed by pipeline
    if incremental_loading:
        # If data is being added to Cognee for the first time calculate the id of the data
        if not isinstance(data_item, Data):
            file_path = await save_data_item_to_storage(data_item, dataset.name)
            # Ingest data and add metadata
            with open_data_file(file_path, s3fs=fs) as file:
                classified_data = ingestion.classify(file, s3fs=fs)
                # data_id is the hash of file contents + owner id to avoid duplicate data
                data_id = ingestion.identify(classified_data, user)
        else:
            # If data was already processed by Cognee get data id
            data_id = data_item.id

        # Check pipeline status, if Data already processed for pipeline before skip current processing
        async with db_engine.get_async_session() as session:
            data_point = (
                await session.execute(select(Data).filter(Data.id == data_id))
            ).scalar_one_or_none()
            if data_point:
                if data_point.pipeline_status.get(pipeline_name) == "Completed":
                    return

    try:
        # Process data based on data_item and list of tasks
        async for result in run_tasks_with_telemetry(
            tasks=tasks,
            data=[data_item],
            user=user,
            pipeline_name=pipeline_id,
            context=context,
        ):
            yield PipelineRunYield(
                pipeline_run_id=pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                payload=result,
            )

        if incremental_loading:
            # Update pipeline status for Data element
            async with db_engine.get_async_session() as session:
                data_point = (
                    await session.execute(select(Data).filter(Data.id == data_id))
                ).scalar_one_or_none()
                data_point.pipeline_status[pipeline_name] = "Completed"
                await session.merge(data_point)
                await session.commit()

            yield {
                "run_info": PipelineRunCompleted(
                    pipeline_run_id=pipeline_run_id,
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                ),
                "data_id": data_id,
            }
        else:
            yield {
                "run_info": PipelineRunCompleted(
                    pipeline_run_id=pipeline_run_id,
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                )
            }

    except Exception as error:
        # Temporarily swallow error and try to process rest of documents first, then re-raise error at end of data ingestion pipeline
        logger.error(
            f"Exception caught while processing data: {error}.\n Data processing failed for data item: {data_item}."
        )
        if incremental_loading:
            yield {
                "run_info": PipelineRunErrored(
                    pipeline_run_id=pipeline_run_id,
                    payload=error,
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                ),
                "data_id": data_id,
            }
        else:
            yield {
                "run_info": PipelineRunErrored(
                    pipeline_run_id=pipeline_run_id,
                    payload=error,
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                )
            }


async def run_tasks_per_data(
    data_item,
    dataset,
    tasks,
    pipeline_name,
    pipeline_id,
    pipeline_run_id,
    context,
    fs,
    user,
    incremental_loading,
):
    # Go through async generator and return data item processing result. Result can be None when data item is skipped,
    # PipelineRunCompleted when processing was successful and PipelineRunErrored if there were issues
    result = None
    async for result in run_tasks_per_data_generator(
        data_item,
        dataset,
        tasks,
        pipeline_name,
        pipeline_id,
        pipeline_run_id,
        context,
        fs,
        user,
        incremental_loading,
    ):
        pass
    return result


@override_run_tasks(run_tasks_distributed)
async def run_tasks(
    tasks: list[Task],
    dataset_id: UUID,
    data: Any = None,
    user: User = None,
    pipeline_name: str = "unknown_pipeline",
    context: dict = None,
    incremental_loading: bool = True,
):
    if not user:
        user = get_default_user()

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

    fs = get_s3_fs()
    try:
        if not isinstance(data, list):
            data = [data]

        if incremental_loading:
            data = await resolve_data_directories(data)

        # Create async tasks per data item that will run the pipeline for the data item
        data_item_tasks = [
            asyncio.create_task(
                run_tasks_per_data(
                    data_item,
                    dataset,
                    tasks,
                    pipeline_name,
                    pipeline_id,
                    pipeline_run_id,
                    context,
                    fs,
                    user,
                    incremental_loading,
                )
            )
            for data_item in data
        ]
        results = await asyncio.gather(*data_item_tasks)
        # Remove skipped data items from results
        results = [result for result in results if result]

        # If any data item could not be processed propagate error
        errored_results = [result for result in results if isinstance(result, PipelineRunErrored)]
        if errored_results:
            raise errored_results[0]["run_info"].payload

        await log_pipeline_run_complete(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data
        )

        yield PipelineRunCompleted(
            pipeline_run_id=pipeline_run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            data_ingestion_info=results,
        )

    except Exception as error:
        await log_pipeline_run_error(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data, error
        )

        yield PipelineRunErrored(
            pipeline_run_id=pipeline_run_id,
            payload=error,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            data_ingestion_info=locals().get(
                "results"
            ),  # Returns results if they exist or returns None
        )

        raise error
