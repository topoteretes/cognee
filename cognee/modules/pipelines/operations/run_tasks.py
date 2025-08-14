import os

import asyncio
from uuid import UUID
from typing import Any, List
from functools import wraps
from sqlalchemy import select

import cognee.modules.ingestion as ingestion
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.operations.run_tasks_distributed import run_tasks_distributed
from cognee.modules.users.models import User
from cognee.modules.data.models import Data
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.modules.pipelines.exceptions import PipelineRunFailedError
from cognee.tasks.ingestion import save_data_item_to_storage, resolve_data_directories
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
    PipelineRunYield,
    PipelineRunAlreadyCompleted,
)
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus

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


@override_run_tasks(run_tasks_distributed)
async def run_tasks(
    tasks: List[Task],
    dataset_id: UUID,
    data: List[Any] = None,
    user: User = None,
    pipeline_name: str = "unknown_pipeline",
    context: dict = None,
    incremental_loading: bool = False,
):
    async def _run_tasks_data_item_incremental(
        data_item,
        dataset,
        tasks,
        pipeline_name,
        pipeline_id,
        pipeline_run_id,
        context,
        user,
    ):
        db_engine = get_relational_engine()
        # If incremental_loading of data is set to True don't process documents already processed by pipeline
        # If data is being added to Cognee for the first time calculate the id of the data
        if not isinstance(data_item, Data):
            file_path = await save_data_item_to_storage(data_item)
            # Ingest data and add metadata
            async with open_data_file(file_path) as file:
                classified_data = ingestion.classify(file)
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
                if (
                    data_point.pipeline_status.get(pipeline_name, {}).get(str(dataset.id))
                    == DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
                ):
                    yield {
                        "run_info": PipelineRunAlreadyCompleted(
                            pipeline_run_id=pipeline_run_id,
                            dataset_id=dataset.id,
                            dataset_name=dataset.name,
                        ),
                        "data_id": data_id,
                    }
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

            # Update pipeline status for Data element
            async with db_engine.get_async_session() as session:
                data_point = (
                    await session.execute(select(Data).filter(Data.id == data_id))
                ).scalar_one_or_none()
                data_point.pipeline_status[pipeline_name] = {
                    str(dataset.id): DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
                }
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

        except Exception as error:
            # Temporarily swallow error and try to process rest of documents first, then re-raise error at end of data ingestion pipeline
            logger.error(
                f"Exception caught while processing data: {error}.\n Data processing failed for data item: {data_item}."
            )
            yield {
                "run_info": PipelineRunErrored(
                    pipeline_run_id=pipeline_run_id,
                    payload=repr(error),
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                ),
                "data_id": data_id,
            }

            if os.getenv("RAISE_INCREMENTAL_LOADING_ERRORS", "true").lower() == "true":
                raise error

    async def _run_tasks_data_item_regular(
        data_item,
        dataset,
        tasks,
        pipeline_id,
        pipeline_run_id,
        context,
        user,
    ):
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

        yield {
            "run_info": PipelineRunCompleted(
                pipeline_run_id=pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
            )
        }

    async def _run_tasks_data_item(
        data_item,
        dataset,
        tasks,
        pipeline_name,
        pipeline_id,
        pipeline_run_id,
        context,
        user,
        incremental_loading,
    ):
        # Go through async generator and return data item processing result. Result can be PipelineRunAlreadyCompleted when data item is skipped,
        # PipelineRunCompleted when processing was successful and PipelineRunErrored if there were issues
        result = None
        if incremental_loading:
            async for result in _run_tasks_data_item_incremental(
                data_item=data_item,
                dataset=dataset,
                tasks=tasks,
                pipeline_name=pipeline_name,
                pipeline_id=pipeline_id,
                pipeline_run_id=pipeline_run_id,
                context=context,
                user=user,
            ):
                pass
        else:
            async for result in _run_tasks_data_item_regular(
                data_item=data_item,
                dataset=dataset,
                tasks=tasks,
                pipeline_id=pipeline_id,
                pipeline_run_id=pipeline_run_id,
                context=context,
                user=user,
            ):
                pass

        return result

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

        # TODO: Return to using async.gather for data items after Cognee release
        # # Create async tasks per data item that will run the pipeline for the data item
        # data_item_tasks = [
        #     asyncio.create_task(
        #         _run_tasks_data_item(
        #             data_item,
        #             dataset,
        #             tasks,
        #             pipeline_name,
        #             pipeline_id,
        #             pipeline_run_id,
        #             context,
        #             user,
        #             incremental_loading,
        #         )
        #     )
        #     for data_item in data
        # ]
        # results = await asyncio.gather(*data_item_tasks)
        # # Remove skipped data items from results
        # results = [result for result in results if result]

        ### TEMP sync data item handling
        results = []
        # Run the pipeline for each data_item sequentially, one after the other
        for data_item in data:
            result = await _run_tasks_data_item(
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

            # Skip items that returned a false-y value
            if result:
                results.append(result)
        ### END

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
