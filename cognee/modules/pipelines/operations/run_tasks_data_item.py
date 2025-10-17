"""
Data item processing functions for pipeline operations.

This module contains reusable functions for processing individual data items
within pipeline operations, supporting both incremental and regular processing modes.
"""

import os
from typing import Any, Dict, AsyncGenerator, Optional
from sqlalchemy import select

import cognee.modules.ingestion as ingestion
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.modules.data.models import Data, Dataset
from cognee.tasks.ingestion import save_data_item_to_storage
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunYield,
    PipelineRunAlreadyCompleted,
)
from cognee.modules.pipelines.models.DataItemStatus import DataItemStatus
from cognee.modules.pipelines.operations.run_tasks_with_telemetry import run_tasks_with_telemetry
from ..tasks.task import Task

logger = get_logger("run_tasks_data_item")


async def run_tasks_data_item_incremental(
    data_item: Any,
    dataset: Dataset,
    tasks: list[Task],
    pipeline_name: str,
    pipeline_id: str,
    pipeline_run_id: str,
    context: Optional[Dict[str, Any]],
    user: User,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Process a single data item with incremental loading support.

    This function handles incremental processing by checking if the data item
    has already been processed for the given pipeline and dataset. If it has,
    it skips processing and returns a completion status.

    Args:
        data_item: The data item to process
        dataset: The dataset containing the data item
        tasks: List of tasks to execute on the data item
        pipeline_name: Name of the pipeline
        pipeline_id: Unique identifier for the pipeline
        pipeline_run_id: Unique identifier for this pipeline run
        context: Optional context dictionary
        user: User performing the operation

    Yields:
        Dict containing run_info and data_id for each processing step
    """
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
            status_for_pipeline = data_point.pipeline_status.setdefault(pipeline_name, {})
            status_for_pipeline[str(dataset.id)] = DataItemStatus.DATA_ITEM_PROCESSING_COMPLETED
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


async def run_tasks_data_item_regular(
    data_item: Any,
    dataset: Dataset,
    tasks: list[Task],
    pipeline_id: str,
    pipeline_run_id: str,
    context: Optional[Dict[str, Any]],
    user: User,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Process a single data item in regular (non-incremental) mode.

    This function processes a data item without checking for previous processing
    status, executing all tasks on the data item.

    Args:
        data_item: The data item to process
        dataset: The dataset containing the data item
        tasks: List of tasks to execute on the data item
        pipeline_id: Unique identifier for the pipeline
        pipeline_run_id: Unique identifier for this pipeline run
        context: Optional context dictionary
        user: User performing the operation

    Yields:
        Dict containing run_info for each processing step
    """
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


async def run_tasks_data_item(
    data_item: Any,
    dataset: Dataset,
    tasks: list[Task],
    pipeline_name: str,
    pipeline_id: str,
    pipeline_run_id: str,
    context: Optional[Dict[str, Any]],
    user: User,
    incremental_loading: bool,
) -> Optional[Dict[str, Any]]:
    """
    Process a single data item, choosing between incremental and regular processing.

    This is the main entry point for data item processing that delegates to either
    incremental or regular processing based on the incremental_loading flag.

    Args:
        data_item: The data item to process
        dataset: The dataset containing the data item
        tasks: List of tasks to execute on the data item
        pipeline_name: Name of the pipeline
        pipeline_id: Unique identifier for the pipeline
        pipeline_run_id: Unique identifier for this pipeline run
        context: Optional context dictionary
        user: User performing the operation
        incremental_loading: Whether to use incremental processing

    Returns:
        Dict containing the final processing result, or None if processing was skipped
    """
    # Go through async generator and return data item processing result. Result can be PipelineRunAlreadyCompleted when data item is skipped,
    # PipelineRunCompleted when processing was successful and PipelineRunErrored if there were issues
    result = None
    if incremental_loading:
        async for result in run_tasks_data_item_incremental(
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
        async for result in run_tasks_data_item_regular(
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
