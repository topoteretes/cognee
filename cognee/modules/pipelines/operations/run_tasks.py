import json
from cognee.shared.logging_utils import get_logger
from uuid import UUID, uuid4

from typing import Any
from cognee.modules.pipelines.operations import (
    log_pipeline_run_start,
    log_pipeline_run_complete,
    log_pipeline_run_error,
)
from cognee.modules.settings import get_current_settings
from cognee.modules.users.models import User
from cognee.modules.data.models import Data
from cognee.shared.utils import send_telemetry
from uuid import uuid5, NAMESPACE_OID

from .run_tasks_base import run_tasks_base
from ..tasks.task import Task

from cognee.modules.data.methods.update_data_processing_status import update_data_processing_status
from cognee.modules.data.models import FileProcessingStatus

logger = get_logger("run_tasks(tasks: [Task], data)")

async def handle_data_processing_status(item, status: FileProcessingStatus):
    """
    Updates the processing status of a `Data` object in the database.

    Args:
        item: The data point to check and update.
        status (FileProcessingStatus): The new processing status to set.
    """

    # items can be Data or other types, so we check if it's an instance of Data
    if isinstance(item, Data):
        await update_data_processing_status(item.id, status)


async def run_tasks_with_telemetry(
    tasks: list[Task], data, user: User, pipeline_name: str, context: dict = None
):
    """
    Executes tasks for a pipeline with telemetry and status tracking.

    This function processes each data point individually, updating its processing 
    status before and after execution. It also sends telemetry events to track 
    the pipeline's progress and logs any errors encountered during execution.

    Args:
        tasks (list[Task]): 
            A list of Task objects defining the operations to be performed on each data point.
        data: 
            A collection of data points to be processed by the tasks.
        user (User): 
            The user initiating the pipeline.
        pipeline_name (str): 
            The name of the pipeline being executed.
        context (dict, optional): 
            Additional context or metadata to be passed to the tasks. Defaults to None.

    Yields:
        Any: 
            Results yielded by the `run_tasks_base` function for each data point.

    Raises:
        Exception: 
            If an error occurs during the execution of tasks for a data point, 
            the error is logged, the data point's status is updated to `ERROR`, 
            and the exception is re-raised.
    """

    config = get_current_settings()

    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent=1))

    try:
        logger.info("Pipeline run started: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Started",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
            }
            | config,
        )

        for item in data:
            await handle_data_processing_status(item, FileProcessingStatus.PROCESSING)

            try:
                async for result in run_tasks_base(tasks, [item], user, context):
                    yield result
                await handle_data_processing_status(item, FileProcessingStatus.PROCESSED)

            except Exception as error:
                await handle_data_processing_status(item, FileProcessingStatus.ERROR)
                logger.error(
                    "Error processing data point `%s` in pipeline `%s`: %s",
                    item.id,
                    pipeline_name,
                    str(error),
                    exc_info=True,
                )
                raise error

        logger.info("Pipeline run completed: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Completed",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
            },
        )
    except Exception as error:
        logger.error(
            "Pipeline run errored: `%s`\n%s\n",
            pipeline_name,
            str(error),
            exc_info=True,
        )
        send_telemetry(
            "Pipeline Run Errored",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
            }
            | config,
        )

        raise error


async def run_tasks(
    tasks: list[Task],
    dataset_id: UUID = uuid4(),
    data: Any = None,
    user: User = None,
    pipeline_name: str = "unknown_pipeline",
    context: dict = None,
):
    pipeline_id = uuid5(NAMESPACE_OID, pipeline_name)

    pipeline_run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset_id, data)

    yield pipeline_run
    pipeline_run_id = pipeline_run.pipeline_run_id

    try:
        async for _ in run_tasks_with_telemetry(
            tasks=tasks,
            data=data,
            user=user,
            pipeline_name=pipeline_id,
            context=context,
        ):
            pass

        yield await log_pipeline_run_complete(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data
        )

    except Exception as e:
        yield await log_pipeline_run_error(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data, e
        )
        raise e
