import os
import cognee.modules.ingestion as ingestion

from uuid import UUID
from typing import Any
from functools import wraps

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.operations.run_tasks_distributed import run_tasks_distributed
from cognee.modules.users.models import User
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


@override_run_tasks(run_tasks_distributed)
async def run_tasks(
    tasks: list[Task],
    dataset_id: UUID,
    data: Any = None,
    user: User = None,
    pipeline_name: str = "unknown_pipeline",
    context: dict = None,
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
    data_items_pipeline_run_info = {}
    ingestion_error = None
    try:
        if not isinstance(data, list):
            data = [data]
        data = await resolve_data_directories(data)

        # TODO: Convert to async gather task instead of for loop (just make sure it can work there were some issues when async gathering datasets)
        for data_item in data:
            file_path = await save_data_item_to_storage(data_item, dataset.name)
            # Ingest data and add metadata
            with open_data_file(file_path, s3fs=fs) as file:
                classified_data = ingestion.classify(file, s3fs=fs)
                # data_id is the hash of file contents + owner id to avoid duplicate data
                data_id = ingestion.identify(classified_data, user)

            try:
                async for result in run_tasks_with_telemetry(
                    tasks=tasks,
                    data=data_item,
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

                data_items_pipeline_run_info[data_id] = {
                    "run_info": PipelineRunCompleted(
                        pipeline_run_id=pipeline_run_id,
                        dataset_id=dataset.id,
                        dataset_name=dataset.name,
                    ),
                    "data_id": data_id,
                }

            except Exception as error:
                # Temporarily swallow error and try to process rest of documents first, then re-raise error at end of data ingestion pipeline
                ingestion_error = error
                logger.error(
                    f"Exception caught while processing data: {error}.\n Data processing failed for data item: {data_item}."
                )

                data_items_pipeline_run_info = {
                    "run_info": PipelineRunErrored(
                        pipeline_run_id=pipeline_run_id,
                        payload=error,
                        dataset_id=dataset.id,
                        dataset_name=dataset.name,
                    ),
                    "data_id": data_id,
                }

        # re-raise error found during data ingestion
        if ingestion_error:
            raise ingestion_error

        await log_pipeline_run_complete(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data
        )

        yield PipelineRunCompleted(
            pipeline_run_id=pipeline_run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            data_ingestion_info=data_items_pipeline_run_info,
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
            data_ingestion_info=data_items_pipeline_run_info,
        )

        raise error
