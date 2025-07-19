try:
    import modal
except ModuleNotFoundError:
    modal = None

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import (
    PipelineRunStarted,
    PipelineRunYield,
    PipelineRunCompleted,
)
from cognee.modules.pipelines.operations import log_pipeline_run_start, log_pipeline_run_complete
from cognee.modules.pipelines.utils.generate_pipeline_id import generate_pipeline_id
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

from .run_tasks_with_telemetry import run_tasks_with_telemetry


logger = get_logger("run_tasks_distributed()")


if modal:
    from distributed.app import app
    from distributed.modal_image import image

    @app.function(
        retries=3,
        image=image,
        timeout=86400,
        max_containers=50,
        secrets=[modal.Secret.from_name("distributed_cognee")],
    )
    async def run_tasks_on_modal(tasks, data_item, user, pipeline_name, context):
        pipeline_run = run_tasks_with_telemetry(tasks, data_item, user, pipeline_name, context)

        run_info = None

        async for pipeline_run_info in pipeline_run:
            run_info = pipeline_run_info

        return run_info


async def run_tasks_distributed(tasks, dataset_id, data, user, pipeline_name, context):
    if not user:
        user = get_default_user()

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

    data_count = len(data) if isinstance(data, list) else 1

    arguments = [
        [tasks] * data_count,
        [[data_item] for data_item in data[:data_count]] if data_count > 1 else [data],
        [user] * data_count,
        [pipeline_name] * data_count,
        [context] * data_count,
    ]

    async for result in run_tasks_on_modal.map.aio(*arguments):
        logger.info(f"Received result: {result}")

        yield PipelineRunYield(
            pipeline_run_id=pipeline_run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            payload=result,
        )

    await log_pipeline_run_complete(pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data)

    yield PipelineRunCompleted(
        pipeline_run_id=pipeline_run_id,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
    )
