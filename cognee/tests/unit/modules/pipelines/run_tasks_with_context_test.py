import asyncio

import cognee
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.operations.run_tasks_base import run_tasks_base
from cognee.infrastructure.databases.relational import create_db_and_tables


async def run_and_check_tasks():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # task_1 and task_3 accept ctx: PipelineContext explicitly.
    # task_2 does not -- the pipeline skips injection for it.
    def task_1(num, ctx: PipelineContext = None):
        return num + len(ctx.pipeline_name)

    def task_2(num):
        return num * 2

    def task_3(num, ctx: PipelineContext = None):
        return num ** len(ctx.pipeline_name)

    await create_db_and_tables()
    user = await get_default_user()

    # pipeline_name = "testing" has length 7
    ctx = PipelineContext(pipeline_name="testing")

    pipeline = run_tasks_base(
        [
            Task(task_1),
            Task(task_2),
            Task(task_3),
        ],
        data=5,
        user=user,
        ctx=ctx,
    )

    # task_1: 5 + 7 = 12, task_2: 12 * 2 = 24, task_3: 24 ^ 7 = 4586471424
    final_result = 4586471424
    async for result in pipeline:
        assert result == final_result


def test_run_tasks():
    asyncio.run(run_and_check_tasks())


if __name__ == "__main__":
    test_run_tasks()
