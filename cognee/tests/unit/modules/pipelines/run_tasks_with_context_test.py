import asyncio

import cognee
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.operations.run_tasks import run_tasks_base
from cognee.infrastructure.databases.relational import create_db_and_tables


async def run_and_check_tasks():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    def task_1(num, context):
        return num + context

    def task_2(num):
        return num * 2

    def task_3(num, context):
        return num**context

    await create_db_and_tables()
    user = await get_default_user()

    pipeline = run_tasks_base(
        [
            Task(task_1),
            Task(task_2),
            Task(task_3),
        ],
        data=5,
        user=user,
        context=7,
    )

    final_result = 4586471424
    async for result in pipeline:
        assert result == final_result


def test_run_tasks():
    asyncio.run(run_and_check_tasks())


if __name__ == "__main__":
    test_run_tasks()
