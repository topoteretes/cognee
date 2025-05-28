import asyncio

import cognee
from cognee.modules.pipelines.operations.run_tasks import run_tasks_base
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.infrastructure.databases.relational import create_db_and_tables


async def run_and_check_tasks():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    def number_generator(num):
        for i in range(num):
            yield i + 1

    async def add_one(nums):
        for num in nums:
            yield num + 1

    async def multiply_by_two(nums):
        yield nums[0] * 2

    async def add_one_single(nums):
        yield nums[0] + 1

    await create_db_and_tables()
    user = await get_default_user()

    pipeline = run_tasks_base(
        [
            Task(number_generator),
            Task(add_one, task_config={"batch_size": 5}),
            Task(multiply_by_two, task_config={"batch_size": 1}),
            Task(add_one_single),
        ],
        data=10,
        user=user,
    )

    results = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    index = 0
    async for result in pipeline:
        assert result[0] == results[index], f"at {index = }: {result = } != {results[index] = }"
        index += 1


def test_run_tasks():
    asyncio.run(run_and_check_tasks())


if __name__ == "__main__":
    test_run_tasks()
