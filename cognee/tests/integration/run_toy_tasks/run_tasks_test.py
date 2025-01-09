import asyncio

from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.tasks.Task import Task


async def run_and_check_tasks():
    def number_generator(num):
        for i in range(num):
            yield i + 1

    async def add_one(nums):
        for num in nums:
            yield num + 1

    async def multiply_by_two(num):
        yield num * 2

    async def add_one_single(num):
        yield num + 1

    pipeline = run_tasks(
        [
            Task(number_generator),
            Task(add_one, task_config={"batch_size": 5}),
            Task(multiply_by_two, task_config={"batch_size": 1}),
            Task(add_one_single),
        ],
        10,
        pipeline_name="test_run_tasks",
    )

    results = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    index = 0
    async for result in pipeline:
        assert result == results[index], f"at {index = }: {result = } != {results[index] = }"
        index += 1


def test_run_tasks():
    asyncio.run(run_and_check_tasks())
