import asyncio

from cognee.modules.pipelines.tasks import Task, TaskConfig, TaskExecutionInfo
from cognee.modules.pipelines.operations.run_tasks_base import run_tasks_base


async def run_and_check_tasks():
    def number_generator(num, context=None):
        for i in range(num):
            yield i + 1

    async def add_one(num, context=None):
        yield num + 1

    async def add_two(num, context=None):
        yield num + 2

    async def multiply_by_two(num1, num2, context=None):
        yield num1 * 2
        yield num2 * 2

    index = 0
    expected_results = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 24]

    async for task_run_info in run_tasks_base(
        [
            Task(number_generator),
            Task(add_one, task_config=TaskConfig(needs=[number_generator])),
            Task(add_two, task_config=TaskConfig(needs=[number_generator])),
            Task(multiply_by_two, task_config=TaskConfig(needs=[add_one, add_two])),
        ],
        data=10,
    ):
        if isinstance(task_run_info, TaskExecutionInfo):
            assert task_run_info.result == expected_results[index], (
                f"at {index = }: {task_run_info.result = } != {expected_results[index] = }"
            )
            index += 1


def test_run_tasks():
    asyncio.run(run_and_check_tasks())


if __name__ == "__main__":
    test_run_tasks()
