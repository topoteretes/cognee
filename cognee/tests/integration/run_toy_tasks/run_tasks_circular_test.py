import pytest
import asyncio

from cognee.modules.pipelines.tasks import Task, TaskConfig
from cognee.modules.pipelines.exceptions import WrongTaskOrderException
from cognee.modules.pipelines.operations.run_tasks_v2 import run_tasks_base


async def run_and_check_tasks():
    def number_generator(num, context={}):
        for i in range(num):
            yield i + 1

    async def add_one(num, context={}):
        yield num + 1

    async def multiply_by_two(num, context={}):
        yield num * 2

    index = 0
    expected_results = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22]

    with pytest.raises(
        WrongTaskOrderException,
        match="1/3 tasks executed. You likely have some disconneted tasks or circular dependency.",
    ):
        async for task_run_info in run_tasks_base(
            [
                Task(number_generator),
                Task(add_one, task_config=TaskConfig(inputs=[number_generator, multiply_by_two])),
                Task(multiply_by_two, task_config=TaskConfig(inputs=[add_one])),
            ],
            data=10,
        ):
            if not task_run_info.is_done:
                assert task_run_info.result == expected_results[index], (
                    f"at {index = }: {task_run_info.result = } != {expected_results[index] = }"
                )
                index += 1


def test_run_tasks():
    asyncio.run(run_and_check_tasks())


if __name__ == "__main__":
    test_run_tasks()
