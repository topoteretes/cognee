import pytest

from cognee.modules.pipelines.operations.run_parallel import run_tasks_parallel
from cognee.modules.pipelines.tasks.task import Task


@pytest.mark.asyncio
async def test_run_tasks_parallel_returns_single_task_result():
    async def return_value(value):
        return value

    parallel_task = run_tasks_parallel([Task(return_value)])

    assert await parallel_task.run("important-value") == "important-value"


@pytest.mark.asyncio
async def test_run_tasks_parallel_returns_last_result_for_multiple_tasks():
    async def first(value):
        return f"first:{value}"

    async def second(value):
        return f"second:{value}"

    parallel_task = run_tasks_parallel([Task(first), Task(second)])

    assert await parallel_task.run("value") == "second:value"


@pytest.mark.asyncio
async def test_run_tasks_parallel_returns_empty_list_without_tasks():
    parallel_task = run_tasks_parallel([])

    assert await parallel_task.run("unused") == []
