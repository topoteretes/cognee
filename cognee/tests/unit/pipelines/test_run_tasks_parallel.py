"""Tests for run_tasks_parallel's return value."""

import pytest

from cognee.modules.pipelines.operations.run_parallel import run_tasks_parallel
from cognee.modules.pipelines.tasks.task import Task


async def echo(value):
    return value


@pytest.mark.asyncio
async def test_single_task_returns_its_result():
    """A one-task list used to return [] because the guard tested len(results) > 1."""
    parallel = run_tasks_parallel([Task(echo)])

    assert await parallel.run("important-value") == "important-value"


@pytest.mark.asyncio
async def test_multiple_tasks_return_last_result():
    parallel = run_tasks_parallel([Task(echo), Task(echo)])

    assert await parallel.run("value") == "value"


@pytest.mark.asyncio
async def test_no_tasks_returns_empty_list():
    parallel = run_tasks_parallel([])

    assert await parallel.run("value") == []
