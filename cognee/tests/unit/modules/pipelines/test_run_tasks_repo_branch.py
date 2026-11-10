"""``run_tasks`` hands the add() call's branch to the repository URLs it resolves.

``run_tasks`` resolves GitHub/GitLab repository URLs itself, before any task list runs,
whenever ``data_cache`` or ``incremental_loading`` is on (the default for add()). With
either on, this read of ``extras["repo_branch"]`` is the only place the branch takes
effect: without it ``add(url, branch=...)`` still completes, and clones the default branch.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.run_tasks as run_tasks_module


@pytest.mark.asyncio
async def test_the_branch_in_extras_reaches_the_repository_resolution(monkeypatch, runner_plumbing):
    dataset = SimpleNamespace(id=uuid4(), name="repo_ds", owner_id=uuid4())
    runner_plumbing(run_tasks_module, dataset)
    monkeypatch.setattr(run_tasks_module, "validate_pipeline_tasks", lambda tasks: None)
    branches = []

    async def item_run(*args, **kwargs):
        return {"run_info": "ok"}

    async def resolve(data, user=None, dataset_id=None, branch=None):
        branches.append(branch)
        return data

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", item_run)
    monkeypatch.setattr(run_tasks_module, "resolve_data_directories", resolve)

    for extras in ({"repo_branch": "dev"}, None):
        async for _event in run_tasks_module.run_tasks(
            tasks=["TASKS"],
            dataset_id=dataset.id,
            data=["https://github.com/org/repo"],
            user=SimpleNamespace(id=uuid4(), tenant_id=None),
            pipeline_name="add_pipeline",
            extras=extras,
            data_cache=True,
        ):
            pass

    assert branches == ["dev", None]
