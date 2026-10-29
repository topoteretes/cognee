"""run_pipeline_per_dataset awaits ``before_run`` under the dataset lock, before any task runs.

cognify() leans on it: a keyless code-only build decides there whether to install the
GLiNER runtime and probe the providers. The callback must see the rows the run will
process, and an empty selection must not widen to the whole dataset.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.locks import held_datasets

pipeline_module = importlib.import_module("cognee.modules.pipelines.operations.pipeline")


@pytest.fixture
def run(monkeypatch):
    """A run of one dataset with the row reads and the task runner replaced."""
    dataset = SimpleNamespace(id=uuid4(), name="docs")
    events = []
    selected = AsyncMock(side_effect=lambda dataset_id, data: events.append("selected") or data)
    whole = AsyncMock(side_effect=lambda dataset_id: events.append("whole") or ["every row"])

    async def run_tasks(tasks, dataset_id, body_data, *args, **kwargs):
        events.append(("tasks", list(body_data)))
        yield "run info"

    monkeypatch.setattr(pipeline_module, "get_selected_dataset_data", selected)
    monkeypatch.setattr(pipeline_module, "get_dataset_data", whole)
    monkeypatch.setattr(pipeline_module, "run_tasks", run_tasks)

    async def go(**kwargs):
        return [
            info
            async for info in pipeline_module.run_pipeline_per_dataset(
                dataset=dataset, user=SimpleNamespace(id=uuid4()), tasks=[], **kwargs
            )
        ]

    return SimpleNamespace(dataset=dataset, events=events, go=go, selected=selected, whole=whole)


@pytest.mark.asyncio
async def test_before_run_sees_the_rows_and_the_dataset_under_the_lock_before_the_tasks(run):
    seen = []

    async def before_run(rows, dataset):
        seen.append((list(rows), dataset, run.dataset.id in held_datasets.get()))
        run.events.append("before_run")

    await run.go(data=["row a", "row b"], before_run=before_run)

    assert seen == [(["row a", "row b"], run.dataset, True)]
    assert run.events == ["selected", "before_run", ("tasks", ["row a", "row b"])]


@pytest.mark.asyncio
async def test_a_failing_before_run_stops_the_run_before_any_task(run):
    async def before_run(rows, dataset):
        raise RuntimeError("provider unreachable")

    with pytest.raises(RuntimeError, match="provider unreachable"):
        await run.go(data=["row a"], before_run=before_run)

    assert not any(isinstance(event, tuple) for event in run.events)


@pytest.mark.asyncio
async def test_a_run_without_before_run_reads_and_runs_as_it_always_did(run):
    await run.go(data=None)

    assert run.events == ["whole", ("tasks", ["every row"])]


@pytest.mark.asyncio
async def test_an_empty_selection_runs_nothing_instead_of_the_whole_dataset(run):
    await run.go(data=[])

    run.whole.assert_not_awaited()
    assert run.events == ["selected", ("tasks", [])]
