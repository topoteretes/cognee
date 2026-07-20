from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.pipelines.operations import pipeline


@pytest.mark.asyncio
async def test_pipeline_holds_dataset_lock_until_generator_finishes(monkeypatch):
    dataset = SimpleNamespace(id=uuid4())
    events = []

    @asynccontextmanager
    async def fake_dataset_lock(dataset_id):
        events.append(("lock", dataset_id))
        try:
            yield
        finally:
            events.append(("unlock", dataset_id))

    async def fake_run_tasks():
        events.append(("run", dataset.id))
        yield "completed"

    monkeypatch.setattr(pipeline, "dataset_pipeline_lock", fake_dataset_lock)
    monkeypatch.setattr(pipeline, "run_tasks", lambda *args, **kwargs: fake_run_tasks())

    results = [
        result
        async for result in pipeline.run_pipeline_per_dataset(
            dataset=dataset,
            user=SimpleNamespace(),
            tasks=[],
            data=[SimpleNamespace()],
        )
    ]

    assert results == ["completed"]
    assert events == [
        ("lock", dataset.id),
        ("run", dataset.id),
        ("unlock", dataset.id),
    ]


@pytest.mark.asyncio
async def test_nested_pipeline_on_held_dataset_does_not_reacquire_lock(monkeypatch):
    dataset = SimpleNamespace(id=uuid4())

    def unexpected_lock(dataset_id):
        raise AssertionError(f"nested run unexpectedly locked {dataset_id}")

    async def fake_run_tasks():
        yield "completed"

    monkeypatch.setattr(pipeline, "dataset_pipeline_lock", unexpected_lock)
    monkeypatch.setattr(pipeline, "run_tasks", lambda *args, **kwargs: fake_run_tasks())
    token = pipeline._held_datasets.set(frozenset({dataset.id}))
    try:
        results = [
            result
            async for result in pipeline.run_pipeline_per_dataset(
                dataset=dataset,
                user=SimpleNamespace(),
                tasks=[],
                data=[SimpleNamespace()],
            )
        ]
    finally:
        pipeline._held_datasets.reset(token)

    assert results == ["completed"]
