"""Unit tests for memify()'s memory-fragment projection gating.

memify() projects the entire graph into a CogneeGraph when the caller passes no
``data``. The default pipeline never reads that projection (its tasks query the
graph themselves or drop non-DataPoint inputs), so paying for it on every
``remember()`` -> ``improve()`` was a full graph read for nothing. These tests
pin the gating: skipped for pipelines built only from tasks marked
``@ignores_memory_fragment``, still performed for anything custom.
"""

import types
from uuid import uuid4

import pytest

from cognee.modules.memify.memify import _any_task_consumes_memory_fragment
from cognee.modules.pipelines.tasks.task import Task, ignores_memory_fragment
from cognee.memify_pipelines.memify_default_tasks import (
    get_default_memify_enrichment_tasks,
    get_default_memify_extraction_tasks,
)


@ignores_memory_fragment
async def _marked_task(data):
    return data


async def _unmarked_task(data):
    return data


def test_marked_tasks_alone_do_not_consume_the_fragment():
    assert _any_task_consumes_memory_fragment([Task(_marked_task)]) is False


def test_unmarked_task_consumes_the_fragment():
    assert _any_task_consumes_memory_fragment([Task(_unmarked_task)]) is True


def test_one_unmarked_task_is_enough_to_keep_the_projection():
    tasks = [Task(_marked_task), Task(_unmarked_task)]

    assert _any_task_consumes_memory_fragment(tasks) is True


def test_empty_pipeline_consumes_nothing():
    assert _any_task_consumes_memory_fragment([]) is False


@pytest.mark.parametrize("triplet_embedding", [False, True])
def test_default_memify_tasks_never_consume_the_fragment(monkeypatch, triplet_embedding):
    """Both default configurations read the graph themselves, not the fragment."""
    import cognee.modules.cognify.config as cognify_config_module

    monkeypatch.setattr(
        cognify_config_module,
        "get_cognify_config",
        lambda: types.SimpleNamespace(triplet_embedding=triplet_embedding),
    )

    tasks = [*get_default_memify_extraction_tasks(), *get_default_memify_enrichment_tasks()]

    assert _any_task_consumes_memory_fragment(tasks) is False


class _FakeDataset:
    def __init__(self):
        self.id = uuid4()
        self.owner_id = uuid4()


@pytest.fixture
def memify_env(monkeypatch):
    """Stub memify's collaborators and record what the pipeline received."""
    import importlib

    # The package re-exports the function under the same name, so import the
    # module explicitly rather than through attribute lookup.
    memify_module = importlib.import_module("cognee.modules.memify.memify")

    calls = {"projections": 0, "data": None, "tasks": None}

    async def fake_get_memory_fragment(**kwargs):
        calls["projections"] += 1
        return "projected-fragment"

    async def fake_setup():
        return None

    async def fake_resolve(dataset, user):
        return user, [_FakeDataset()]

    async def fake_run_pipeline(**kwargs):
        calls["data"] = kwargs["data"]
        calls["tasks"] = kwargs["tasks"]
        return {"status": "ok"}

    def fake_get_pipeline_executor(run_in_background):
        async def execute(pipeline, **kwargs):
            return await pipeline(**kwargs)

        return execute

    class _NullContext:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *exc_info):
            return False

    monkeypatch.setattr(memify_module, "get_memory_fragment", fake_get_memory_fragment)
    monkeypatch.setattr(memify_module, "setup", fake_setup)
    monkeypatch.setattr(memify_module, "resolve_authorized_user_datasets", fake_resolve)
    monkeypatch.setattr(memify_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(memify_module, "get_pipeline_executor", fake_get_pipeline_executor)
    monkeypatch.setattr(
        memify_module,
        "set_database_global_context_variables",
        lambda *args, **kwargs: _NullContext(),
    )

    return calls


@pytest.mark.asyncio
async def test_memify_skips_projection_for_fragment_ignoring_pipeline(memify_env):
    from cognee.modules.memify.memify import memify

    await memify(
        extraction_tasks=[Task(_marked_task)],
        enrichment_tasks=[Task(_marked_task)],
        user=types.SimpleNamespace(id=uuid4()),
    )

    assert memify_env["projections"] == 0
    assert memify_env["data"] == [{}]


@pytest.mark.asyncio
async def test_memify_projects_for_custom_pipeline(memify_env):
    from cognee.modules.memify.memify import memify

    await memify(
        extraction_tasks=[Task(_unmarked_task)],
        enrichment_tasks=[Task(_marked_task)],
        user=types.SimpleNamespace(id=uuid4()),
    )

    assert memify_env["projections"] == 1
    assert memify_env["data"] == ["projected-fragment"]


@pytest.mark.asyncio
async def test_memify_never_projects_when_data_is_supplied(memify_env):
    from cognee.modules.memify.memify import memify

    await memify(
        extraction_tasks=[Task(_unmarked_task)],
        data=["caller-data"],
        user=types.SimpleNamespace(id=uuid4()),
    )

    assert memify_env["projections"] == 0
    assert memify_env["data"] == ["caller-data"]
