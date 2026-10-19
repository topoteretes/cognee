"""``add()`` hands back the ids of the rows it stored, on every path it takes.

The pipeline is stubbed at the executor: these tests pin what add() does with
the run info the executor returns (and with a remote server's reply), not the
ingestion itself. The per-item entries use the shapes the two item paths emit
(``run_tasks_data_item_incremental`` / ``_regular``).
"""

import importlib
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
)

# ``cognee.api.v1.add`` and ``cognee.api.v1.serve`` re-export functions of the
# same name, so the modules are resolved here rather than by attribute path.
add_module = importlib.import_module("cognee.api.v1.add.add")
serve_state = importlib.import_module("cognee.api.v1.serve.state")
preflight = importlib.import_module("cognee.modules.preflight")
migrations_startup = importlib.import_module("cognee.modules.migrations.startup")


def _run_info(cls, dataset, **kwargs):
    return cls(pipeline_run_id=uuid4(), dataset_id=dataset.id, dataset_name=dataset.name, **kwargs)


@pytest.fixture
def stubbed_add(monkeypatch):
    """add() with everything but its own result handling stubbed out.

    Returns a function that installs an executor returning ``run_info`` and
    records the kwargs add() passed to it.
    """
    dataset = SimpleNamespace(id=uuid4(), name="notes")
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    calls = {}

    async def _noop(*_args, **_kwargs):
        return None

    async def _resolve(**_kwargs):
        return user, dataset

    async def _dlt(data, **_kwargs):
        return data, None

    async def _same(data):
        return data

    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(preflight, "validate_provider_config", lambda **_: None)
    monkeypatch.setattr(migrations_startup, "run_migrations_and_block", _noop)
    monkeypatch.setattr(add_module, "setup", _noop)
    monkeypatch.setattr(add_module, "resolve_authorized_user_dataset", _resolve)
    monkeypatch.setattr(add_module, "resolve_dlt_sources", _dlt)
    monkeypatch.setattr(add_module, "refuse_changed_existing_documents", _noop)
    monkeypatch.setattr(add_module, "materialize_stream_for_background", _same)

    def install(run_info):
        async def executor(**kwargs):
            calls.update(kwargs)
            return {dataset.id: run_info}

        monkeypatch.setattr(add_module, "get_pipeline_executor", lambda **_: executor)
        return calls

    return SimpleNamespace(install=install, dataset=dataset)


@pytest.mark.asyncio
async def test_add_fills_data_ids_on_the_regular_path(stubbed_add):
    """incremental_loading=False, data_cache=False: the explicit re-ingest path."""
    dataset = stubbed_add.dataset
    stored, re_stored = uuid4(), uuid4()
    calls = stubbed_add.install(
        _run_info(
            PipelineRunCompleted,
            dataset,
            data_ingestion_info=[
                {"run_info": _run_info(PipelineRunCompleted, dataset), "data_id": stored},
                {"run_info": _run_info(PipelineRunCompleted, dataset), "data_id": re_stored},
            ],
        )
    )

    run = await add_module.add(
        ["one", "two"], dataset_name="notes", incremental_loading=False, data_cache=False
    )

    assert calls["incremental_loading"] is False and calls["data_cache"] is False
    assert run.data_ids == [stored, re_stored]
    assert run.model_dump()["data_ids"] == [stored, re_stored]


@pytest.mark.asyncio
async def test_add_keeps_dedup_hits_and_drops_errored_items(stubbed_add):
    dataset = stubbed_add.dataset
    stored, existing, failed = uuid4(), uuid4(), uuid4()
    stubbed_add.install(
        _run_info(
            PipelineRunErrored,
            dataset,
            data_ingestion_info=[
                {"run_info": _run_info(PipelineRunCompleted, dataset), "data_id": stored},
                {"run_info": _run_info(PipelineRunAlreadyCompleted, dataset), "data_id": existing},
                {"run_info": _run_info(PipelineRunErrored, dataset), "data_id": failed},
            ],
        )
    )

    run = await add_module.add("text", dataset_name="notes")

    assert run.data_ids == [stored, existing]


@pytest.mark.asyncio
async def test_add_in_the_background_reports_no_ids_yet(stubbed_add):
    """PipelineRunStarted: no row exists when the call returns, so the field
    is None (not an empty list that would read as "nothing stored")."""
    stubbed_add.install(_run_info(PipelineRunStarted, stubbed_add.dataset, payload=["text"]))

    run = await add_module.add("text", dataset_name="notes", run_in_background=True)

    assert isinstance(run, PipelineRunStarted)
    assert run.data_ids is None
    assert "data_ids" not in run.model_dump()


@pytest.mark.asyncio
async def test_add_against_a_remote_instance_converts_the_ids(monkeypatch):
    stored = uuid4()

    class _Client:
        async def add(self, data, dataset_name):
            return {"status": "PipelineRunCompleted", "data_ids": [str(stored)]}

    monkeypatch.setattr(serve_state, "get_remote_client", lambda: _Client())

    run = await add_module.add("text", dataset_name="notes")

    assert run.data_ids == [stored]
    assert isinstance(run.data_ids[0], UUID)


@pytest.mark.asyncio
async def test_add_against_an_older_remote_instance_reports_no_ids(monkeypatch):
    class _Client:
        async def add(self, data, dataset_name):
            return {"status": "PipelineRunCompleted"}

    monkeypatch.setattr(serve_state, "get_remote_client", lambda: _Client())

    run = await add_module.add("text", dataset_name="notes")

    assert run.data_ids is None
