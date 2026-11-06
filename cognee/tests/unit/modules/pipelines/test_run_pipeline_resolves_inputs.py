"""``run_pipeline`` resolves the call's inputs once, under the dataset lock, before the probes.

Directories, repository URLs and declared repository specs are resolved per dataset
with the add() call's ``extras`` (``codegraph_config`` and ``repo_branch``), inside
the dataset lock (resolving a repository rewrites its manifest row's stamps), and
before the first-use connection probes: a run whose resolved items are all code
repository manifests embeds nothing, so its embedding probe is skipped unless the
call indexes vectors.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.pipeline as pipeline_module
from cognee.tasks.ingestion.data_item import DataItem


@pytest.fixture
def plumbing(monkeypatch):
    dataset = SimpleNamespace(id=uuid4(), name="repo_ds", owner_id=uuid4())
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    events = []

    @asynccontextmanager
    async def recording_lock(dataset_id):
        events.append(("lock", dataset_id))
        yield
        events.append(("unlock", dataset_id))

    async def resolve(data, **kwargs):
        events.append(("resolve", kwargs))
        return resolve.result

    resolve.result = ["resolved"]

    async def per_dataset(**kwargs):
        events.append(("run", kwargs["data"]))
        yield "run-info"

    monkeypatch.setattr(pipeline_module, "setup_environment", AsyncMock())
    probes = AsyncMock()
    monkeypatch.setattr(pipeline_module, "check_connections", probes)
    monkeypatch.setattr(
        pipeline_module,
        "resolve_authorized_user_datasets",
        AsyncMock(return_value=(user, [dataset])),
    )
    monkeypatch.setattr(pipeline_module, "dataset_lock", recording_lock)
    monkeypatch.setattr(pipeline_module, "run_pipeline_per_dataset", per_dataset)
    # _resolve_inputs imports the function from the package at call time.
    monkeypatch.setattr("cognee.tasks.ingestion.resolve_data_directories", resolve)
    return SimpleNamespace(
        dataset=dataset, user=user, events=events, resolve=resolve, probes=probes
    )


async def _run(data, extras, **kwargs):
    async for _info in pipeline_module.run_pipeline(
        tasks=lambda item: [], data=data, extras=extras, data_cache=True, **kwargs
    ):
        pass


@pytest.mark.asyncio
async def test_inputs_are_resolved_once_under_the_lock_with_the_calls_options(plumbing):
    extras = {
        "repo_branch": "dev",
        "codegraph_config": {"repo_credentials": "tok", "include_documents": False},
    }

    await _run(["https://github.com/org/repo"], extras)

    kinds = [event[0] for event in plumbing.events]
    assert kinds == ["lock", "resolve", "unlock", "run"]
    resolve_kwargs = plumbing.events[1][1]
    assert resolve_kwargs == {
        "user": plumbing.user,
        "dataset_id": plumbing.dataset.id,
        "credentials": "tok",
        "include_documents": False,
        "treat_as_repository": False,
        "branch": "dev",
    }
    assert plumbing.events[3][1] == ["resolved"]


@pytest.mark.asyncio
async def test_without_extras_the_defaults_resolve(plumbing):
    await _run(["https://github.com/org/repo"], None)

    assert plumbing.events[1][1] == {
        "user": plumbing.user,
        "dataset_id": plumbing.dataset.id,
        "credentials": None,
        "include_documents": True,
        "treat_as_repository": False,
        "branch": None,
    }


@pytest.mark.asyncio
async def test_a_run_of_repository_manifests_alone_skips_the_probes(plumbing):
    manifest = DataItem(data="{}", system_metadata={"source": "code_repo"})
    plumbing.resolve.result = [manifest]

    await _run(["/some/repo"], {"codegraph_config": {}})
    assert plumbing.probes.await_args.kwargs["skip_connection_test"] is True

    await _run(["/some/repo"], {"codegraph_config": {"index_vectors": True}})
    assert plumbing.probes.await_args.kwargs["skip_connection_test"] is False

    plumbing.resolve.result = [manifest, "/some/repo/README.md"]
    await _run(["/some/repo"], {"codegraph_config": {}})
    assert plumbing.probes.await_args.kwargs["skip_connection_test"] is False
