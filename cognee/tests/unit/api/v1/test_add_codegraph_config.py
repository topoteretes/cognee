"""add() clones a repository with codegraph_config: credentials and documents.

add() reads ``repo_credentials`` and ``include_documents``. A call that sets either runs
the repository resolver ahead of the pipeline, with the options, because the runner
resolves with the defaults. The options need a repository to apply to and are refused
while connected to a remote instance, where they would be dropped.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from cognee.api.v1.serve import state as serve_state
from cognee.tasks.code_graph.config import validate_codegraph_config
from cognee.tasks.ingestion.data_item import DataItem

add_module = importlib.import_module("cognee.api.v1.add.add")

_REPO = "https://github.com/org/repo.git"


def _manifest(url=True):
    """What resolving a repository gives: a code_repo manifest, with its URL if cloned."""
    metadata = {"source": "code_repo"}
    if url:
        metadata["repo_url"] = _REPO
    return DataItem(data="{}", label="code_repo:repo", system_metadata=metadata)


@pytest.fixture
def executor(monkeypatch):
    """add() with the database and pipeline stubbed; the returned mocks record the run."""
    dataset = SimpleNamespace(id=uuid4(), name="documents")
    user = SimpleNamespace(id=uuid4())
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.preflight"), "validate_provider_config", Mock()
    )
    monkeypatch.setattr(add_module, "setup", AsyncMock())
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.migrations.startup"),
        "run_migrations_and_block",
        AsyncMock(),
    )
    monkeypatch.setattr(
        add_module, "resolve_authorized_user_dataset", AsyncMock(return_value=(user, dataset))
    )
    monkeypatch.setattr(
        add_module, "resolve_dlt_sources", AsyncMock(side_effect=lambda data, **kw: (data, None))
    )
    monkeypatch.setattr(add_module, "refuse_changed_existing_documents", AsyncMock())
    run = AsyncMock(return_value=None)
    monkeypatch.setattr(add_module, "get_pipeline_executor", lambda **kwargs: run)
    resolver = AsyncMock(side_effect=lambda data, **kwargs: [_manifest()])
    monkeypatch.setattr(add_module, "resolve_data_directories", resolver)
    return SimpleNamespace(run=run, resolver=resolver, dataset=dataset, user=user)


def test_the_config_accepts_only_its_keys_and_real_values():
    assert validate_codegraph_config(None) == {}
    assert validate_codegraph_config({"repo_credentials": "tok", "include_documents": None}) == {
        "repo_credentials": "tok"
    }
    with pytest.raises(ValueError, match="Unknown codegraph_config keys: treat_as_repository"):
        validate_codegraph_config({"treat_as_repository": True})
    with pytest.raises(TypeError, match="must be a bool"):  # "false" is truthy
        validate_codegraph_config({"include_documents": "false"})
    with pytest.raises(TypeError, match="non-empty str"):
        validate_codegraph_config({"repo_credentials": ""})
    with pytest.raises(TypeError, match="must be a dict"):
        validate_codegraph_config(["include_documents"])


@pytest.mark.asyncio
async def test_options_resolve_the_repository_ahead_of_the_run(executor):
    await add_module.add(
        _REPO,
        branch="dev",
        codegraph_config={"repo_credentials": "tok", "include_documents": False},
    )

    executor.resolver.assert_awaited_once()
    assert executor.resolver.await_args.args == (_REPO,)
    assert executor.resolver.await_args.kwargs == {
        "include_subdirectories": True,
        "user": executor.user,
        "dataset_id": executor.dataset.id,
        "credentials": "tok",
        "include_documents": False,
        "branch": "dev",
    }
    # The pipeline then runs on the resolved items, not on the original spec.
    assert [item.label for item in executor.run.await_args.kwargs["data"]] == ["code_repo:repo"]


@pytest.mark.asyncio
async def test_a_call_without_options_leaves_resolving_to_the_runner(executor):
    await add_module.add(_REPO)
    await add_module.add(_REPO, codegraph_config={"include_documents": True})

    executor.resolver.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_token_never_reaches_the_pipeline_arguments(executor):
    await add_module.add(_REPO, codegraph_config={"repo_credentials": "s3cret-tok"})

    assert "s3cret-tok" not in repr(executor.run.await_args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config", [{"repo_credentials": "tok"}, {"include_documents": False}], ids=["token", "docs"]
)
async def test_options_with_nothing_to_apply_them_to_are_refused(executor, config):
    """Plain text resolves to no repository: the options would be silently ignored."""
    executor.resolver.side_effect = lambda data, **kwargs: ["a note"]

    with pytest.raises(ValueError, match="codegraph_config applies to repositories"):
        await add_module.add("a note", codegraph_config=config)

    executor.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_token_for_a_local_directory_is_refused(executor):
    executor.resolver.side_effect = lambda data, **kwargs: [_manifest(url=False)]

    with pytest.raises(ValueError, match="repository URLs, not to a local directory"):
        await add_module.add("/some/repo", codegraph_config={"repo_credentials": "tok"})

    executor.run.assert_not_awaited()

    # Without a token the same local repository is fine: only documents are left out.
    await add_module.add("/some/repo", codegraph_config={"include_documents": False})
    executor.run.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config", [{"repo_credentials": "tok"}, {"include_documents": False}], ids=["token", "docs"]
)
async def test_a_remote_instance_refuses_options_it_would_drop(monkeypatch, config):
    client = SimpleNamespace(add=AsyncMock())
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)

    with pytest.raises(ValueError, match="not supported while connected to a remote"):
        await add_module.add(_REPO, codegraph_config=config)

    client.add.assert_not_awaited()
