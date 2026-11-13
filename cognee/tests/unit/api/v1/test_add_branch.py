"""add() and remember() clone a chosen branch of a repository URL.

The branch reaches the clone from two places: run_tasks, which resolves repository URLs
before any task list runs, and the add task itself. remember() passes it on to add(),
and refuses it wherever it would be dropped: with a content_type or a session_id, or
while connected to a remote instance.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.serve import state as serve_state

add_module = importlib.import_module("cognee.api.v1.add.add")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")

_REPO = "https://github.com/org/repo"


@pytest.fixture
def executor(monkeypatch):
    """add() with the database and pipeline stubbed; the returned mock records the run."""
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
    return run


@pytest.mark.asyncio
async def test_the_branch_reaches_run_tasks_and_the_add_task(executor):
    await add_module.add([_REPO, "a note"], branch="dev")
    await add_module.add(_REPO)

    with_branch, without = executor.await_args_list
    # run_tasks resolves repository URLs before the task list runs, from the extras.
    assert with_branch.kwargs["extras"] == {"repo_branch": "dev"}
    assert with_branch.kwargs["tasks"][0].default_params["kwargs"]["branch"] == "dev"
    assert without.kwargs["extras"] == {"repo_branch": None}

    with pytest.raises(ValueError, match="Invalid branch name"):  # never reaches git
        await add_module.add(_REPO, branch="-x")
    assert executor.await_count == 2


@pytest.mark.asyncio
async def test_remember_passes_the_branch_to_add_and_not_to_cognify():
    import cognee

    user = MagicMock(id=uuid4())
    dataset = MagicMock(id=uuid4())

    with (
        patch("cognee.shared.utils.send_telemetry"),
        patch.object(serve_state, "get_remote_client", return_value=None),
        patch("cognee.modules.engine.operations.setup.setup", AsyncMock()),
        patch("cognee.modules.users.methods.get_default_user", AsyncMock(return_value=user)),
        patch.object(
            remember_module,
            "resolve_authorized_user_datasets",
            AsyncMock(return_value=(user, [dataset])),
        ),
        patch("cognee.api.v1.add.add", AsyncMock()) as add,
        patch("cognee.api.v1.cognify.cognify", AsyncMock(return_value={})) as cognify,
    ):
        await cognee.remember(_REPO, dataset_name="code", branch="dev", self_improvement=False)

    assert add.await_args.kwargs["branch"] == "dev"
    assert "branch" not in cognify.await_args.kwargs


@pytest.mark.asyncio
async def test_a_branch_that_would_be_dropped_is_refused(monkeypatch):
    # content_type and session_id never reach add(), and a remote instance gets no branch
    # field: each would ingest the default branch while the caller asked for another.
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.migrations.startup"),
        "run_migrations_and_block",
        AsyncMock(),
    )

    with pytest.raises(ValueError, match="branch is supported only without content_type"):
        await remember_module.remember(_REPO, content_type="code", branch="dev")
    with pytest.raises(ValueError, match="branch is not supported when session_id"):
        await remember_module.remember(_REPO, session_id="s1", branch="dev")

    client = Mock(add=AsyncMock(), remember=AsyncMock())
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)
    with pytest.raises(ValueError, match="connected to a remote Cognee instance"):
        await add_module.add(_REPO, branch="dev")
    with pytest.raises(ValueError, match="connected to a remote Cognee instance"):
        await remember_module.remember(_REPO, branch="dev")
    client.add.assert_not_awaited()
    client.remember.assert_not_awaited()
