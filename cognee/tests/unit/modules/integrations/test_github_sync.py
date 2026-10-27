"""Unit tests for cognee.modules.integrations.github.sync.

Token minting, repo listing, add() and cognify() are mocked. What is under test is
the orchestration: one dataset per installation, each repository added by its
credential-free clone URL with the token in ``codegraph_config`` and built in its
own ``cognify(data_ids=...)`` run, a failing repository logged without stopping the
others, a fresh token before each clone, and the full-installation default when no
explicit repo list is given.
"""

import asyncio
import importlib
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import aiohttp
import pytest

from cognee.modules.pipelines.exceptions import CognifyFailedError
from cognee.modules.pipelines.models import PipelineRunCompleted, PipelineRunErrored
from cognee.tasks.code_graph.resolve_repo import CodeRepositoryError

sync_module = importlib.import_module("cognee.modules.integrations.github.sync")

# The add package's __init__ shadows the submodule with the function, so
# patch the submodule itself.
_add_module = importlib.import_module("cognee.api.v1.add")
_cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
_users_methods = importlib.import_module("cognee.modules.users.methods")

_USER_ID = uuid4()


def _credential(**overrides):
    defaults = {
        "id": uuid4(),
        "provider": "github",
        "workspace_id": None,
        "status": "active",
        "provider_account_id": "42",
        "provider_metadata": {"account_login": "Acme-Org"},
        "user_id": _USER_ID,
    }
    return SimpleNamespace(**{**defaults, **overrides})


def _added(data_ids=None):
    return PipelineRunCompleted(
        pipeline_run_id=uuid4(),
        dataset_id=uuid4(),
        dataset_name="github_acme_org",
        data_ingestion_info=[
            {"data_id": data_id, "run_info": None} for data_id in (data_ids or [uuid4()])
        ],
    )


def test_dataset_name_is_one_per_account_and_identifier_safe():
    assert sync_module.dataset_name_for_account("Acme-Org") == "github_acme_org"
    assert sync_module.dataset_name_for_account("user.name") == "github_user_name"
    assert sync_module.dataset_name_for_account("---") == "github_account"


def test_clone_url_carries_no_credentials():
    # Auth travels out-of-band as repo_credentials; a token in the URL would
    # taint every URL-derived string (slugs, logs, git errors) with a secret.
    assert sync_module.clone_url("acme/api") == "https://github.com/acme/api.git"


@pytest.fixture
def mocks(monkeypatch):
    from cognee.modules.integrations.sync_progress import SyncProgressReporter

    observations = []

    original_report = SyncProgressReporter.report

    async def capture(reporter, **kwargs):
        await original_report(reporter, **kwargs)
        observations.append(reporter.snapshot())

    monkeypatch.setattr(SyncProgressReporter, "report", capture)
    monkeypatch.setattr(SyncProgressReporter, "_publish", AsyncMock())
    owner = SimpleNamespace(id=_USER_ID)
    mocked = SimpleNamespace(
        mint=AsyncMock(return_value=("tok123", None)),
        list_repos=AsyncMock(return_value=["acme/api", "acme/web"]),
        add=AsyncMock(side_effect=lambda *args, **kwargs: _added()),
        cognify=AsyncMock(return_value={}),
        get_user=AsyncMock(return_value=owner),
        owner=owner,
        observations=observations,
    )
    monkeypatch.setattr(sync_module, "mint_installation_token", mocked.mint)
    monkeypatch.setattr(sync_module, "list_installation_repositories", mocked.list_repos)
    # Patched at their home modules: _add_and_build imports both lazily.
    monkeypatch.setattr(_add_module, "add", mocked.add)
    monkeypatch.setattr(_cognify_module, "cognify", mocked.cognify)
    monkeypatch.setattr(_users_methods, "get_user", mocked.get_user)
    return mocked


@pytest.mark.asyncio
async def test_default_sync_covers_every_installation_repo(mocks):
    await sync_module.sync_repositories(_credential())

    mocks.list_repos.assert_awaited_once_with("tok123")
    assert [call.args[0] for call in mocks.add.await_args_list] == [
        "https://github.com/acme/api.git",
        "https://github.com/acme/web.git",
    ]
    for call in mocks.add.await_args_list:
        assert call.kwargs == {
            "dataset_name": "github_acme_org",
            "user": mocks.owner,
            "skip_connection_test": True,
            "codegraph_config": {"repo_credentials": "tok123", "include_documents": False},
        }


@pytest.mark.asyncio
async def test_each_repository_is_built_in_its_own_scoped_run(mocks):
    added = [_added(), _added([uuid4(), uuid4()])]
    mocks.add.side_effect = added

    await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.cognify.await_count == 2
    for run, call in zip(added, mocks.cognify.await_args_list):
        assert call.kwargs == {
            "datasets": [run.dataset_id],
            "data_ids": run.data_ids,
            "user": mocks.owner,
        }


@pytest.mark.asyncio
async def test_a_token_without_an_expiry_is_minted_before_each_clone(mocks):
    mocks.mint.side_effect = [("listing", None), ("first", None), ("second", None)]

    await sync_module.sync_repositories(_credential())

    assert [
        call.kwargs["codegraph_config"]["repo_credentials"] for call in mocks.add.await_args_list
    ] == ["first", "second"]


@pytest.mark.asyncio
async def test_explicit_repo_list_skips_the_listing_call(mocks):
    await sync_module.sync_repositories(_credential(), ["acme/api"])

    mocks.list_repos.assert_not_awaited()
    mocks.add.assert_awaited_once()
    assert mocks.add.await_args.args == ("https://github.com/acme/api.git",)


@pytest.mark.asyncio
async def test_empty_installation_syncs_nothing(mocks):
    mocks.list_repos.return_value = []

    await sync_module.sync_repositories(_credential())

    mocks.add.assert_not_awaited()
    mocks.cognify.assert_not_awaited()


@pytest.mark.asyncio
async def test_account_login_falls_back_to_the_installation_id(mocks):
    await sync_module.sync_repositories(_credential(provider_metadata=None), ["acme/api"])

    assert mocks.add.await_args.kwargs["dataset_name"] == "github_42"


@pytest.mark.asyncio
async def test_sync_reports_live_work_without_inventing_file_progress(mocks):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def add(*args, **kwargs):
        entered.set()
        await release.wait()
        return _added()

    mocks.add.side_effect = add
    task = asyncio.create_task(sync_module.sync_repositories(_credential(), ["acme/api"]))
    await entered.wait()
    progress = mocks.observations[-1]
    assert progress.state == "running"
    assert progress.unit == "repositories"
    assert progress.completed == 0
    assert progress.total == 1
    release.set()
    await task
    assert mocks.observations[-1].state == "succeeded"
    assert mocks.observations[-1].counts == {"processed": 1, "failed": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing,state",
    [(["acme/api", "acme/web"], "failed"), (["acme/api"], "partial")],
)
async def test_handled_repo_failures_are_not_reported_successful(mocks, failing, state):
    mocks.cognify.side_effect = [
        CognifyFailedError(error_class="RuntimeError", error_message="secret")
        if name in failing
        else {}
        for name in ("acme/api", "acme/web")
    ]

    await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    progress = mocks.observations[-1]
    assert progress.state == state
    assert progress.counts["failed"] == len(failing)
    assert progress.reason == "processing_error"
    assert "secret" not in progress.model_dump_json()


@pytest.mark.asyncio
async def test_a_repository_that_cannot_be_cloned_fails_alone(mocks, caplog):
    mocks.add.side_effect = [CodeRepositoryError(message="Failed to clone"), _added()]

    with caplog.at_level(logging.ERROR):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.add.await_count == 2 and mocks.cognify.await_count == 1
    assert "failed for acme/api" in caplog.text
    assert mocks.observations[-1].state == "partial"


@pytest.mark.asyncio
async def test_a_failed_add_keeps_its_cause_and_the_token_never_reaches_the_log(mocks, caplog):
    mocks.add.side_effect = [
        PipelineRunErrored(
            pipeline_run_id=uuid4(),
            dataset_id=uuid4(),
            dataset_name="github_acme_org",
            error_class="CodeRepositoryError",
            error_message="Failed to clone with tok123",
        ),
        _added(),
    ]

    with caplog.at_level(logging.ERROR):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert "CodeRepositoryError: Failed to clone" in caplog.text
    assert "tok123" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("data_ids", [None, []], ids=["none", "empty"])
async def test_an_add_without_rows_fails_that_repository_and_builds_nothing(
    mocks, caplog, data_ids
):
    """cognify(data_ids=None) would build the whole dataset."""
    mocks.add.side_effect = [SimpleNamespace(dataset_id=uuid4(), data_ids=data_ids), _added()]

    with caplog.at_level(logging.ERROR):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.add.await_count == 2 and mocks.cognify.await_count == 1
    assert "failed for acme/api" in caplog.text


@pytest.mark.asyncio
async def test_a_failed_build_fails_that_repository_alone(mocks, caplog):
    """cognify() wraps what a build raises, so the loop sees a cognee error."""
    mocks.cognify.side_effect = [
        CognifyFailedError(error_class="RuntimeError", error_message="graph write failed"),
        {},
    ]

    with caplog.at_level(logging.ERROR):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.add.await_count == 2 and mocks.cognify.await_count == 2
    assert "failed for acme/api" in caplog.text
    assert mocks.observations[-1].state == "partial"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [OSError("no space left"), asyncio.TimeoutError(), aiohttp.ClientError("connection reset")],
    ids=["os", "timeout", "client"],
)
async def test_an_io_or_network_failure_fails_that_repository_alone(mocks, caplog, error):
    mocks.add.side_effect = [error, _added()]

    with caplog.at_level(logging.ERROR):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.add.await_count == 2 and mocks.cognify.await_count == 1
    assert "failed for acme/api" in caplog.text
    assert mocks.observations[-1].state == "partial"


@pytest.mark.asyncio
async def test_an_error_add_raises_outside_the_caught_types_aborts_the_sync(mocks):
    mocks.add.side_effect = [TypeError("a bug"), _added()]

    with pytest.raises(TypeError):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.add.await_count == 1
    mocks.cognify.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_token_mint_aborts_the_sync(mocks):
    mocks.mint.side_effect = [("listing", None), RuntimeError("mint failed")]

    with pytest.raises(RuntimeError, match="mint failed"):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    mocks.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_repo_sync_preserves_cancellation(mocks):
    mocks.add.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await sync_module.sync_repositories(_credential(), ["acme/api"])
    assert mocks.observations[-1].state == "cancelled"


@pytest.mark.asyncio
async def test_repository_discovery_reports_each_page_without_claiming_a_total(monkeypatch):
    responses = []
    for names, next_url in [(["acme/api"], "https://api.github.com/page2"), (["acme/web"], None)]:
        response = MagicMock()
        response.status = 200
        response.json = AsyncMock(return_value={"repositories": [{"full_name": n} for n in names]})
        response.links = {"next": {"url": next_url}} if next_url else {}
        response.__aenter__ = AsyncMock(return_value=response)
        responses.append(response)
    session = MagicMock()
    session.get.side_effect = responses
    session.__aenter__ = AsyncMock(return_value=session)
    monkeypatch.setattr(sync_module.aiohttp, "ClientSession", lambda **kwargs: session)
    report = AsyncMock()
    monkeypatch.setattr(sync_module, "report_sync_progress", report)
    assert await sync_module.list_installation_repositories("secret") == ["acme/api", "acme/web"]
    assert [call.kwargs["completed"] for call in report.await_args_list] == [1, 2]
    assert all("total" not in call.kwargs for call in report.await_args_list)
    assert all("secret" not in str(call.kwargs) for call in report.await_args_list)


@pytest.mark.asyncio
async def test_a_token_far_from_expiry_is_reused_for_every_clone(mocks):
    from datetime import datetime, timedelta, timezone

    mocks.mint.return_value = ("tok123", datetime.now(timezone.utc) + timedelta(minutes=50))

    await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web", "acme/docs"])

    mocks.mint.assert_awaited_once()  # the listing token served all three clones
    assert {
        call.kwargs["codegraph_config"]["repo_credentials"] for call in mocks.add.await_args_list
    } == {"tok123"}


@pytest.mark.asyncio
async def test_a_token_close_to_expiry_is_renewed_before_the_clone(mocks):
    from datetime import datetime, timedelta, timezone

    soon = datetime.now(timezone.utc) + timedelta(minutes=2)
    later = datetime.now(timezone.utc) + timedelta(minutes=55)
    mocks.mint.side_effect = [("old", soon), ("new", later)]

    await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.mint.await_count == 2
    assert [
        call.kwargs["codegraph_config"]["repo_credentials"] for call in mocks.add.await_args_list
    ] == ["new", "new"]


@pytest.mark.asyncio
async def test_the_token_never_reaches_the_log_of_a_failed_repository(mocks, caplog):
    mocks.add.side_effect = [OSError("write failed, token tok123 rejected"), _added()]

    with caplog.at_level(logging.ERROR):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert "failed for acme/api" in caplog.text
    assert "tok123" not in caplog.text


@pytest.fixture
def legacy(monkeypatch, mocks):
    """The dataset and legacy rows of a repository and the calls that forget them, with the
    real add and cognify replaced as in every sync test."""
    code_repo = importlib.import_module("cognee.tasks.code_graph.code_repo")
    forget_module = importlib.import_module("cognee.api.v1.forget")
    data_methods = importlib.import_module("cognee.modules.data.methods")
    dataset = SimpleNamespace(id=uuid4())
    ids = [uuid4(), uuid4()]
    forget = AsyncMock(return_value={"status": "success"})
    datasets = AsyncMock(return_value=[dataset])
    row_ids = AsyncMock(return_value=ids)
    monkeypatch.setattr(forget_module, "forget", forget)
    monkeypatch.setattr(data_methods, "get_authorized_existing_datasets", datasets)
    monkeypatch.setattr(code_repo, "legacy_repository_row_ids", row_ids)
    return SimpleNamespace(
        forget=forget,
        datasets=datasets,
        row_ids=row_ids,
        dataset=dataset,
        ids=ids,
        legacy_error=code_repo.LegacyRepositoryError,
        changed_error=code_repo.RepositoryChangedError,
    )


@pytest.mark.asyncio
async def test_a_legacy_repository_is_forgotten_and_built_again_once(mocks, legacy, caplog):
    mocks.add.side_effect = [legacy.legacy_error("a remote repository row"), _added(), _added()]

    with caplog.at_level(logging.INFO):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert [call.kwargs for call in legacy.forget.await_args_list] == [
        {"data_id": data_id, "dataset_id": legacy.dataset.id, "user": mocks.owner}
        for data_id in legacy.ids
    ]
    assert mocks.add.await_count == 3 and mocks.cognify.await_count == 2
    assert mocks.observations[-1].state == "succeeded"
    assert "forgot 2 row(s)" in caplog.text and "tok123" not in caplog.text
    assert "success" in caplog.text  # the forget receipt is logged


@pytest.mark.asyncio
async def test_the_legacy_rows_are_looked_up_in_the_dataset_the_owner_can_write(mocks, legacy):
    mocks.add.side_effect = [legacy.legacy_error("a remote repository row"), _added()]

    await sync_module.sync_repositories(_credential(), ["acme/api"])

    legacy.datasets.assert_awaited_once_with(["github_acme_org"], "write", mocks.owner, strict=True)
    legacy.row_ids.assert_awaited_once_with("https://github.com/acme/api.git", legacy.dataset.id)


@pytest.mark.asyncio
async def test_a_legacy_error_without_a_legacy_row_is_not_retried(mocks, legacy, caplog):
    """A legacy graph that no row owns cannot be healed by forgetting rows: the error
    must not be hidden behind a second build that fails the same way."""
    legacy.row_ids.return_value = []
    mocks.add.side_effect = [_added(), _added()]
    mocks.cognify.side_effect = [legacy.legacy_error("a graph written before"), {}]

    with caplog.at_level(logging.WARNING):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    legacy.forget.assert_not_awaited()
    assert mocks.add.await_count == 2 and mocks.cognify.await_count == 2  # web is built
    assert "forgot" not in caplog.text
    assert mocks.observations[-1].state == "partial"


@pytest.mark.asyncio
async def test_a_repository_that_stays_legacy_fails_alone_after_one_rebuild(mocks, legacy):
    mocks.add.side_effect = [
        legacy.legacy_error("a remote repository row"),
        legacy.legacy_error("a remote repository row"),
        _added(),
    ]

    await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert legacy.forget.await_count == len(legacy.ids)  # forgotten once, never in a loop
    assert mocks.add.await_count == 3 and mocks.cognify.await_count == 1
    assert mocks.observations[-1].state == "partial"


@pytest.mark.asyncio
async def test_a_dataset_that_cannot_be_resolved_fails_that_repository_alone(mocks, legacy):
    from cognee.modules.data.exceptions import DatasetNotFoundError

    legacy.datasets.side_effect = DatasetNotFoundError(message="gone")
    mocks.add.side_effect = [legacy.legacy_error("a remote repository row"), _added()]

    await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    legacy.forget.assert_not_awaited()
    assert mocks.add.await_count == 2 and mocks.cognify.await_count == 1
    assert mocks.observations[-1].state == "partial"


@pytest.mark.asyncio
async def test_a_failing_forget_aborts_the_sync(mocks, legacy):
    """A forget that fails for a reason of its own is not one repository's failure."""
    legacy.forget.side_effect = RuntimeError("database is gone")
    mocks.add.side_effect = [legacy.legacy_error("a remote repository row"), _added()]

    with pytest.raises(RuntimeError, match="database is gone"):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.add.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True], ids=["raised", "reported_as_errored_run"])
async def test_a_repository_that_moved_after_add_is_added_and_built_again(
    mocks, legacy, caplog, wrapped
):
    """Two overlapping syncs, or an upstream change between add and cognify: the second
    add pins the current revision. With RAISE_INCREMENTAL_LOADING_ERRORS off cognify
    raises the errored run as CognifyFailedError carrying the error class."""
    changed = (
        CognifyFailedError(error_class="RepositoryChangedError", error_message="moved")
        if wrapped
        else legacy.changed_error()
    )
    mocks.add.side_effect = [_added(), _added(), _added()]
    mocks.cognify.side_effect = [changed, {}, {}]

    with caplog.at_level(logging.WARNING):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.add.await_count == 3 and mocks.cognify.await_count == 3
    assert "changed while it was being built" in caplog.text
    assert mocks.observations[-1].state == "succeeded"
    legacy.forget.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failure_of_another_kind_is_not_retried(mocks, legacy):
    mocks.cognify.side_effect = [
        CognifyFailedError(error_class="RuntimeError", error_message="graph write failed"),
        {},
    ]

    await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.add.await_count == 2 and mocks.cognify.await_count == 2
    assert mocks.observations[-1].state == "partial"


@pytest.mark.asyncio
async def test_a_repository_that_keeps_moving_fails_after_one_more_build(mocks, legacy):
    mocks.cognify.side_effect = [legacy.changed_error(), legacy.changed_error(), {}]

    await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.add.await_count == 3 and mocks.cognify.await_count == 3  # web is built
    assert mocks.observations[-1].state == "partial"


@pytest.mark.asyncio
async def test_an_add_reported_as_moved_is_added_and_built_again(mocks, legacy, caplog):
    """With RAISE_INCREMENTAL_LOADING_ERRORS off add() returns the error as an errored run."""
    moved = PipelineRunErrored(
        pipeline_run_id=uuid4(),
        dataset_id=uuid4(),
        dataset_name="github_acme_org",
        error_class="RepositoryChangedError",
        error_message="moved",
    )
    mocks.add.side_effect = [moved, _added(), _added()]

    with caplog.at_level(logging.WARNING):
        await sync_module.sync_repositories(_credential(), ["acme/api", "acme/web"])

    assert mocks.add.await_count == 3 and mocks.cognify.await_count == 2
    assert "changed while it was being built" in caplog.text
    assert mocks.observations[-1].state == "succeeded"
