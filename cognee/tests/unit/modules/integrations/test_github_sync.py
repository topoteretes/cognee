"""Unit tests for cognee.modules.integrations.github.sync.

Token minting, repo listing, and remember() are mocked — what's under test is
the orchestration: one dataset per installation, every repo remembered in ONE
call by its credential-free clone URL with the installation token in
``codegraph_config["repo_credentials"]``, declared repositories
(``treat_as_repository``) and code-graph-only ingestion
(``include_documents=False``), per-repo failures read from the result items,
and the full-installation default when no explicit repo list is given.
"""

import importlib
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

sync_module = importlib.import_module("cognee.modules.integrations.github.sync")

# The remember package's __init__ shadows the submodule with the function, so
# patch the submodule itself.
_remember_module = importlib.import_module("cognee.api.v1.remember.remember")
_users_methods = importlib.import_module("cognee.modules.users.methods")

_USER_ID = uuid4()


def _credential(**overrides):
    defaults = {
        "status": "active",
        "provider_account_id": "42",
        "provider_metadata": {"account_login": "Acme-Org"},
        "user_id": _USER_ID,
    }
    return SimpleNamespace(**{**defaults, **overrides})


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
    owner = SimpleNamespace(id=_USER_ID)
    mocked = SimpleNamespace(
        mint=AsyncMock(return_value=("tok123", None)),
        list_repos=AsyncMock(return_value=["acme/api", "acme/web"]),
        remember=AsyncMock(return_value=SimpleNamespace(status="completed", error=None)),
        get_user=AsyncMock(return_value=owner),
        owner=owner,
    )
    monkeypatch.setattr(sync_module, "mint_installation_token", mocked.mint)
    monkeypatch.setattr(sync_module, "list_installation_repositories", mocked.list_repos)
    # Patched at their home modules: sync_repositories imports both lazily.
    monkeypatch.setattr(_remember_module, "remember", mocked.remember)
    monkeypatch.setattr(_users_methods, "get_user", mocked.get_user)
    return mocked


@pytest.mark.asyncio
async def test_default_sync_covers_every_installation_repo_in_one_call(mocks):
    await sync_module.sync_repositories(_credential())

    mocks.mint.assert_awaited_once_with(42)
    mocks.list_repos.assert_awaited_once_with("tok123")
    # One remember for the whole installation; remember isolates each repository.
    mocks.remember.assert_awaited_once()
    assert mocks.remember.await_args.args == (
        ["https://github.com/acme/api.git", "https://github.com/acme/web.git"],
    )
    assert mocks.remember.await_args.kwargs == {
        "dataset_name": "github_acme_org",
        "user": mocks.owner,
        "codegraph_config": {
            "repo_credentials": "tok123",
            "treat_as_repository": True,
            "include_documents": False,  # code graphs only, no README extraction
        },
        "self_improvement": False,
        "raise_on_error": False,
    }


@pytest.mark.asyncio
async def test_explicit_repo_list_skips_the_listing_call(mocks):
    await sync_module.sync_repositories(_credential(), ["acme/api"])

    mocks.list_repos.assert_not_awaited()
    mocks.remember.assert_awaited_once()
    assert mocks.remember.await_args.args == (["https://github.com/acme/api.git"],)


@pytest.mark.asyncio
async def test_per_repo_failures_are_logged_by_repository(mocks, caplog):
    mocks.remember.return_value = SimpleNamespace(
        status="errored",
        error="https://github.com/acme/api.git: clone failed",
        items=[
            {"id": "x", "kind": "code_repo", "source": "https://github.com/acme/web.git"},
            {
                "kind": "code_repo",
                "source": "https://github.com/acme/api.git",
                "status": "errored",
                "error": "clone failed",
            },
        ],
    )

    with caplog.at_level(logging.WARNING, logger=sync_module.logger.name):
        await sync_module.sync_repositories(_credential())

    mocks.remember.assert_awaited_once()
    assert "GitHub sync for acme/api errored: clone failed" in caplog.text
    assert "1 failed repositories: acme/api" in caplog.text
    assert "acme/web" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["mint", "remember"])
async def test_a_sync_wide_failure_propagates_to_the_detached_caller(mocks, failing):
    """Only per-repository failures are summarised here; anything that fails the
    whole sync (minting the token alike) reaches the caller's logging guard."""
    getattr(mocks, failing).side_effect = RuntimeError("database down")

    with pytest.raises(RuntimeError, match="database down"):
        await sync_module.sync_repositories(_credential())


@pytest.mark.asyncio
async def test_empty_installation_syncs_nothing(mocks):
    mocks.list_repos.return_value = []

    await sync_module.sync_repositories(_credential())

    mocks.remember.assert_not_awaited()


@pytest.mark.asyncio
async def test_account_login_falls_back_to_the_installation_id(mocks):
    await sync_module.sync_repositories(_credential(provider_metadata=None), ["acme/api"])

    assert mocks.remember.await_args.kwargs["dataset_name"] == "github_42"
