"""Unit tests for cognee.modules.integrations.github.sync.

Token minting, repo listing, and remember() are mocked — what's under test
is the orchestration: one dataset per installation, authenticated clone
URLs handed to the code path, and the full-installation default when no
explicit repo list is given.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

sync_module = importlib.import_module("cognee.modules.integrations.github.sync")

# The remember package's __init__ rebinds `remember` to the function,
# shadowing the submodule — string-target monkeypatching resolves the
# function and fails. Importing the submodule explicitly sidesteps it (same
# workaround as test_get_integrations_router.py).
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
async def test_default_sync_covers_every_installation_repo(mocks):
    await sync_module.sync_repositories(_credential())

    mocks.mint.assert_awaited_once_with(42)
    mocks.list_repos.assert_awaited_once_with("tok123")
    mocks.remember.assert_awaited_once_with(
        [
            "https://github.com/acme/api.git",
            "https://github.com/acme/web.git",
        ],
        dataset_name="github_acme_org",
        user=mocks.owner,
        content_type="code",
        repo_credentials="tok123",
    )


@pytest.mark.asyncio
async def test_explicit_repo_list_skips_the_listing_call(mocks):
    await sync_module.sync_repositories(_credential(), ["acme/api"])

    mocks.list_repos.assert_not_awaited()
    mocks.remember.assert_awaited_once()
    (urls,) = mocks.remember.await_args.args
    assert urls == ["https://github.com/acme/api.git"]
    assert mocks.remember.await_args.kwargs["repo_credentials"] == "tok123"


@pytest.mark.asyncio
async def test_empty_installation_syncs_nothing(mocks):
    mocks.list_repos.return_value = []

    await sync_module.sync_repositories(_credential())

    mocks.remember.assert_not_awaited()


@pytest.mark.asyncio
async def test_account_login_falls_back_to_the_installation_id(mocks):
    await sync_module.sync_repositories(_credential(provider_metadata=None), ["acme/api"])

    assert mocks.remember.await_args.kwargs["dataset_name"] == "github_42"
