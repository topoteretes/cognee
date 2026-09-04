"""Unit tests for cognee.modules.integrations.linear.sync.

The GraphQL client and remember()/get_user() are mocked — what's under test
is the orchestration: one dataset per workspace, deterministic issue text,
webhook syncs kept cheap with self_improvement=False, and the post-install
seed batching every issue into ONE remember() call.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

sync_module = importlib.import_module("cognee.modules.integrations.linear.sync")
adapter_module = importlib.import_module("cognee.modules.integrations.linear.adapter")

# The remember package's __init__ rebinds `remember` to the function,
# shadowing the submodule — string-target monkeypatching resolves the
# function and fails. Importing the submodule explicitly sidesteps it (same
# workaround as test_github_sync.py).
_remember_module = importlib.import_module("cognee.api.v1.remember.remember")
_users_methods = importlib.import_module("cognee.modules.users.methods")

_USER_ID = uuid4()

_ISSUE = {
    "id": "issue-uuid-1",
    "identifier": "COG-1",
    "title": "Fix login",
    "description": "Users cannot log in.",
    "url": "https://linear.app/acme-co/issue/COG-1",
    "state": {"name": "In Progress"},
}


def _credential(**overrides):
    defaults = {
        "status": "active",
        "provider_account_id": "org-1",
        "provider_metadata": {"organization_url_key": "Acme-Co"},
        "user_id": _USER_ID,
    }
    return SimpleNamespace(**{**defaults, **overrides})


@pytest.fixture
def mocks(monkeypatch):
    owner = SimpleNamespace(id=_USER_ID)
    mocked = SimpleNamespace(
        graphql=AsyncMock(return_value={"issues": {"nodes": [_ISSUE]}}),
        remember=AsyncMock(return_value=SimpleNamespace(status="completed", error=None)),
        get_user=AsyncMock(return_value=owner),
        owner=owner,
    )
    monkeypatch.setattr(sync_module, "graphql", mocked.graphql)
    # Patched at their home modules: sync imports both lazily.
    monkeypatch.setattr(_remember_module, "remember", mocked.remember)
    monkeypatch.setattr(_users_methods, "get_user", mocked.get_user)
    # Imported lazily inside sync_recent_issues (circular-import seam).
    monkeypatch.setattr(adapter_module, "access_token_for", lambda _credential: "lin_tok")
    return mocked


def test_dataset_name_is_one_per_workspace_and_identifier_safe():
    assert sync_module.dataset_name_for_org("Acme-Co") == "linear_acme_co"
    assert sync_module.dataset_name_for_org("my.workspace") == "linear_my_workspace"
    assert sync_module.dataset_name_for_org("---") == "linear_workspace"


def test_format_issue_renders_stable_plain_text():
    assert sync_module.format_issue(_ISSUE) == (
        "Linear issue COG-1: Fix login\n"
        "URL: https://linear.app/acme-co/issue/COG-1\n"
        "State: In Progress\n"
        "Description: Users cannot log in."
    )


def test_format_issue_omits_an_absent_description_and_tolerates_gaps():
    assert sync_module.format_issue({"identifier": "COG-2", "title": "Add SSO"}) == (
        "Linear issue COG-2: Add SSO\nURL: \nState: Unknown"
    )
    # Identifier falls back to the id, then to a placeholder.
    assert sync_module.format_issue({}).startswith("Linear issue unknown")


@pytest.mark.asyncio
async def test_sync_issue_remembers_into_the_workspace_dataset(mocks):
    await sync_module.sync_issue(_credential(), _ISSUE)

    mocks.remember.assert_awaited_once_with(
        sync_module.format_issue(_ISSUE),
        dataset_name="linear_acme_co",
        user=mocks.owner,
        # Webhook syncs must stay cheap: no improve() pass per issue edit.
        self_improvement=False,
    )


@pytest.mark.asyncio
async def test_sync_issue_falls_back_to_the_organization_id_dataset(mocks):
    await sync_module.sync_issue(_credential(provider_metadata=None), _ISSUE)

    assert mocks.remember.await_args.kwargs["dataset_name"] == "linear_org_1"


@pytest.mark.asyncio
async def test_sync_recent_issues_batches_one_remember_call(mocks):
    second_issue = {**_ISSUE, "id": "issue-uuid-2", "identifier": "COG-2", "title": "Add SSO"}
    mocks.graphql.return_value = {"issues": {"nodes": [_ISSUE, second_issue]}}

    await sync_module.sync_recent_issues(_credential())

    mocks.graphql.assert_awaited_once()
    assert mocks.graphql.await_args.args[0] == "lin_tok"
    assert mocks.graphql.await_args.args[2] == {"limit": 50}
    # One pipeline run over the whole batch, not one per issue.
    mocks.remember.assert_awaited_once_with(
        [sync_module.format_issue(_ISSUE), sync_module.format_issue(second_issue)],
        dataset_name="linear_acme_co",
        user=mocks.owner,
        self_improvement=False,
    )


@pytest.mark.asyncio
async def test_sync_recent_issues_with_no_issues_remembers_nothing(mocks):
    mocks.graphql.return_value = {"issues": {"nodes": []}}

    await sync_module.sync_recent_issues(_credential())

    mocks.remember.assert_not_awaited()
