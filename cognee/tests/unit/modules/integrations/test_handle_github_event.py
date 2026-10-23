"""Unit tests for cognee.modules.integrations.github.handle_github_event.

The credential store and the sync layer are mocked — what's under test is
the routing: which deliveries revoke, which sync, and which are dropped
(unknown installation, non-default-branch push, malformed body) without
raising, since the handler runs detached and GitHub retries on errors.
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

handle_module = importlib.import_module("cognee.modules.integrations.github.handle_github_event")
handle_github_event = handle_module.handle_github_event

_ACTIVE_CREDENTIAL = SimpleNamespace(
    status="active", provider_account_id="42", provider_metadata={}
)


@pytest.fixture
def mocks(monkeypatch):
    mocked = SimpleNamespace(
        revoke=AsyncMock(return_value=True),
        get_credential=AsyncMock(return_value=_ACTIVE_CREDENTIAL),
        sync=AsyncMock(),
    )
    monkeypatch.setattr(handle_module, "revoke_credential_by_account", mocked.revoke)
    monkeypatch.setattr(handle_module, "get_credential_by_account", mocked.get_credential)
    monkeypatch.setattr(handle_module, "sync_repositories", mocked.sync)
    return mocked


def _body(payload: dict) -> bytes:
    return json.dumps(payload).encode()


@pytest.mark.asyncio
async def test_installation_deleted_revokes_the_credential(mocks):
    await handle_github_event(
        _body({"action": "deleted", "installation": {"id": 42}}),
        {"x-github-event": "installation"},
    )

    mocks.revoke.assert_awaited_once_with("github", "42")
    mocks.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_push_to_default_branch_resyncs_that_repo(mocks):
    await handle_github_event(
        _body(
            {
                "ref": "refs/heads/main",
                "installation": {"id": 42},
                "repository": {"full_name": "acme/api", "default_branch": "main"},
            }
        ),
        {"x-github-event": "push"},
    )

    mocks.sync.assert_awaited_once_with(_ACTIVE_CREDENTIAL, ["acme/api"])


@pytest.mark.asyncio
async def test_push_to_another_branch_is_ignored(mocks):
    await handle_github_event(
        _body(
            {
                "ref": "refs/heads/feature-x",
                "installation": {"id": 42},
                "repository": {"full_name": "acme/api", "default_branch": "main"},
            }
        ),
        {"x-github-event": "push"},
    )

    mocks.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_added_repositories_are_synced_removed_only_logged(mocks):
    await handle_github_event(
        _body(
            {
                "action": "added",
                "installation": {"id": 42},
                "repositories_added": [{"full_name": "acme/new"}],
                "repositories_removed": [{"full_name": "acme/old"}],
            }
        ),
        {"x-github-event": "installation_repositories"},
    )

    # Removed repos never trigger deletion — indexed data outlives the
    # webhook; forget() stays a human decision.
    mocks.sync.assert_awaited_once_with(_ACTIVE_CREDENTIAL, ["acme/new"])


@pytest.mark.asyncio
async def test_unknown_installation_is_dropped(mocks):
    mocks.get_credential.return_value = None

    await handle_github_event(
        _body({"ref": "refs/heads/main", "installation": {"id": 999}}),
        {"x-github-event": "push"},
    )

    mocks.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoked_installation_is_dropped(mocks):
    mocks.get_credential.return_value = SimpleNamespace(status="revoked")

    await handle_github_event(
        _body(
            {
                "ref": "refs/heads/main",
                "installation": {"id": 42},
                "repository": {"full_name": "acme/api", "default_branch": "main"},
            }
        ),
        {"x-github-event": "push"},
    )

    mocks.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_delivery_without_installation_id_is_dropped(mocks):
    await handle_github_event(_body({"zen": "Design for failure."}), {"x-github-event": "ping"})

    mocks.get_credential.assert_not_awaited()
    mocks.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_unparseable_body_never_raises(mocks):
    await handle_github_event(b"not json", {"x-github-event": "push"})

    mocks.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_push_tracks_the_selected_branch_instead_of_default(mocks):
    mocks.get_credential.return_value = SimpleNamespace(
        status="active",
        provider_account_id="42",
        provider_metadata={"repository_branches": {"acme/api": "feature/ui"}},
    )
    for branch in ("main", "feature/ui"):
        await handle_github_event(
            _body(
                {
                    "ref": f"refs/heads/{branch}",
                    "installation": {"id": 42},
                    "repository": {"full_name": "acme/api", "default_branch": "main"},
                }
            ),
            {"x-github-event": "push"},
        )
    mocks.sync.assert_awaited_once_with(mocks.get_credential.return_value, ["acme/api"])


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [None, {}])
async def test_push_without_a_stored_branch_selection_follows_the_default(mocks, stored):
    """A null or empty selection map is the same as none: the default branch is tracked."""
    mocks.get_credential.return_value = SimpleNamespace(
        status="active", provider_account_id="42", provider_metadata={"repository_branches": stored}
    )
    await handle_github_event(
        _body(
            {
                "ref": "refs/heads/main",
                "installation": {"id": 42},
                "repository": {"full_name": "acme/api", "default_branch": "main"},
            }
        ),
        {"x-github-event": "push"},
    )
    mocks.sync.assert_awaited_once_with(mocks.get_credential.return_value, ["acme/api"])


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [["acme/api"], "dev"])
async def test_a_selection_that_is_not_a_mapping_ignores_the_push(mocks, stored):
    mocks.get_credential.return_value = SimpleNamespace(
        status="active", provider_account_id="42", provider_metadata={"repository_branches": stored}
    )
    await handle_github_event(
        _body(
            {
                "ref": "refs/heads/main",
                "installation": {"id": 42},
                "repository": {"full_name": "acme/api", "default_branch": "main"},
            }
        ),
        {"x-github-event": "push"},
    )
    mocks.sync.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", ["", "--evil", 0, ["main"], {"a": 1}])
async def test_a_stored_branch_that_is_not_a_valid_name_ignores_the_push(mocks, stored, caplog):
    """The webhook agrees with the sync, which refuses such a repository: it does not
    quietly wait for a branch no push can ever match."""
    mocks.get_credential.return_value = SimpleNamespace(
        status="active",
        provider_account_id="42",
        provider_metadata={"repository_branches": {"acme/api": stored}},
    )
    await handle_github_event(
        _body(
            {
                "ref": "refs/heads/main",
                "installation": {"id": 42},
                "repository": {"full_name": "acme/api", "default_branch": "main"},
            }
        ),
        {"x-github-event": "push"},
    )
    mocks.sync.assert_not_awaited()
    assert "stored branch of acme/api is invalid" in caplog.text
