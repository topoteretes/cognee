"""Branch controls keep every page and stay within the current installation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.integrations.routers import get_github_branches_router as router_module
from cognee.modules.integrations.github import branches
from cognee.modules.integrations.github import sync as github_sync
from cognee.modules.users.methods import get_authenticated_user


@pytest.fixture
def credential():
    return SimpleNamespace(
        provider_account_id="42",
        provider_metadata={"repository_branches": {"org/other": "release"}},
    )


@pytest.mark.asyncio
async def test_branch_listing_follows_all_pages(monkeypatch, credential):
    monkeypatch.setattr(
        branches, "mint_installation_token", AsyncMock(return_value=("secret", None))
    )
    monkeypatch.setattr(
        branches, "list_installation_repositories", AsyncMock(return_value=["org/repo"])
    )
    responses = []
    for names, next_url in [
        (["main"], "https://api.github.com/repos/org/repo/branches?page=2"),
        (["feature/ui"], None),
    ]:
        response = MagicMock(status=200)
        response.json = AsyncMock(return_value=[{"name": name} for name in names])
        response.links = {"next": {"url": next_url}} if next_url else {}
        response.__aenter__ = AsyncMock(return_value=response)
        responses.append(response)
    session = MagicMock()
    session.get.side_effect = responses
    session.__aenter__ = AsyncMock(return_value=session)
    monkeypatch.setattr(github_sync.aiohttp, "ClientSession", lambda **kwargs: session)
    assert await branches.list_repository_branches(credential, "org/repo") == ["main", "feature/ui"]
    assert session.get.call_count == 2


@pytest.mark.asyncio
async def test_repositories_outside_installation_are_rejected_before_branch_lookup(
    monkeypatch, credential
):
    monkeypatch.setattr(
        branches, "mint_installation_token", AsyncMock(return_value=("secret", None))
    )
    monkeypatch.setattr(
        branches, "list_installation_repositories", AsyncMock(return_value=["org/repo"])
    )
    client = MagicMock()
    monkeypatch.setattr(github_sync.aiohttp, "ClientSession", client)
    with pytest.raises(PermissionError):
        await branches.list_repository_branches(credential, "someone/private")
    client.assert_not_called()


@pytest.fixture
def installation(monkeypatch):
    """The installation covers org/repo and org/other; org/repo has the one branch listed."""
    monkeypatch.setattr(
        branches,
        "_installation_repositories",
        AsyncMock(return_value=("secret", ["org/repo", "org/other"])),
    )
    listing = AsyncMock(return_value=["feature/ui"])
    monkeypatch.setattr(branches, "_list_branches", listing)
    update = AsyncMock()
    monkeypatch.setattr(branches, "update_provider_metadata", update)

    def stored_after(metadata):
        """What the save writes, computed from ``metadata`` as the database holds it."""
        return update.await_args.args[2](metadata)

    return SimpleNamespace(listing=listing, update=update, stored_after=stored_after)


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["feature/ui", None])
async def test_branch_selection_preserves_other_repositories(installation, credential, branch):
    await branches.select_repository_branch(credential, "org/repo", branch)
    expected = {"org/other": "release"}
    if branch is not None:
        expected["org/repo"] = branch
    installation.update.assert_awaited_once()
    assert installation.stored_after(credential.provider_metadata) == {
        "repository_branches": expected
    }


@pytest.mark.asyncio
async def test_a_selection_made_meanwhile_is_not_overwritten(installation, credential):
    """The map is rebuilt from what is stored when the write happens, not from the
    copy read before the GitHub calls."""
    await branches.select_repository_branch(credential, "org/repo", "feature/ui")

    meanwhile = {"repository_branches": {"org/other": "release", "org/third": "dev"}}
    written = installation.stored_after(meanwhile)["repository_branches"]
    # org/third is not in the installation here, so only the pruning drops it.
    assert written == {"org/other": "release", "org/repo": "feature/ui"}
    meanwhile["repository_branches"]["org/repo"] = "old"
    assert installation.stored_after(meanwhile)["repository_branches"]["org/repo"] == "feature/ui"


@pytest.mark.asyncio
async def test_a_connection_that_went_away_is_reported(installation, credential):
    installation.update.return_value = None
    with pytest.raises(PermissionError, match="no longer active"):
        await branches.select_repository_branch(credential, "org/repo", "feature/ui")


@pytest.mark.asyncio
async def test_a_stored_selection_that_is_not_a_mapping_is_refused(installation, credential):
    await branches.select_repository_branch(credential, "org/repo", "feature/ui")
    with pytest.raises(ValueError, match="repository-to-branch mapping"):
        installation.stored_after({"repository_branches": ["org/repo"]})


@pytest.mark.asyncio
async def test_returning_to_the_default_branch_lists_no_branches(installation, credential):
    await branches.select_repository_branch(credential, "org/repo", None)
    installation.listing.assert_not_awaited()


@pytest.mark.asyncio
async def test_selections_of_repositories_the_installation_lost_are_dropped(
    installation, credential
):
    credential.provider_metadata = {
        "repository_branches": {"org/other": "release", "org/gone": "dev"}
    }
    await branches.select_repository_branch(credential, "org/repo", "feature/ui")
    assert installation.stored_after(credential.provider_metadata) == {
        "repository_branches": {"org/other": "release", "org/repo": "feature/ui"}
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["missing", "--evil", ""])
async def test_invalid_or_missing_branch_is_never_persisted(installation, credential, branch):
    with pytest.raises(ValueError):
        await branches.select_repository_branch(credential, "org/repo", branch)
    installation.update.assert_not_awaited()


@pytest.fixture
def client(monkeypatch, credential):
    app = FastAPI()
    app.include_router(router_module.get_github_branches_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id="user")
    monkeypatch.setattr(
        router_module, "get_active_credential_for_user", AsyncMock(return_value=credential)
    )
    return TestClient(app)


def test_branch_routes_resolve_the_authenticated_users_connection(client, monkeypatch, credential):
    listing = AsyncMock(return_value=["main", "dev"])
    save = AsyncMock()
    monkeypatch.setattr(router_module, "list_repository_branches", listing)
    monkeypatch.setattr(router_module, "select_repository_branch", save)
    assert client.get(
        "/api/v1/integrations/github/repositories/branches?repository=org/repo"
    ).json() == {"branches": ["main", "dev"]}
    assert client.put(
        "/api/v1/integrations/github/repositories/branches",
        json={"repository": "org/repo", "branch": "dev"},
    ).json() == {"branch": "dev"}
    save.assert_awaited_once_with(credential, "org/repo", "dev")
    router_module.get_active_credential_for_user.assert_awaited_with("user", "github", None)


def test_branch_routes_reject_disconnected_users(client, monkeypatch):
    monkeypatch.setattr(
        router_module, "get_active_credential_for_user", AsyncMock(return_value=None)
    )
    response = client.get("/api/v1/integrations/github/repositories/branches?repository=org/repo")
    assert response.status_code == 404


def test_branch_save_validation_reaches_the_picker(client, monkeypatch):
    monkeypatch.setattr(
        router_module,
        "select_repository_branch",
        AsyncMock(side_effect=ValueError("Branch not available")),
    )
    response = client.put(
        "/api/v1/integrations/github/repositories/branches",
        json={"repository": "org/repo", "branch": "missing"},
    )
    assert response.status_code == 422
