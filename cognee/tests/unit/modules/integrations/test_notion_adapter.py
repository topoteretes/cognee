"""Notion adapter: what a connect stores, how tokens rotate, what the picker offers."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from cognee.modules.integrations.notion import adapter, client
from cognee.modules.integrations.notion.adapter import NotionIntegration, access_token_for


@pytest.fixture(autouse=True)
def active_credential():
    with patch.object(adapter, "require_active_credential", AsyncMock(side_effect=lambda c: c)):
        yield


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    settings = "cognee.modules.integrations.notion.notion_settings.notion_settings"
    monkeypatch.setattr(f"{settings}.client_id", "n_client")
    monkeypatch.setattr(f"{settings}.client_secret", "shhh")
    monkeypatch.setattr(
        f"{settings}.redirect_uri", "http://localhost:8000/api/v1/integrations/notion/callback"
    )
    monkeypatch.setattr(f"{settings}.state_secret", "state-secret")
    monkeypatch.setattr(f"{settings}.frontend_base_url", "http://localhost:3000")


def token_response(**overrides):
    return {
        "access_token": "secret_access",
        "refresh_token": "secret_refresh",
        "workspace_id": "ws-1",
        "workspace_name": "Topoteretes",
        "workspace_icon": "https://example.com/icon.png",
        "bot_id": "bot-1",
        **overrides,
    }


def make_credential(**overrides):
    credential = SimpleNamespace(
        provider_account_id="ws-1",
        user_id="user-1",
        account_label="Topoteretes",
        scopes=None,
        token_expires_at=None,
        provider_metadata={"selected_roots": []},
    )
    for key, value in overrides.items():
        setattr(credential, key, value)
    return credential


def test_authorize_url_asks_for_a_user_owned_code_grant():
    url = urlsplit(NotionIntegration().authorize_url("the-state"))
    query = {key: value[0] for key, value in parse_qs(url.query).items()}

    assert url.netloc == "api.notion.com" and url.path == "/v1/oauth/authorize"
    assert query == {
        "client_id": "n_client",
        "redirect_uri": "http://localhost:8000/api/v1/integrations/notion/callback",
        "response_type": "code",
        "owner": "user",
        "state": "the-state",
    }


def test_the_workspace_is_the_account_and_secrets_stay_in_the_payload():
    installation = NotionIntegration().parse_installation(token_response())

    assert installation.provider_account_id == "ws-1"
    assert installation.account_label == "Topoteretes"
    assert installation.token_payload == {
        "access_token": "secret_access",
        "refresh_token": "secret_refresh",
    }
    assert installation.provider_metadata == {
        "bot_id": "bot-1",
        "workspace_icon": "https://example.com/icon.png",
        "selected_roots": [],
    }
    assert "secret" not in str(installation.provider_metadata)


def test_a_token_without_an_expiry_is_stored_without_one():
    installation = NotionIntegration().parse_installation(
        token_response(refresh_token=None, expires_in=None)
    )

    assert installation.token_expires_at is None
    assert installation.token_payload == {"access_token": "secret_access"}


def test_an_expiry_in_the_response_is_kept():
    installation = NotionIntegration().parse_installation(token_response(expires_in=3600))

    assert installation.token_expires_at > datetime.now(timezone.utc) + timedelta(minutes=59)


def test_a_response_without_a_workspace_is_refused():
    with pytest.raises(ValueError, match="workspace_id"):
        NotionIntegration().parse_installation(token_response(workspace_id=None))


def test_two_workspaces_are_two_accounts_with_two_datasets():
    integration = NotionIntegration()
    first = integration.parse_installation(token_response())
    second = integration.parse_installation(token_response(workspace_id="ws-2"))

    assert first.provider_account_id != second.provider_account_id
    names = {
        integration.dataset_name(
            make_credential(
                provider_account_id=installation.provider_account_id,
                account_label=installation.account_label,
            )
        )
        for installation in (first, second)
    }
    assert len(names) == 2 and all(name.startswith("notion_account_") for name in names)


def test_renaming_the_workspace_does_not_move_its_dataset():
    integration = NotionIntegration()

    before = integration.dataset_name(make_credential(account_label="Topoteretes"))
    after = integration.dataset_name(make_credential(account_label="Topoteretes Renamed"))
    unnamed = integration.dataset_name(make_credential(account_label=None))

    assert before == after == unnamed


@pytest.mark.asyncio
async def test_a_token_without_an_expiry_is_never_refreshed():
    credential = make_credential(token_expires_at=None)
    with (
        patch.object(adapter, "decrypt_token_payload", return_value={"access_token": "tok"}),
        patch.object(NotionIntegration, "refresh", AsyncMock()) as refresh,
    ):
        assert await access_token_for(credential) == "tok"

    refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_token_near_its_expiry_is_refreshed_first_and_read_back():
    credential = make_credential(token_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1))
    payloads = iter([{"access_token": "new"}])
    with (
        patch.object(adapter, "decrypt_token_payload", side_effect=lambda c: next(payloads)),
        patch.object(NotionIntegration, "refresh", AsyncMock()) as refresh,
    ):
        assert await access_token_for(credential) == "new"

    refresh.assert_awaited_once_with(credential)


@pytest.mark.asyncio
async def test_a_refresh_stores_the_rotated_pair_and_the_new_expiry():
    credential = make_credential()
    refreshed = {"access_token": "new_a", "refresh_token": "new_r", "expires_in": 3600}
    with (
        patch.object(adapter, "decrypt_token_payload", return_value={"refresh_token": "old_r"}),
        patch.object(client, "refresh_access_token", AsyncMock(return_value=refreshed)) as call,
        patch.object(adapter, "update_refreshed_credential", AsyncMock()) as store,
    ):
        await NotionIntegration().refresh(credential)

    call.assert_awaited_once_with("old_r", client_id="n_client", client_secret="shhh")
    kwargs = store.await_args.kwargs
    assert kwargs["token_payload"] == {"access_token": "new_a", "refresh_token": "new_r"}
    assert kwargs["token_expires_at"] > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_a_refresh_without_a_new_refresh_token_keeps_the_old_one():
    refreshed = {"access_token": "new_a", "refresh_token": None, "expires_in": None}
    with (
        patch.object(adapter, "decrypt_token_payload", return_value={"refresh_token": "old_r"}),
        patch.object(client, "refresh_access_token", AsyncMock(return_value=refreshed)),
        patch.object(adapter, "update_refreshed_credential", AsyncMock()) as store,
    ):
        await NotionIntegration().refresh(make_credential())

    assert store.await_args.kwargs["token_payload"]["refresh_token"] == "old_r"
    assert store.await_args.kwargs["token_expires_at"] is None


@pytest.mark.asyncio
async def test_a_refresh_with_no_refresh_token_names_the_reconnect():
    with (
        patch.object(adapter, "decrypt_token_payload", return_value={"access_token": "a"}),
        pytest.raises(RuntimeError, match="must reconnect"),
    ):
        await NotionIntegration().refresh(make_credential())


@pytest.mark.asyncio
async def test_an_invalid_grant_revokes_the_local_credential_and_raises():
    error = client.NotionError("token refresh", "invalid_grant")
    with (
        patch.object(adapter, "decrypt_token_payload", return_value={"refresh_token": "r"}),
        patch.object(client, "refresh_access_token", AsyncMock(side_effect=error)),
        patch.object(adapter, "revoke_credential_if_current", AsyncMock()) as revoke,
        pytest.raises(client.NotionError),
    ):
        await NotionIntegration().refresh(make_credential())

    revoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_any_other_refresh_error_leaves_the_credential_alone():
    error = client.NotionError("token refresh", "http_503")
    with (
        patch.object(adapter, "decrypt_token_payload", return_value={"refresh_token": "r"}),
        patch.object(client, "refresh_access_token", AsyncMock(side_effect=error)),
        patch.object(adapter, "revoke_credential_if_current", AsyncMock()) as revoke,
        pytest.raises(client.NotionError),
    ):
        await NotionIntegration().refresh(make_credential())

    revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_revoke_of_one_token_does_not_stop_the_other():
    revoke = AsyncMock(side_effect=[RuntimeError("down"), None])
    payload = {"access_token": "a", "refresh_token": "r"}
    with (
        patch.object(adapter, "decrypt_token_payload", return_value=payload),
        patch.object(client, "revoke_token", revoke),
    ):
        await NotionIntegration().revoke_remote(make_credential())

    assert [call.args[0] for call in revoke.await_args_list] == ["a", "r"]


def hit(object_id, kind="page", parent_id=None, title="T"):
    return {
        "id": object_id,
        "kind": kind,
        "title": title,
        "parent_kind": "page_id" if parent_id else "workspace",
        "parent_id": parent_id,
    }


@pytest.mark.asyncio
async def test_the_picker_offers_only_roots_whose_parent_is_not_shared():
    hits = [
        hit("AAAA-1111-2222-3333-4444-5555-6666-7777"),
        hit("child", parent_id="aaaa1111222233334444555566667777"),
        hit("ds", kind="data_source", parent_id="outside-the-share", title=""),
    ]
    with (
        patch.object(adapter, "access_token_for", AsyncMock(return_value="tok")),
        patch.object(client, "search_shared", AsyncMock(return_value=hits)),
    ):
        resources = await NotionIntegration().list_resources(make_credential())

    # Ids come out in the one spelling the sync hashes, whatever Notion returned.
    assert [(item["id"], item["name"]) for item in resources] == [
        ("page:aaaa1111222233334444555566667777", "T"),
        ("data_source:ds", "Untitled"),
    ]
    assert resources[1]["attributes"] == {
        "kind": "data_source",
        "parent_kind": "page_id",
        "parent_id": "outside-the-share",
    }


def test_nothing_syncs_until_roots_are_picked():
    assert NotionIntegration.resource_selection_allows_all is False
    assert NotionIntegration.resource_selection_key == "selected_roots"


def expiring_credential():
    return make_credential(token_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1))


@pytest.mark.asyncio
async def test_two_callers_that_find_the_token_expiring_refresh_it_once():
    """The picker and a background sync arrive together. Notion rotates the refresh token,
    so a second refresh with the spent one would fail and could revoke a healthy connection."""
    rotated = {"done": False}

    async def read_back(credential):
        # The row reads as expiring until the first refresh has written its rotation.
        credential.token_expires_at = (
            datetime.now(timezone.utc) + timedelta(hours=1)
            if rotated["done"]
            else datetime.now(timezone.utc) + timedelta(minutes=1)
        )
        return credential

    async def refresh(credential):
        await asyncio.sleep(0.01)  # a real refresh yields to the loop while Notion answers
        rotated["done"] = True

    refreshed = AsyncMock(side_effect=refresh)
    with (
        patch.object(adapter, "require_active_credential", AsyncMock(side_effect=read_back)),
        patch.object(adapter, "decrypt_token_payload", return_value={"access_token": "tok"}),
        patch.object(NotionIntegration, "refresh", refreshed),
        patch.object(adapter, "_refresh_locks", {}),
    ):
        first, second = await asyncio.gather(
            access_token_for(expiring_credential()), access_token_for(expiring_credential())
        )

    assert first == second == "tok"
    refreshed.assert_awaited_once()


@pytest.mark.asyncio
async def test_workspaces_do_not_wait_for_each_others_refresh():
    started = asyncio.Event()
    release = asyncio.Event()

    async def read_back(credential):
        return credential

    async def slow_refresh(self, credential):
        if credential.provider_account_id == "ws-1":
            started.set()
            await release.wait()

    with (
        patch.object(adapter, "require_active_credential", AsyncMock(side_effect=read_back)),
        patch.object(adapter, "decrypt_token_payload", return_value={"access_token": "tok"}),
        patch.object(NotionIntegration, "refresh", slow_refresh),
        patch.object(adapter, "_refresh_locks", {}),
    ):
        blocked = asyncio.create_task(access_token_for(expiring_credential()))
        await started.wait()
        other = expiring_credential()
        other.provider_account_id = "ws-2"
        # ws-2 is not behind ws-1's lock: it refreshes and returns while ws-1 is still waiting.
        assert await asyncio.wait_for(access_token_for(other), timeout=1) == "tok"
        release.set()
        await blocked
