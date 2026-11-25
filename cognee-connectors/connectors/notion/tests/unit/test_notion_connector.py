"""Contract tests for the Notion connector: ``CONNECTOR`` against Notion's real shapes.

The token-response fixture is the one the adapter tests carry; the token endpoint is
faked the way the client tests fake it (``FakeSession`` standing in for
``aiohttp.ClientSession``), because Notion's exchange goes through the client, not
through the core's form-encoded ``HttpClient``. A ``FakeHttpClient`` is wired in
anyway, to prove exactly that: it never sees a call.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest
from test_notion_client import BASIC_ID_SECRET, FakeSession

from cognee_connector_notion import CONNECTOR, client
from cognee_connector_notion.connector import DESCRIPTOR, POLL_INTERVAL_SECONDS, NotionConnector
from cognee_connector_notion.workspace import dataset_name_for_workspace
from cognee_connectors_core import registry
from cognee_connectors_core.base import ConnectorResource
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient, FakeSettings

WORKSPACE = "ws-1"
# cognee's name for WORKSPACE's dataset: ``notion_account_`` plus 10 hex of the id. A
# literal, because the name is a stored identifier that must survive every refactor.
DATASET = "notion_account_769446044a"


def nid(number: int) -> str:
    """A Notion id in its canonical spelling: 32 lowercase hex digits."""
    return f"{number:032x}"


def page(number: int) -> str:
    return f"page:{nid(number)}"


def source(number: int) -> str:
    return f"data_source:{nid(number)}"


def token_response(**overrides):
    return {
        "access_token": "secret_access",
        "refresh_token": "secret_refresh",
        "workspace_id": WORKSPACE,
        "workspace_name": "Topoteretes",
        "workspace_icon": "https://example.com/icon.png",
        "bot_id": "bot-1",
        **overrides,
    }


def settings(**overrides):
    values = {
        "client_id": "n_client",
        "client_secret": "shhh",
        "redirect_uri": "http://localhost:8000/api/v1/integrations/notion/callback",
        "frontend_base_url": "http://localhost:3000",
        "env_prefix": "NOTION_",
    }
    values.update(overrides)
    return FakeSettings(**values)


def connector(http=None, **overrides) -> NotionConnector:
    return NotionConnector(DESCRIPTOR, settings=settings(**overrides), http=http)


def credential(**overrides) -> FakeCredential:
    values = {
        "provider": "notion",
        "provider_account_id": WORKSPACE,
        "account_label": "Topoteretes",
        "provider_metadata": {"selected_roots": []},
    }
    values.update(overrides)
    return FakeCredential(**values)


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.reset_registry()
    yield
    registry.reset_registry()


@pytest.fixture
def accepted_token():
    """Notion accepts whatever token ``sync_scope`` checks; the mock records the check."""
    with patch.object(client, "verify_token", AsyncMock()) as verify:
        yield verify


# ---- Identity and catalog ---------------------------------------------------------


def test_the_descriptor_names_notion_as_a_polled_oauth2_provider():
    assert (DESCRIPTOR.provider, DESCRIPTOR.display_name, DESCRIPTOR.auth) == (
        "notion",
        "Notion",
        "oauth2",
    )
    assert DESCRIPTOR.authorize_url == "https://api.notion.com/v1/oauth/authorize"
    assert DESCRIPTOR.token_url == client.TOKEN_URL
    assert DESCRIPTOR.signature_scheme is None
    assert DESCRIPTOR.poll_interval_seconds == POLL_INTERVAL_SECONDS


def test_the_connector_is_tier_two_with_a_picker_and_no_webhooks():
    assert (CONNECTOR.provider, CONNECTOR.display_name, CONNECTOR.auth) == (
        "notion",
        "Notion",
        "oauth2",
    )
    assert isinstance(CONNECTOR, NotionConnector)
    assert registry.tier_of(CONNECTOR) == 2
    assert CONNECTOR.webhook_verifier() is None
    assert type(CONNECTOR).supports_resource_selection() is True
    assert CONNECTOR.poll_interval_seconds == POLL_INTERVAL_SECONDS
    assert CONNECTOR.poll_targets(credential()) == []
    assert CONNECTOR.content_type is None


def test_registering_the_connector_lists_it_in_the_catalog():
    registry.use_connector(CONNECTOR)

    assert registry.get_connector("notion") is CONNECTOR
    (entry,) = registry.catalog()
    assert entry == {
        "provider": "notion",
        "display_name": "Notion",
        "auth": "oauth2",
        "tier": 2,
        "has_webhooks": False,
        "supports_resource_selection": True,
        "poll_interval_seconds": POLL_INTERVAL_SECONDS,
        "content_type": None,
        "degraded": False,
    }


def test_the_connector_reads_the_real_settings_object():
    from cognee_connector_notion.settings import notion_settings

    assert CONNECTOR.settings is notion_settings


# ---- Connect -----------------------------------------------------------------------


def test_authorize_url_asks_for_a_user_owned_code_grant():
    url = urlsplit(connector().authorize_url("the-state"))
    query = {key: value[0] for key, value in parse_qs(url.query).items()}

    assert url.netloc == "api.notion.com" and url.path == "/v1/oauth/authorize"
    assert query == {
        "client_id": "n_client",
        "redirect_uri": "http://localhost:8000/api/v1/integrations/notion/callback",
        "response_type": "code",
        "owner": "user",
        "state": "the-state",
    }


def test_a_missing_setting_is_named_by_its_environment_variable():
    with pytest.raises(RuntimeError, match="NOTION_CLIENT_ID is not configured"):
        connector(client_id="").authorize_url("s")


@pytest.mark.asyncio
async def test_the_exchange_uses_basic_auth_and_a_json_body_not_the_form_client():
    http = FakeHttpClient()
    session = FakeSession([(200, token_response(), None)])
    with patch.object(client.aiohttp, "ClientSession", return_value=session):
        body = await connector(
            http=http, client_id="id", client_secret="secret", redirect_uri="http://cb"
        ).exchange_callback("the-code", {"state": "ignored"})

    _method, url, kwargs = session.requests[0]
    assert url == client.TOKEN_URL
    assert kwargs["json"] == {
        "grant_type": "authorization_code",
        "code": "the-code",
        "redirect_uri": "http://cb",
    }
    assert kwargs["headers"]["Authorization"] == BASIC_ID_SECRET
    assert kwargs["headers"]["Notion-Version"] == client.NOTION_VERSION
    assert body["workspace_id"] == WORKSPACE and body["access_token"] == "secret_access"
    # Nothing went through the core's form-encoded token client.
    assert http.calls == []


@pytest.mark.asyncio
async def test_a_rejected_exchange_is_a_runtime_error_without_the_body():
    body = {"error": "invalid_grant", "error_description": "secret workspace content"}
    session = FakeSession([(400, body, None)])
    with (
        patch.object(client.aiohttp, "ClientSession", return_value=session),
        pytest.raises(RuntimeError) as caught,
    ):
        await connector().exchange_callback("bad-code", {})

    assert isinstance(caught.value, client.NotionError) and caught.value.code == "invalid_grant"
    assert "secret" not in str(caught.value)


def test_the_workspace_is_the_account_and_secrets_stay_in_the_payload():
    installation = CONNECTOR.parse_installation(token_response())

    assert installation.provider_account_id == WORKSPACE
    assert installation.account_label == "Topoteretes"
    assert installation.auth_type == "oauth2"
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
    assert installation.token_expires_at is None


def test_a_token_without_an_expiry_is_stored_without_one():
    installation = CONNECTOR.parse_installation(token_response(refresh_token=None, expires_in=None))

    assert installation.token_expires_at is None
    assert installation.token_payload == {"access_token": "secret_access"}


def test_an_expiry_in_the_response_is_kept():
    installation = CONNECTOR.parse_installation(token_response(expires_in=3600))

    assert installation.token_expires_at > datetime.now(timezone.utc) + timedelta(minutes=59)


def test_a_response_without_a_workspace_is_refused():
    with pytest.raises(RuntimeError, match="workspace_id"):
        CONNECTOR.parse_installation(token_response(workspace_id=None))


def test_two_workspaces_are_two_accounts_with_two_datasets():
    first = CONNECTOR.parse_installation(token_response())
    second = CONNECTOR.parse_installation(token_response(workspace_id="ws-2"))

    assert first.provider_account_id != second.provider_account_id
    names = {
        CONNECTOR.dataset_name(
            credential(
                provider_account_id=installation.provider_account_id,
                account_label=installation.account_label,
            )
        )
        for installation in (first, second)
    }
    assert len(names) == 2 and all(name.startswith("notion_account_") for name in names)


def test_renaming_the_workspace_does_not_move_its_dataset():
    before = CONNECTOR.dataset_name(credential(account_label="Topoteretes"))
    after = CONNECTOR.dataset_name(credential(account_label="Topoteretes Renamed"))
    unnamed = CONNECTOR.dataset_name(credential(account_label=None))

    assert before == after == unnamed == DATASET


def test_the_dataset_is_named_as_cognees_adapter_names_it():
    assert CONNECTOR.dataset_name(credential()) == DATASET
    assert dataset_name_for_workspace(WORKSPACE) == DATASET
    assert dataset_name_for_workspace("") == "notion_account"


# ---- Refresh and revoke --------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refresh_hands_back_the_rotated_pair_and_the_new_expiry():
    refreshed = {"access_token": "new_a", "refresh_token": "new_r", "expires_in": 3600}
    with patch.object(client, "refresh_access_token", AsyncMock(return_value=refreshed)) as call:
        result = await connector().refresh(credential(), {"refresh_token": "old_r"})

    call.assert_awaited_once_with("old_r", client_id="n_client", client_secret="shhh")
    assert result.token_payload == {"access_token": "new_a", "refresh_token": "new_r"}
    assert result.token_expires_at > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_a_refresh_without_a_new_refresh_token_keeps_the_old_one():
    refreshed = {"access_token": "new_a", "refresh_token": None, "expires_in": None}
    with patch.object(client, "refresh_access_token", AsyncMock(return_value=refreshed)):
        result = await connector().refresh(credential(), {"refresh_token": "old_r"})

    assert result.token_payload["refresh_token"] == "old_r"
    assert result.token_expires_at is None


@pytest.mark.asyncio
async def test_a_payload_with_no_refresh_token_has_nothing_to_refresh():
    with patch.object(client, "refresh_access_token", AsyncMock()) as call:
        assert await connector().refresh(credential(), {"access_token": "a"}) is None

    call.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_invalid_grant_reaches_the_host_as_notions_error():
    error = client.NotionError("token refresh", "invalid_grant")
    with (
        patch.object(client, "refresh_access_token", AsyncMock(side_effect=error)),
        pytest.raises(client.NotionError) as caught,
    ):
        await connector().refresh(credential(), {"refresh_token": "r"})

    assert caught.value.code == "invalid_grant"


@pytest.mark.asyncio
async def test_a_failed_revoke_of_one_token_does_not_stop_the_other():
    revoke = AsyncMock(side_effect=[RuntimeError("down"), None])
    with patch.object(client, "revoke_token", revoke):
        await connector().revoke_remote(credential(), {"access_token": "a", "refresh_token": "r"})

    assert [call.args[0] for call in revoke.await_args_list] == ["a", "r"]
    assert revoke.await_args_list[0].kwargs == {"client_id": "n_client", "client_secret": "shhh"}


# ---- Resources ----------------------------------------------------------------------


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
    with patch.object(client, "search_shared", AsyncMock(return_value=hits)) as search:
        resources = await connector().list_resources(credential(), {"access_token": "tok"})

    search.assert_awaited_once_with("tok")
    assert all(isinstance(resource, ConnectorResource) for resource in resources)
    # Ids come out in the one spelling the sync hashes, whatever Notion returned.
    assert [(item.id, item.name) for item in resources] == [
        ("page:aaaa1111222233334444555566667777", "T"),
        ("data_source:ds", "Untitled"),
    ]
    assert resources[1].description is None
    assert resources[1].attributes == {
        "kind": "data_source",
        "parent_kind": "page_id",
        "parent_id": "outside-the-share",
    }


@pytest.mark.asyncio
async def test_the_picker_refuses_a_payload_without_an_access_token():
    with (
        patch.object(client, "search_shared", AsyncMock()) as search,
        pytest.raises(RuntimeError, match="holds no access token"),
    ):
        await connector().list_resources(credential(), {})

    search.assert_not_awaited()


# ---- Sync scope -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_scope_reads_the_stored_selection_once_per_root_and_drops_nested_ones(
    accepted_token,
):
    dashed = "-".join([nid(1)[:8], nid(1)[8:12], nid(1)[12:16], nid(1)[16:20], nid(1)[20:]])
    # inner page -> block -> outer page, the outer one written in the other spelling
    chain = {
        ("page", nid(1)): ("block", nid(3)),
        ("block", nid(3)): ("page", nid(2).upper()),
        ("page", nid(2)): None,
        ("data_source", nid(4)): None,
    }
    parent = AsyncMock(side_effect=lambda _token, kind, object_id: chain[(kind, object_id)])
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    stored = credential(
        provider_metadata={
            "selected_roots": [page(1), f"page:{dashed.upper()}", source(4), page(2)]
        },
        token_expires_at=expires,
    )
    with patch.object(client, "get_parent", parent):
        scope = await connector().sync_scope(stored, {"access_token": "tok", "refresh_token": "r"})

    assert scope.items == [source(4), page(2)]
    assert scope.dataset_name == DATASET
    assert scope.content_type is None
    assert scope.credential == "tok"
    assert scope.credential_expires_at == expires
    assert scope.credential_refreshable is True
    assert scope.options == {"workspace_id": WORKSPACE}
    assert scope.item_resource_ids is None
    assert parent.await_args_list[0].args == ("tok", "page", nid(1))
    accepted_token.assert_awaited_once_with("tok")


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [{}, {"selected_roots": []}, {"selected_roots": None}])
async def test_sync_scope_without_picked_roots_is_empty_and_only_checks_the_token(
    metadata, accepted_token
):
    with patch.object(client, "get_parent", AsyncMock()) as parent:
        scope = await connector().sync_scope(
            credential(provider_metadata=metadata), {"access_token": "tok"}
        )

    assert scope.items == [] and scope.credential == "tok"
    assert scope.credential_refreshable is False
    # The one request a poll with nothing picked makes: a host learns here that the
    # workspace removed the integration.
    accepted_token.assert_awaited_once_with("tok")
    parent.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_scope_takes_the_roots_a_host_names_over_the_stored_selection(accepted_token):
    with patch.object(client, "get_parent", AsyncMock(return_value=None)):
        scope = await connector().sync_scope(
            credential(provider_metadata={"selected_roots": [page(1)]}),
            {"access_token": "tok"},
            resources=[source(2), source(2)],
        )

    assert scope.items == [source(2)]


@pytest.mark.asyncio
async def test_an_explicit_empty_resources_list_is_refused_like_the_base_class(accepted_token):
    with (
        patch.object(client, "get_parent", AsyncMock()) as parent,
        pytest.raises(ValueError, match="empty"),
    ):
        await connector().sync_scope(
            credential(provider_metadata={"selected_roots": [page(1)]}),
            {"access_token": "tok"},
            resources=[],
        )

    accepted_token.assert_not_awaited()
    parent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection",
    [["plain-id"], ["folder:" + nid(1)], ["page:"], [5], "page:x", ["page:not-a-notion-id"]],
)
async def test_a_malformed_selection_stops_the_scope_before_any_request(selection, accepted_token):
    with (
        patch.object(client, "get_parent", AsyncMock()) as parent,
        pytest.raises(ValueError),
    ):
        await connector().sync_scope(
            credential(provider_metadata={"selected_roots": selection}), {"access_token": "tok"}
        )

    accepted_token.assert_not_awaited()
    parent.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_scope_refuses_a_payload_without_an_access_token(accepted_token):
    with pytest.raises(RuntimeError, match="holds no access token"):
        await connector().sync_scope(
            credential(provider_metadata={"selected_roots": [page(1)]}), {"refresh_token": "r"}
        )

    accepted_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_token_notion_no_longer_accepts_stops_the_scope_before_the_parent_walk():
    """The check runs before any root is looked at, and its error carries the code a host
    revokes on (``client.UNAUTHORIZED_CODES``), as ``refresh()``'s ``invalid_grant`` does."""
    error = client.NotionError("token check", "unauthorized")
    with (
        patch.object(client, "verify_token", AsyncMock(side_effect=error)) as verify,
        patch.object(client, "get_parent", AsyncMock()) as parent,
        pytest.raises(client.NotionError) as caught,
    ):
        await connector().sync_scope(
            credential(provider_metadata={"selected_roots": [page(1)]}), {"access_token": "tok"}
        )

    verify.assert_awaited_once_with("tok")
    assert caught.value is error and caught.value.code in client.UNAUTHORIZED_CODES
    parent.assert_not_awaited()
