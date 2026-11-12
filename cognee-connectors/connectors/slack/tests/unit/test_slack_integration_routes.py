"""Slack on the generic integration surface: picker, selection, manual sync, token refresh.

Copied from cognee.tests.unit.modules.integrations.test_slack_integration_routes (cognee dev, 2026-11-12).

The real ``SlackIntegration`` is registered under ``slack`` and the real
``/api/v1/integrations`` router serves it; only the credential store, Slack's channel
listing and the sync itself are stand-ins.
"""

import asyncio
import base64
import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs
from uuid import uuid4

import httpx
import pytest
from cognee.api.v1.integrations.routers.get_integrations_router import get_integrations_router
from cognee.modules.integrations import ingestion
from cognee.modules.integrations.registry import supported_integrations, use_integration
from cognee.modules.users.methods import get_authenticated_user
from fake_slack import FakeSlack, rate_limited
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee_connector_slack import source as slack_source_module
from cognee_connector_slack.cognee_host import adapter
from cognee_connector_slack.cognee_host.adapter import SlackIntegration, access_token_for


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    """No connection is paused unless a test says so (the pause lives in the database)."""
    monkeypatch.setattr(
        "cognee.modules.integrations.ingestion.sync_is_paused", AsyncMock(return_value=False)
    )


USER_ID = uuid4()
TEAM = "T0TEAM"
_router_module = importlib.import_module(
    "cognee.api.v1.integrations.routers.get_integrations_router"
)
INSTALLER = "U0INSTALLER"
CHANNELS = [
    {"id": "C1", "name": "general", "is_private": False, "is_member": True},
    {"id": "C2", "name": "random", "is_private": False, "is_member": False},
    {"id": "G3", "name": "leads", "is_private": True, "is_member": True},
    {"id": "G4", "name": "hr-cases", "is_private": True, "is_member": True},
]


def directory() -> FakeSlack:
    """The members of the private channels: the installer is in G3 only."""
    fake = FakeSlack()
    fake.members = {"G3": ["U0SOMEONE", INSTALLER], "G4": ["U0SOMEONE"]}
    return fake


def make_credential(selection=None, **overrides):
    metadata = {"installed_by_slack_user_id": INSTALLER}
    metadata.update(overrides.pop("metadata", {}))
    if selection is not None:
        metadata["selected_channels"] = selection
    return SimpleNamespace(
        id=uuid4(),
        provider="slack",
        provider_account_id=TEAM,
        user_id=USER_ID,
        workspace_id=None,
        provider_metadata=metadata,
        token_expires_at=overrides.pop("token_expires_at", None),
        **overrides,
    )


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=USER_ID)
    before = dict(supported_integrations)
    supported_integrations.clear()
    use_integration(SlackIntegration())
    yield TestClient(app)
    supported_integrations.clear()
    supported_integrations.update(before)


def connected(credential):
    return patch.object(
        _router_module, "get_active_credential_for_user", AsyncMock(return_value=credential)
    )


def picker(client, credential, fake=None):
    listing = AsyncMock(return_value=CHANNELS)
    with (
        connected(credential),
        patch.object(adapter, "access_token_for", AsyncMock(return_value="xoxb-token")),
        patch.object(adapter, "list_channels", listing),
        patch.object(adapter, "_picker_client", lambda token: fake or directory()),
    ):
        response = client.get("/api/v1/integrations/slack/resources")
    return response, listing


def test_the_picker_lists_the_channels_the_app_is_in_with_the_selection(client):
    response, listing = picker(client, make_credential(["G3"]))

    assert response.status_code == 200
    body = response.json()
    assert [(r["id"], r["name"], r["selected"]) for r in body["resources"]] == [
        ("C1", "#general", False),
        ("G3", "#leads", True),
    ]
    assert body["resources"][1]["attributes"] == {
        "is_private": True,
        "is_archived": False,
        "is_shared": False,
    }
    assert body["selected"] == ["G3"]
    listing.assert_awaited_once_with("xoxb-token", types="public_channel,private_channel")


def test_a_private_channel_the_installer_is_not_in_is_never_named(client):
    """The app may be in private channels the person who connected it cannot see."""
    response, _ = picker(client, make_credential([]))
    assert "G4" not in [r["id"] for r in response.json()["resources"]]
    assert "hr-cases" not in response.text

    no_installer = make_credential([], metadata={"installed_by_slack_user_id": None})
    response, _ = picker(client, no_installer)
    assert [r["id"] for r in response.json()["resources"]] == ["C1"]


def test_a_selected_archived_channel_stays_in_the_picker_marked(client):
    """Slack's channel list leaves archived channels out; a client saving back the listed
    channels would drop a selected one and retire the data the sync keeps for it."""
    fake = directory()
    fake.add_channel("C5", "old-project", is_member=False)  # membership after archiving: unknown
    fake.channels["C5"]["is_archived"] = True
    response, _ = picker(client, make_credential(["G3", "C5", "C9"]), fake)  # C9 was deleted

    resources = response.json()["resources"]
    assert [(r["id"], r["name"], r["selected"]) for r in resources] == [
        ("C1", "#general", False),
        ("G3", "#leads", True),
        ("C5", "#old-project", True),
    ]
    assert resources[2]["attributes"] == {
        "is_private": False,
        "is_archived": True,
        "is_shared": False,
    }
    infos = [params["channel"] for method, params in fake.calls if method == "conversations.info"]
    assert infos == ["C5", "C9"], "only the selected channels the list left out are asked"


def test_a_selected_archived_private_channel_the_installer_is_not_in_is_never_named(client):
    fake = directory()
    fake.add_channel("G6", "old-secret", is_private=True)
    fake.channels["G6"]["is_archived"] = True
    fake.members["G6"] = ["U0SOMEONE"]
    response, _ = picker(client, make_credential(["G6"]), fake)

    assert "G6" not in [r["id"] for r in response.json()["resources"]]
    assert "old-secret" not in response.text


def test_a_selected_private_channel_whose_members_cannot_be_read_stays_in_the_picker(client):
    """A rate limit on the member list must not drop a selected channel from the list: a
    client saving back the listed channels would deselect it, and the sync that saving
    starts retires its data. The owner picked it, so its name is no news to them."""
    fake = directory()
    fake.fail("conversations.members", rate_limited("conversations.members"))
    response, _ = picker(client, make_credential(["G3"]), fake)

    assert [(r["id"], r["selected"]) for r in response.json()["resources"]] == [
        ("C1", False),
        ("G3", True),
    ]


def test_a_rate_limited_channel_list_answers_429_with_slacks_retry_after(client):
    """The picker does not wait out Slack's limit inside the request; it says when to retry."""
    from cognee.modules.integrations.ingestion import RateLimitedError

    limited = RateLimitedError("Slack conversations.list is rate limited")
    limited.retry_after = 30
    with (
        connected(make_credential([])),
        patch.object(adapter, "access_token_for", AsyncMock(return_value="xoxb-token")),
        patch.object(adapter, "list_channels", AsyncMock(side_effect=limited)),
    ):
        response = client.get("/api/v1/integrations/slack/resources")

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "30"


def test_the_picker_marks_a_channel_shared_with_another_organization(client):
    listing = AsyncMock(
        return_value=[
            {"id": "C1", "name": "general", "is_private": False, "is_member": True},
            {
                "id": "C7",
                "name": "acme-partners",
                "is_private": False,
                "is_member": True,
                "is_shared": True,
            },
        ]
    )
    with (
        connected(make_credential([])),
        patch.object(adapter, "access_token_for", AsyncMock(return_value="xoxb-token")),
        patch.object(adapter, "list_channels", listing),
        patch.object(adapter, "_picker_client", lambda token: directory()),
    ):
        resources = client.get("/api/v1/integrations/slack/resources").json()["resources"]

    shared = {r["id"]: r["attributes"].get("is_shared") for r in resources}
    assert shared == {"C1": False, "C7": True}


def test_the_picker_ignores_the_command_allowlist(client):
    """The allowlist decides where Cognee's commands work; which channels are synced is
    the picker's alone."""
    response, _ = picker(client, make_credential([], metadata={"allowed_channel_ids": ["G3"]}))
    assert [r["id"] for r in response.json()["resources"]] == ["C1", "G3"]


def test_a_private_channel_whose_members_cannot_be_read_is_hidden_without_waiting(
    client, monkeypatch, caplog
):
    """A rate limit is not waited out inside the picker's HTTP request, and one unreadable
    channel does not fail the list. Goes through the real client, one request per channel."""
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        channel = parse_qs(request.content.decode())["channel"][0]
        requests.append(channel)
        if channel == "G4":
            return httpx.Response(429, headers={"Retry-After": "1"})
        if channel == "G5":
            return httpx.Response(200, json={"ok": False, "error": "channel_not_found"})
        return httpx.Response(200, json={"ok": True, "members": [INSTALLER]})

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler))
    )
    listing = AsyncMock(
        return_value=[
            *CHANNELS,
            {"id": "G5", "name": "gone-soon", "is_private": True, "is_member": True},
        ]
    )
    with (
        connected(make_credential([])),
        patch.object(adapter, "access_token_for", AsyncMock(return_value="xoxb-token")),
        patch.object(adapter, "list_channels", listing),
    ):
        response = client.get("/api/v1/integrations/slack/resources")

    assert response.status_code == 200
    assert [r["id"] for r in response.json()["resources"]] == ["C1", "G3"]
    assert sorted(requests) == ["G3", "G4", "G5"], "no retry, so no Retry-After sleep"
    assert "hid 2 private channel(s)" in caplog.text
    for secret in ("G4", "G5", "hr-cases", "gone-soon"):
        assert secret not in caplog.text


def test_the_sdk_app_manifest_asks_for_exactly_the_bot_scopes():
    from pathlib import Path

    import yaml

    from cognee_connector_slack.cognee_host import oauth

    manifest = Path(__file__).parents[2] / "manifest.yml"
    scopes = yaml.safe_load(manifest.read_text())["oauth_config"]["scopes"]["bot"]
    assert sorted(scopes) == sorted(oauth._BOT_SCOPES.split(","))


def test_an_app_with_message_shortcuts_asks_for_the_commands_scope():
    """Slack shows a shortcut only to an app granted ``commands`` ("your app must have the
    commands permission scope", docs.slack.dev shortcuts); without it "Remember this" and
    "Forget this" never appear (G54)."""
    from cognee_connector_slack.cognee_host import oauth

    assert "commands" in oauth._BOT_SCOPES.split(",")


def test_the_sdk_app_manifest_subscribes_to_every_event_the_app_acts_on():
    """An app built from it must receive @cognee, direct messages, and the edits and
    deletions that forget a remembered copy (G45); there are no slash commands."""
    from pathlib import Path

    import yaml

    package = Path(__file__).parents[2]
    manifest = yaml.safe_load((package / "manifest.yml").read_text())
    assert sorted(manifest["settings"]["event_subscriptions"]["bot_events"]) == [
        "app_home_opened",
        "app_mention",
        "app_uninstalled",
        "message.channels",
        "message.groups",
        "message.im",
        "tokens_revoked",
    ]
    assert "slash_commands" not in manifest["features"]
    assert sorted(shortcut["callback_id"] for shortcut in manifest["features"]["shortcuts"]) == [
        "forget_this",
        "remember_this",
    ]
    assert [path.name for path in package.glob("*.yml")] == ["manifest.yml"], "one manifest"


def test_a_selection_is_stored_and_null_is_refused(client):
    stored = SimpleNamespace(provider_metadata={"selected_channels": ["C1"]})
    written = []

    async def change(provider, account_id, compute):
        # The route computes the patch from what is stored, under the connection's lock.
        written.append((provider, account_id, compute(stored.provider_metadata)))
        return stored

    with (
        connected(make_credential([])),
        patch.object(_router_module, "change_provider_metadata", change),
    ):
        refused = client.put("/api/v1/integrations/slack/resources", json={"resourceIds": None})
        saved = client.put(
            "/api/v1/integrations/slack/resources", json={"resourceIds": ["C1", "C1"]}
        )

    assert refused.status_code == 422
    assert saved.status_code == 200
    assert written == [("slack", TEAM, {"selected_channels": ["C1"]})]


@pytest.mark.parametrize("bad_id", ["D0DM", "general", "#general", "c1"])
def test_a_selection_that_is_not_channel_ids_is_refused_and_nothing_is_stored(client, bad_id):
    """One id that is not a channel's would stop every later sync before any channel runs,
    so a deselected channel's documents would stay (G51)."""
    update = AsyncMock()
    with (
        connected(make_credential(["C1"])),
        patch.object(_router_module, "change_provider_metadata", update),
    ):
        response = client.put(
            "/api/v1/integrations/slack/resources", json={"resourceIds": ["C1", bad_id]}
        )

    assert response.status_code == 422
    assert bad_id in response.json()["detail"]
    update.assert_not_awaited()


def test_a_manual_sync_runs_the_slack_sync(client):
    credential = make_credential(["C1"])
    sync = AsyncMock()
    with connected(credential), patch.object(adapter, "sync_slack", sync):
        response = client.post("/api/v1/integrations/slack/sync")

    assert response.json() == {"accepted": True}
    sync.assert_awaited_once_with(credential)


@pytest.mark.asyncio
async def test_a_manual_sync_of_an_empty_selection_reads_nothing_from_slack():
    credential = make_credential([])
    retire = AsyncMock()
    record = AsyncMock()
    with (
        patch.object(ingestion, "require_active_credential", AsyncMock(side_effect=lambda c: c)),
        patch.object(ingestion, "retire_resources", retire),
        patch.object(slack_source_module, "build_slack_service") as build_service,
        patch.object(slack_source_module, "slack_source") as source_factory,
        patch("cognee.modules.integrations.credentials.record_sync_result", record),
    ):
        await SlackIntegration().sync_now(credential)

    build_service.assert_not_called()
    source_factory.assert_not_called()
    assert retire.await_args.args[3] == set()
    assert record.await_args.kwargs["status"] == "ok"


def test_a_disconnect_can_delete_the_workspace_dataset():
    credential = make_credential([])
    assert SlackIntegration().dataset_name(credential) == ingestion.dataset_name_for_account(
        "slack", "", TEAM
    )


def test_a_new_connection_selects_nothing_and_asks_for_users_read():
    from cognee_connector_slack.cognee_host import oauth

    installation = SlackIntegration().parse_installation(
        {"team": {"id": TEAM}, "access_token": "xoxb", "authed_user": {"id": "U1"}}
    )
    assert installation.provider_metadata["selected_channels"] == []
    assert "users:read" in oauth._BOT_SCOPES.split(",")
    assert SlackIntegration.resource_selection_allows_all is False


# --- access_token_for ---


@pytest.mark.asyncio
async def test_a_token_without_an_expiry_is_never_refreshed():
    refresh = AsyncMock()
    with (
        patch.object(adapter, "require_active_credential", AsyncMock(side_effect=lambda c: c)),
        patch.object(adapter, "decrypt_token_payload", return_value={"access_token": "tok"}),
        patch.object(SlackIntegration, "refresh", refresh),
    ):
        assert await access_token_for(make_credential()) == "tok"
    refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_callers_that_find_the_token_expiring_refresh_it_once():
    """Slack rotates the refresh token, so a second refresh with the spent one would fail."""
    rotated = {"done": False}

    async def read_back(credential):
        credential.token_expires_at = datetime.now(timezone.utc) + (
            timedelta(hours=12) if rotated["done"] else timedelta(seconds=30)
        )
        return credential

    async def refresh(credential):
        await asyncio.sleep(0.01)
        rotated["done"] = True

    refreshed = AsyncMock(side_effect=refresh)
    with (
        patch.object(adapter, "require_active_credential", AsyncMock(side_effect=read_back)),
        patch.object(adapter, "decrypt_token_payload", return_value={"access_token": "tok"}),
        patch.object(SlackIntegration, "refresh", refreshed),
        patch.object(adapter, "_refresh_locks", {}),
    ):
        first, second = await asyncio.gather(
            access_token_for(make_credential()), access_token_for(make_credential())
        )

    assert first == second == "tok"
    refreshed.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_credential_without_an_access_token_names_the_reconnect():
    with (
        patch.object(adapter, "require_active_credential", AsyncMock(side_effect=lambda c: c)),
        patch.object(adapter, "decrypt_token_payload", return_value={}),
        pytest.raises(RuntimeError, match="reconnect Slack"),
    ):
        await access_token_for(make_credential())


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/v1/slack/history/T0TEAM"),
        ("POST", "/api/v1/slack/history/T0TEAM/import"),
        ("PUT", "/api/v1/slack/history/T0TEAM/sync"),
    ],
)
def test_the_former_history_routes_are_gone(method, path):
    from cognee_connector_slack.cognee_host.routers import get_slack_channels_router

    app = FastAPI()
    app.include_router(get_slack_channels_router(), prefix="/api/v1/slack")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=USER_ID)

    assert TestClient(app).request(method, path, json={}).status_code == 404


def test_an_uninstalled_workspace_says_when_its_data_is_deleted():
    """G77: 14 days after the uninstall, unless the workspace is connected again; a token
    revoked without an uninstall deletes nothing."""
    from datetime import datetime, timezone

    from cognee_connector_slack.cognee_host import sync as sync_module

    uninstalled = make_credential([])
    uninstalled.provider_metadata[sync_module.UNINSTALLED_AT_KEY] = "2026-11-07T10:00:00+00:00"
    assert SlackIntegration().data_deleted_after(uninstalled) == datetime(
        2026, 11, 21, 10, tzinfo=timezone.utc
    )
    assert SlackIntegration().data_deleted_after(make_credential([])) is None


# -- Review 2026-11-11 (F3): red until fixed ---------------------------------------------


@pytest.mark.parametrize("integration_name", ["SlackIntegration", "SlackMemberIntegration"])
def test_a_state_signed_with_the_slack_signing_secret_is_refused(monkeypatch, integration_name):
    """F3: the OAuth state was signed with the Slack signing secret, which Slack and every
    collaborator of the Slack app can read. With it anyone can mint a state carrying another
    cognee user's id and their own browser nonce (so the nonce check passes), finish Sign in
    with Slack with their own identity, and have ``@cognee`` answer them from that user's
    memory. A state must be signed with a key only this server holds."""
    from cognee.modules.integrations.crypto import oauth_state_signing_key

    from cognee_connector_slack import settings as slack_settings
    from cognee_connector_slack.cognee_host import adapter, member_link
    from cognee_connectors_core.oauth_state import make_state, validate_state

    monkeypatch.setenv("INTEGRATION_CREDENTIALS_KEY", base64.b64encode(b"k" * 32).decode())
    monkeypatch.setattr(slack_settings.slack_settings, "signing_secret", "known-to-slack")
    integration = {
        "SlackIntegration": adapter.SlackIntegration,
        "SlackMemberIntegration": member_link.SlackMemberIntegration,
    }[integration_name]()
    forged = make_state(str(uuid4()), "attackers-nonce", signing_secret="known-to-slack")

    fields = validate_state(forged, signing_secret=oauth_state_signing_key(), field_count=2)

    assert not isinstance(fields, tuple)
    assert not hasattr(integration, "state_signing_secret")
