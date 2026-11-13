"""Unit tests for get_slack_channels_router (the channel-allowlist settings API).

get_active_credential_for_user, decrypt_token_payload, list_channels, and
update_provider_metadata are all patched — these tests only check the
router's own logic: 404 when Slack isn't connected, correct allowed-flag
marking, and a clean 502 (not a raw crash) when Slack rejects the API call.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.slack.routers.get_slack_channels_router import get_slack_channels_router
from cognee.modules.users.methods import get_authenticated_user

USER_ID = uuid4()

# cognee.api.v1.slack.routers's __init__.py does `from .get_slack_channels_router
# import get_slack_channels_router`, rebinding that name on the package to the
# *function* and shadowing the submodule Python auto-attaches there on import.
# String-target patch.object(MODULE, "X") resolves that name differently across
# Python versions (attribute traversal on 3.10, importlib.import_module on
# 3.12+), so it silently returns the function instead of the module on 3.10
# and AttributeErrors. Importing the submodule explicitly sidesteps the
# shadowed attribute entirely.
MODULE = importlib.import_module("cognee.api.v1.slack.routers.get_slack_channels_router")


class _FakeUser:
    id = USER_ID


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_slack_channels_router(), prefix="/api/v1/slack")
    app.dependency_overrides[get_authenticated_user] = lambda: _FakeUser()
    return TestClient(app)


def _credential(allowed_channel_ids=None):
    return SimpleNamespace(
        provider_account_id="T123",
        provider_metadata={"allowed_channel_ids": allowed_channel_ids}
        if allowed_channel_ids
        else {},
    )


def test_get_channels_404s_when_slack_not_connected(client):
    with patch.object(MODULE, "get_active_credential_for_user", new=AsyncMock(return_value=None)):
        response = client.get("/api/v1/slack/channels")
    assert response.status_code == 404


def test_get_channels_marks_allowed_flag_correctly(client):
    credential = _credential(allowed_channel_ids=["C1"])
    with (
        patch.object(
            MODULE, "get_active_credential_for_user", new=AsyncMock(return_value=credential)
        ),
        patch.object(MODULE, "decrypt_token_payload", return_value={"access_token": "xoxb-x"}),
        patch.object(
            MODULE,
            "visible_channels",
            new=AsyncMock(
                return_value=[
                    {"id": "C1", "name": "general", "is_private": False},
                    {"id": "C2", "name": "eng", "is_private": False},
                ]
            ),
        ),
    ):
        response = client.get("/api/v1/slack/channels")

    body = response.json()
    channels_by_id = {c["id"]: c for c in body["channels"]}
    assert channels_by_id["C1"]["allowed"] is True
    assert channels_by_id["C2"]["allowed"] is False


def test_get_channels_surfaces_slack_rejection_as_502(client):
    credential = _credential()
    with (
        patch.object(
            MODULE, "get_active_credential_for_user", new=AsyncMock(return_value=credential)
        ),
        patch.object(MODULE, "decrypt_token_payload", return_value={"access_token": "xoxb-x"}),
        patch.object(
            MODULE, "visible_channels", new=AsyncMock(side_effect=RuntimeError("missing_scope"))
        ),
    ):
        response = client.get("/api/v1/slack/channels")
    assert response.status_code == 502


def test_put_channels_404s_when_slack_not_connected(client):
    with patch.object(MODULE, "get_active_credential_for_user", new=AsyncMock(return_value=None)):
        response = client.put("/api/v1/slack/channels", json={"channelIds": ["C1"]})
    assert response.status_code == 404


def test_put_channels_saves_and_echoes_the_allowlist(client):
    credential = _credential()
    updated = _credential(allowed_channel_ids=["C1", "C2"])
    with (
        patch.object(
            MODULE, "get_active_credential_for_user", new=AsyncMock(return_value=credential)
        ),
        patch.object(MODULE, "decrypt_token_payload", return_value={"access_token": "xoxb-x"}),
        patch.object(
            MODULE,
            "visible_channels",
            new=AsyncMock(return_value=[{"id": "C1"}, {"id": "C2"}]),
        ),
        patch.object(
            MODULE, "update_provider_metadata", new=AsyncMock(return_value=updated)
        ) as update_mock,
    ):
        response = client.put("/api/v1/slack/channels", json={"channelIds": ["C1", "C2"]})

    assert response.status_code == 200
    assert response.json()["allowedChannelIds"] == ["C1", "C2"]
    update_mock.assert_awaited_once_with("slack", "T123", {"allowed_channel_ids": ["C1", "C2"]})


def test_settings_are_off_until_the_owner_turns_them_on(client):
    with patch.object(
        MODULE, "get_active_credential_for_user", new=AsyncMock(return_value=_credential())
    ):
        response = client.get("/api/v1/slack/settings")

    assert response.status_code == 200
    assert response.json() == {"forgetUnavailableChannels": False}


def test_put_settings_404s_when_slack_not_connected(client):
    with patch.object(MODULE, "get_active_credential_for_user", new=AsyncMock(return_value=None)):
        response = client.put("/api/v1/slack/settings", json={"forgetUnavailableChannels": True})
    assert response.status_code == 404


def test_put_settings_saves_the_owners_choice_on_their_connection(client):
    updated = SimpleNamespace(
        provider_account_id="T123", provider_metadata={"forget_unavailable_channels": True}
    )
    with (
        patch.object(
            MODULE, "get_active_credential_for_user", new=AsyncMock(return_value=_credential())
        ) as lookup,
        patch.object(
            MODULE, "update_provider_metadata", new=AsyncMock(return_value=updated)
        ) as update_mock,
    ):
        response = client.put("/api/v1/slack/settings", json={"forgetUnavailableChannels": True})

    assert response.status_code == 200
    assert response.json() == {"forgetUnavailableChannels": True}
    assert lookup.await_args.args[0] == _FakeUser().id
    update_mock.assert_awaited_once_with("slack", "T123", {"forget_unavailable_channels": True})


# --- the allowlist governs commands: it offers what the owner can see, refuses the rest ---

INSTALLER = "U0INSTALLER"

# What conversations.list returns to the bot token. Private channels are listed only
# where the app is a member; C3 is one the installer is not in.
SLACK_CHANNELS = [
    {"id": "C1", "name": "general", "is_private": False, "is_member": True},
    {"id": "C2", "name": "board", "is_private": True, "is_member": True},
    {"id": "C3", "name": "hr", "is_private": True, "is_member": True},
]


@pytest.fixture
def slack(monkeypatch):
    """Slack faked at its HTTP edge: the channel list by type and private members."""
    from cognee.modules.integrations.slack import adapter as adapter_module
    from cognee.modules.integrations.slack import channels as channels_module
    from cognee.tests.unit.tasks.connectors.fake_slack import FakeSlack

    async def fetch_page(session, access_token, params):
        private = {"public_channel": False, "private_channel": True}
        wanted = {private[kind] for kind in params["types"].split(",")}
        return {"ok": True, "channels": [c for c in SLACK_CHANNELS if c["is_private"] in wanted]}

    members = FakeSlack()
    members.members = {"C2": [INSTALLER, "U1"], "C3": ["U1"]}
    monkeypatch.setattr(channels_module, "_fetch_page", fetch_page)
    monkeypatch.setattr(adapter_module, "_picker_client", lambda token: members)
    monkeypatch.setattr(adapter_module, "access_token_for", AsyncMock(return_value="xoxb-test"))
    credential = SimpleNamespace(
        provider_account_id="T123",
        provider_metadata={"installed_by_slack_user_id": INSTALLER},
    )
    with (
        patch.object(
            MODULE, "get_active_credential_for_user", new=AsyncMock(return_value=credential)
        ),
        patch.object(MODULE, "decrypt_token_payload", return_value={"access_token": "xoxb-test"}),
    ):
        yield SimpleNamespace(credential=credential, members=members)


def test_the_allowlist_editor_offers_the_private_channels_the_installer_is_in(client, slack):
    channel_ids = {
        channel["id"] for channel in client.get("/api/v1/slack/channels").json()["channels"]
    }

    assert {"C1", "C2"} <= channel_ids
    # A private channel the installer cannot see: its name never leaves Slack.
    assert "C3" not in channel_ids


@pytest.mark.parametrize(
    "channel_ids",
    [["general"], ["C3"]],
    ids=["a_channel_name_instead_of_its_id", "a_private_channel_the_installer_is_not_in"],
)
def test_the_allowlist_refuses_a_channel_it_does_not_offer(client, slack, channel_ids):
    save = AsyncMock()
    with patch.object(MODULE, "update_provider_metadata", new=save):
        response = client.put("/api/v1/slack/channels", json={"channelIds": channel_ids})

    assert response.status_code == 422
    save.assert_not_awaited()


def test_the_allowlist_accepts_a_private_channel_the_installer_is_in(client, slack):
    updated = SimpleNamespace(provider_metadata={"allowed_channel_ids": ["C1", "C2"]})
    save = AsyncMock(return_value=updated)
    with patch.object(MODULE, "update_provider_metadata", new=save):
        response = client.put("/api/v1/slack/channels", json={"channelIds": ["C1", "C2"]})

    assert response.status_code == 200
    save.assert_awaited_once()


def test_an_allowed_private_channel_whose_members_cannot_be_read_stays_in_the_editor(client, slack):
    """A rate limit on the member list must not drop an allowed channel: saving the editor
    back would take it off the allowlist, and sending it again would be refused."""
    from cognee.tests.unit.tasks.connectors.fake_slack import rate_limited

    slack.credential.provider_metadata["allowed_channel_ids"] = ["C2"]
    slack.members.fail("conversations.members", rate_limited("conversations.members"))

    channels = client.get("/api/v1/slack/channels").json()["channels"]
    assert [(c["id"], c["allowed"]) for c in channels if c["isPrivate"]] == [("C2", True)]

    updated = SimpleNamespace(provider_metadata={"allowed_channel_ids": ["C2"]})
    with patch.object(MODULE, "update_provider_metadata", new=AsyncMock(return_value=updated)):
        response = client.put("/api/v1/slack/channels", json={"channelIds": ["C2"]})
    assert response.status_code == 200


@pytest.mark.parametrize("method", ["get", "put"])
def test_a_rate_limited_channel_list_answers_429_in_the_allowlist_editor(
    client, slack, monkeypatch, method
):
    from cognee.modules.integrations.ingestion import RateLimitedError
    from cognee.modules.integrations.slack import channels as channels_module

    async def fetch_page(session, access_token, params):
        limited = RateLimitedError("Slack conversations.list is rate limited")
        limited.retry_after = 30
        raise limited

    monkeypatch.setattr(channels_module, "_fetch_page", fetch_page)
    if method == "get":
        response = client.get("/api/v1/slack/channels")
    else:
        response = client.put("/api/v1/slack/channels", json={"channelIds": ["C1"]})

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "30"
