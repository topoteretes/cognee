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
            "list_channels",
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
            MODULE, "list_channels", new=AsyncMock(side_effect=RuntimeError("missing_scope"))
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
        patch.object(
            MODULE, "update_provider_metadata", new=AsyncMock(return_value=updated)
        ) as update_mock,
    ):
        response = client.put("/api/v1/slack/channels", json={"channelIds": ["C1", "C2"]})

    assert response.status_code == 200
    assert response.json()["allowedChannelIds"] == ["C1", "C2"]
    update_mock.assert_awaited_once_with("slack", "T123", {"allowed_channel_ids": ["C1", "C2"]})
