"""Unit tests for SlackIntegration.revoke_remote / .refresh.

Network calls are mocked via aioresponses-free patching of aiohttp — these
tests check the *contract* each hook must honor: revoke_remote never raises
(disconnect must always succeed locally even if Slack is unreachable or the
token is already dead), and refresh is a true no-op when there's nothing to
refresh, since most Slack apps never enable token rotation.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.integrations.slack.adapter import SlackIntegration

integration = SlackIntegration()


def _fake_credential(**overrides):
    credential = MagicMock()
    credential.provider_account_id = "T123"
    credential.user_id = uuid4()
    credential.account_label = "Acme"
    credential.scopes = "commands,chat:write"
    for key, value in overrides.items():
        setattr(credential, key, value)
    return credential


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload


def _fake_session(payload):
    session = MagicMock()
    session.post = MagicMock(return_value=_FakeResponse(payload))
    return session


class _FakeSessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_revoke_remote_does_nothing_without_an_access_token():
    credential = _fake_credential()
    with patch(
        "cognee.modules.integrations.slack.adapter.decrypt_token_payload",
        return_value={},
    ):
        # No aiohttp session should even be opened.
        with patch("aiohttp.ClientSession") as session_cls:
            await integration.revoke_remote(credential)
            session_cls.assert_not_called()


@pytest.mark.asyncio
async def test_revoke_remote_calls_auth_revoke_with_bearer_token():
    credential = _fake_credential()
    session = _fake_session({"ok": True})
    with (
        patch(
            "cognee.modules.integrations.slack.adapter.decrypt_token_payload",
            return_value={"access_token": "xoxb-secret"},
        ),
        patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)),
    ):
        await integration.revoke_remote(credential)

    session.post.assert_called_once()
    _, kwargs = session.post.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer xoxb-secret"}


@pytest.mark.asyncio
async def test_revoke_remote_never_raises_on_network_failure():
    credential = _fake_credential()
    with (
        patch(
            "cognee.modules.integrations.slack.adapter.decrypt_token_payload",
            return_value={"access_token": "xoxb-secret"},
        ),
        patch("aiohttp.ClientSession", side_effect=RuntimeError("network is down")),
    ):
        await integration.revoke_remote(credential)  # must not raise


@pytest.mark.asyncio
async def test_revoke_remote_never_raises_on_slack_error_response():
    credential = _fake_credential()
    session = _fake_session({"ok": False, "error": "invalid_auth"})
    with (
        patch(
            "cognee.modules.integrations.slack.adapter.decrypt_token_payload",
            return_value={"access_token": "xoxb-secret"},
        ),
        patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)),
    ):
        await integration.revoke_remote(credential)  # must not raise


@pytest.mark.asyncio
async def test_refresh_is_a_noop_without_a_refresh_token():
    credential = _fake_credential()
    with patch(
        "cognee.modules.integrations.slack.adapter.decrypt_token_payload",
        return_value={"access_token": "xoxb-secret"},  # no refresh_token
    ):
        with patch("aiohttp.ClientSession") as session_cls:
            await integration.refresh(credential)
            session_cls.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_persists_rotated_tokens():
    credential = _fake_credential()
    expires_at = datetime.now(UTC) + timedelta(hours=1)
    session = _fake_session(
        {
            "ok": True,
            "team": {"id": "T123", "name": "Acme"},
            "access_token": "xoxb-new",
            "refresh_token": "xoxe-new",
            "expires_in": 3600,
        }
    )
    with (
        patch(
            "cognee.modules.integrations.slack.adapter.decrypt_token_payload",
            return_value={"access_token": "xoxb-old", "refresh_token": "xoxe-old"},
        ),
        patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)),
        patch(
            "cognee.modules.integrations.slack.adapter.upsert_credential", new=AsyncMock()
        ) as upsert,
        patch("cognee.modules.integrations.slack.adapter.require", return_value="x"),
    ):
        await integration.refresh(credential)

    upsert.assert_awaited_once()
    _, kwargs = upsert.call_args
    assert kwargs["token_payload"] == {"access_token": "xoxb-new", "refresh_token": "xoxe-new"}
    assert kwargs["provider_account_id"] == "T123"


@pytest.mark.asyncio
async def test_refresh_raises_on_rejected_refresh():
    credential = _fake_credential()
    session = _fake_session({"ok": False, "error": "invalid_grant"})
    with (
        patch(
            "cognee.modules.integrations.slack.adapter.decrypt_token_payload",
            return_value={"access_token": "xoxb-old", "refresh_token": "xoxe-old"},
        ),
        patch("aiohttp.ClientSession", return_value=_FakeSessionContext(session)),
        patch("cognee.modules.integrations.slack.adapter.require", return_value="x"),
    ):
        with pytest.raises(RuntimeError, match="invalid_grant"):
            await integration.refresh(credential)
