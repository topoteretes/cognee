"""Unit tests for SlackIntegration.revoke_remote / .refresh.

Network calls are mocked via aioresponses-free patching of aiohttp — these
tests check the *contract* each hook must honor: revoke_remote never raises
(disconnect must always succeed locally even if Slack is unreachable or the
token is already dead), and refresh is a true no-op when there's nothing to
refresh, since most Slack apps never enable token rotation.
"""

from datetime import datetime, timedelta, timezone
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
    with (
        patch("cognee.modules.integrations.slack.adapter.decrypt_token_payload", return_value={}),
        patch("aiohttp.ClientSession") as session_cls,
    ):
        # No aiohttp session should even be opened.
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
    with (
        patch(
            "cognee.modules.integrations.slack.adapter.decrypt_token_payload",
            return_value={"access_token": "xoxb-secret"},
        ),  # no refresh_token
        patch("aiohttp.ClientSession") as session_cls,
    ):
        await integration.refresh(credential)
        session_cls.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_persists_rotated_tokens():
    credential = _fake_credential(id=uuid4(), ciphertext=b"old")
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    db = AsyncMock()
    db.__aenter__.return_value = db
    db.execute.return_value = MagicMock(rowcount=1)
    engine = MagicMock()
    engine.get_async_session.return_value = db
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
            "cognee.modules.integrations.slack.adapter.get_relational_engine", return_value=engine
        ),
        patch(
            "cognee.modules.integrations.slack.adapter.encrypt_credentials",
            return_value=(b"new", b"nonce", 1, "1"),
        ) as encrypt,
        patch("cognee.modules.integrations.slack.adapter.require", return_value="x"),
    ):
        await integration.refresh(credential)

    encrypt.assert_called_once_with({"access_token": "xoxb-new", "refresh_token": "xoxe-new"})
    db.execute.assert_awaited_once()
    db.commit.assert_awaited_once()
    values = {
        column.name: value.value for column, value in db.execute.call_args.args[0]._values.items()
    }
    assert values["ciphertext"] == b"new"
    assert "provider_metadata" not in values
    # expires_in: 3600 in the mocked response should become an expiry ~1 hour
    # out; allow a few seconds of slack for the two `datetime.now(timezone.utc)` calls
    # (this test's and the code under test's) not landing in the same instant.
    assert abs((values["token_expires_at"] - expires_at).total_seconds()) < 5


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
        pytest.raises(RuntimeError, match="invalid_grant"),
    ):
        await integration.refresh(credential)
