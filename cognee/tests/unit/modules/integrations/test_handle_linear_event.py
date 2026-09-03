"""Unit tests for cognee.modules.integrations.linear.handle_linear_event.

The credential store, the agent session handler, and the issue sync are
mocked — what's under test is the routing: which deliveries revoke, which
open an agent session, which sync an issue, and which are dropped (unknown
or revoked organization, malformed body, unknown types) without raising,
since the handler runs detached and Linear retries on errors.
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

handle_module = importlib.import_module("cognee.modules.integrations.linear.handle_linear_event")
handle_linear_event = handle_module.handle_linear_event

_ACTIVE_CREDENTIAL = SimpleNamespace(status="active", provider_account_id="org-1")


@pytest.fixture
def mocks(monkeypatch):
    mocked = SimpleNamespace(
        revoke=AsyncMock(return_value=True),
        get_credential=AsyncMock(return_value=_ACTIVE_CREDENTIAL),
        agent_session=AsyncMock(),
        sync_issue=AsyncMock(),
    )
    monkeypatch.setattr(handle_module, "revoke_credential_by_account", mocked.revoke)
    monkeypatch.setattr(handle_module, "get_credential_by_account", mocked.get_credential)
    monkeypatch.setattr(handle_module, "handle_agent_session", mocked.agent_session)
    monkeypatch.setattr(handle_module, "sync_issue", mocked.sync_issue)
    return mocked


def _body(payload: dict) -> bytes:
    return json.dumps(payload).encode()


@pytest.mark.asyncio
async def test_unparseable_body_never_raises(mocks):
    await handle_linear_event(b"not json", {"linear-event": "Issue"})

    mocks.agent_session.assert_not_awaited()
    mocks.sync_issue.assert_not_awaited()
    mocks.revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_delivery_without_organization_id_is_dropped(mocks):
    await handle_linear_event(_body({"type": "Issue", "action": "create"}), {})

    mocks.get_credential.assert_not_awaited()
    mocks.sync_issue.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_organization_is_dropped(mocks):
    mocks.get_credential.return_value = None

    await handle_linear_event(
        _body({"type": "Issue", "action": "create", "organizationId": "org-999", "data": {}}),
        {},
    )

    mocks.sync_issue.assert_not_awaited()
    mocks.agent_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoked_credential_is_dropped(mocks):
    mocks.get_credential.return_value = SimpleNamespace(status="revoked")

    await handle_linear_event(
        _body({"type": "Issue", "action": "create", "organizationId": "org-1", "data": {}}),
        {},
    )

    mocks.sync_issue.assert_not_awaited()
    mocks.agent_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_oauth_app_revoked_revokes_the_credential(mocks):
    await handle_linear_event(
        _body({"type": "OAuthApp", "action": "revoked", "organizationId": "org-1"}),
        {},
    )

    mocks.revoke.assert_awaited_once_with("linear", "org-1")
    mocks.agent_session.assert_not_awaited()
    mocks.sync_issue.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["created", "prompted"])
async def test_agent_session_events_dispatch_to_the_session_handler(mocks, action):
    payload = {
        "type": "AgentSessionEvent",
        "action": action,
        "organizationId": "org-1",
        "agentSession": {"id": "sess-1"},
    }

    await handle_linear_event(_body(payload), {})

    mocks.agent_session.assert_awaited_once_with(_ACTIVE_CREDENTIAL, payload)
    mocks.sync_issue.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "update"])
async def test_issue_events_dispatch_to_sync_issue(mocks, action):
    issue = {"identifier": "COG-1", "title": "Fix login"}

    await handle_linear_event(
        _body({"type": "Issue", "action": action, "organizationId": "org-1", "data": issue}),
        {},
    )

    mocks.sync_issue.assert_awaited_once_with(_ACTIVE_CREDENTIAL, issue)
    mocks.agent_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_event_types_are_ignored(mocks):
    # Issue deletions included: deleting indexed data on a webhook would be
    # a silent destructive surprise — forget() stays a human decision.
    for payload in (
        {"type": "Issue", "action": "remove", "organizationId": "org-1", "data": {}},
        {"type": "Comment", "action": "create", "organizationId": "org-1", "data": {}},
        {"type": "AgentSessionEvent", "action": "closed", "organizationId": "org-1"},
    ):
        await handle_linear_event(_body(payload), {})

    mocks.agent_session.assert_not_awaited()
    mocks.sync_issue.assert_not_awaited()
    mocks.revoke.assert_not_awaited()
