"""Unit tests for cognee.modules.integrations.linear.handle_linear_event.

The credential store, the agent session handler, and the team sync are
mocked — what's under test is the routing: which deliveries revoke, which
open an agent session, which sync the teams they name, and which are dropped (unknown
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

_ACTIVE_CREDENTIAL = SimpleNamespace(
    status="active", provider_account_id="org-1", provider_metadata={"dlt_seeded": True}
)


@pytest.fixture
def mocks(monkeypatch):
    mocked = SimpleNamespace(
        revoke=AsyncMock(return_value=True),
        get_credential=AsyncMock(return_value=_ACTIVE_CREDENTIAL),
        agent_session=AsyncMock(),
        request_sync=AsyncMock(),
    )
    monkeypatch.setattr(handle_module, "revoke_credential_by_account", mocked.revoke)
    monkeypatch.setattr(handle_module, "get_credential_by_account", mocked.get_credential)
    monkeypatch.setattr(handle_module, "handle_agent_session", mocked.agent_session)
    monkeypatch.setattr(handle_module, "request_sync", mocked.request_sync)
    return mocked


def _body(payload: dict) -> bytes:
    return json.dumps(payload).encode()


@pytest.mark.asyncio
async def test_unparseable_body_never_raises(mocks):
    await handle_linear_event(b"not json", {"linear-event": "Issue"})

    mocks.agent_session.assert_not_awaited()
    mocks.request_sync.assert_not_awaited()
    mocks.revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_delivery_without_organization_id_is_dropped(mocks):
    await handle_linear_event(_body({"type": "Issue", "action": "create"}), {})

    mocks.get_credential.assert_not_awaited()
    mocks.request_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_organization_is_dropped(mocks):
    mocks.get_credential.return_value = None

    await handle_linear_event(
        _body({"type": "Issue", "action": "create", "organizationId": "org-999", "data": {}}),
        {},
    )

    mocks.request_sync.assert_not_awaited()
    mocks.agent_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoked_credential_is_dropped(mocks):
    mocks.get_credential.return_value = SimpleNamespace(status="revoked")

    await handle_linear_event(
        _body({"type": "Issue", "action": "create", "organizationId": "org-1", "data": {}}),
        {},
    )

    mocks.request_sync.assert_not_awaited()
    mocks.agent_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_oauth_app_revoked_revokes_the_credential(mocks):
    await handle_linear_event(
        _body({"type": "OAuthApp", "action": "revoked", "organizationId": "org-1"}),
        {},
    )

    mocks.revoke.assert_awaited_once_with("linear", "org-1")
    mocks.agent_session.assert_not_awaited()
    mocks.request_sync.assert_not_awaited()


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
    mocks.request_sync.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "update", "remove"])
@pytest.mark.parametrize(
    ("event_type", "data", "teams"),
    [
        ("Issue", {"teamId": "t1"}, ["t1"]),
        ("Comment", {"issue": {"teamId": "t1"}}, ["t1"]),
        ("Project", {"teamIds": ["t1", "t2"]}, ["t1", "t2"]),
    ],
)
async def test_team_events_sync_the_teams_they_name(mocks, action, event_type, data, teams):
    await handle_linear_event(
        _body({"type": event_type, "action": action, "organizationId": "org-1", "data": data}),
        {},
    )

    mocks.request_sync.assert_awaited_once_with(_ACTIVE_CREDENTIAL, teams)
    mocks.agent_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_team_events_wait_for_the_first_full_sync(mocks):
    mocks.get_credential.return_value = SimpleNamespace(
        status="active", provider_account_id="org-1", provider_metadata={}
    )

    await handle_linear_event(
        _body(
            {
                "type": "Issue",
                "action": "update",
                "organizationId": "org-1",
                "data": {"teamId": "t1"},
            }
        ),
        {},
    )

    mocks.request_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_team_events_only_sync_selected_teams(mocks):
    mocks.get_credential.return_value = SimpleNamespace(
        status="active",
        provider_account_id="org-1",
        provider_metadata={"dlt_seeded": True, "selected_team_ids": ["t2"]},
    )

    await handle_linear_event(
        _body(
            {
                "type": "Project",
                "action": "update",
                "organizationId": "org-1",
                "data": {"teamIds": ["t1", "t2"]},
            }
        ),
        {},
    )
    mocks.request_sync.assert_awaited_once_with(mocks.get_credential.return_value, ["t2"])

    mocks.request_sync.reset_mock()
    await handle_linear_event(
        _body(
            {
                "type": "Issue",
                "action": "update",
                "organizationId": "org-1",
                "data": {"teamId": "t1"},
            }
        ),
        {},
    )
    mocks.request_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_event_types_are_ignored(mocks):
    for payload in (
        {"type": "Issue", "action": "archive", "organizationId": "org-1", "data": {}},
        {"type": "Cycle", "action": "create", "organizationId": "org-1", "data": {"teamId": "t1"}},
        {"type": "AgentSessionEvent", "action": "closed", "organizationId": "org-1"},
    ):
        await handle_linear_event(_body(payload), {})

    mocks.agent_session.assert_not_awaited()
    mocks.request_sync.assert_not_awaited()
    mocks.revoke.assert_not_awaited()
