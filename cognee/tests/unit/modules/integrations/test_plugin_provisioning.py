"""Unit tests for the /plugins/{plugin_key} routes in get_integrations_router.

Agent plugins (claude-code, codex, ...) are not OAuth providers — each
connected plugin gets its own agent sub-user + labeled API key. These tests
exercise the router's provisioning logic (idempotent get-or-create, key
rotation on re-provision, unknown-plugin 404, disconnect-revokes-keys) with
the DB-facing collaborators mocked at module level, the same way
test_get_integrations_router.py mocks the OAuth credential layer. The
throttled UserApiKey.last_used_at write that provisioned keys rely on for
"last seen" is covered here too, via UserManager directly.
"""

import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_users.exceptions import UserAlreadyExists

from cognee.api.v1.integrations.routers.get_integrations_router import get_integrations_router
from cognee.modules.agents.registry import AGENT_CONFIG_NAME, build_agent_connection_id
from cognee.modules.users.get_user_manager import UserManager
from cognee.modules.users.methods import get_authenticated_user

USER_ID = uuid4()
AGENT_ID = uuid4()

# Import the submodule explicitly: the router package's __init__.py rebinds
# get_integrations_router to the *function*, shadowing the submodule (see the
# fuller explanation in test_get_integrations_router.py).
_router_module = importlib.import_module(
    "cognee.api.v1.integrations.routers.get_integrations_router"
)

USER_MANAGER_MODULE = "cognee.modules.users.get_user_manager"


class _FakeUser:
    id = USER_ID


def _fake_agent_user(plugin_key: str = "claude-code"):
    return SimpleNamespace(
        id=AGENT_ID,
        email=f"{plugin_key}+{USER_ID}@cognee.agent",
        tenant_id=None,
    )


def _fake_agent_info(plugin_key: str = "claude-code"):
    return SimpleNamespace(user=_fake_agent_user(plugin_key), api_key_label="abcd1234****")


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: _FakeUser()
    yield TestClient(app)


def test_unknown_plugin_404s_on_every_route(client):
    assert client.post("/api/v1/integrations/plugins/notreal/provision").status_code == 404
    assert client.delete("/api/v1/integrations/plugins/notreal").status_code == 404


def test_provision_creates_agent_key_and_plugin_record(client):
    agent_user = _fake_agent_user()
    with (
        patch.object(
            _router_module,
            "create_agent",
            new=AsyncMock(return_value=(agent_user, "raw-key-shown-once")),
        ) as create_agent,
        patch.object(
            _router_module, "get_principal_all_configuration", new=AsyncMock(return_value=[])
        ),
        patch.object(
            _router_module, "store_principal_configuration", new=AsyncMock()
        ) as store_config,
        patch.object(
            _router_module, "register_agent_connection", new=AsyncMock()
        ) as register_connection,
    ):
        response = client.post("/api/v1/integrations/plugins/claude-code/provision")

    assert response.status_code == 200
    body = response.json()
    assert body["pluginKey"] == "claude-code"
    assert body["agentId"] == str(AGENT_ID)
    assert body["apiKey"] == "raw-key-shown-once"
    assert body["created"] is True

    create_agent.assert_awaited_once()
    assert create_agent.await_args.args[0] == "claude-code"

    # plugin_key recorded in the agent's principal configuration blob.
    store_config.assert_awaited_once()
    store_kwargs = store_config.await_args.kwargs
    assert store_kwargs["principal_id"] == AGENT_ID
    assert store_kwargs["name"] == AGENT_CONFIG_NAME
    plugin_entry = store_kwargs["configuration"]["plugin"]
    assert plugin_entry["key"] == "claude-code"
    assert plugin_entry["provisioned_at"]

    # Registered so /agents/connections sees the plugin before first traffic.
    register_connection.assert_awaited_once()
    register_kwargs = register_connection.await_args.kwargs
    assert register_kwargs["agent_session_name"] == "plugin:claude-code"
    assert register_kwargs["connection_type"] == "claude_code"
    assert register_kwargs["user_id"] == AGENT_ID
    assert register_kwargs["metadata"] == {"plugin_key": "claude-code"}


def test_second_provision_returns_same_agent_with_rotated_key(client):
    old_key = SimpleNamespace(id=uuid4())
    with (
        patch.object(
            _router_module,
            "create_agent",
            new=AsyncMock(side_effect=UserAlreadyExists()),
        ),
        patch.object(
            _router_module,
            "list_agents",
            new=AsyncMock(return_value=[_fake_agent_info()]),
        ),
        patch.object(_router_module, "get_api_keys", new=AsyncMock(return_value=[old_key])),
        patch.object(_router_module, "delete_api_key", new=AsyncMock()) as delete_key,
        patch.object(
            _router_module,
            "create_api_key",
            new=AsyncMock(return_value=SimpleNamespace(api_key="rotated-key")),
        ) as create_key,
        patch.object(
            _router_module, "get_principal_all_configuration", new=AsyncMock(return_value=[])
        ),
        patch.object(_router_module, "store_principal_configuration", new=AsyncMock()),
        patch.object(_router_module, "register_agent_connection", new=AsyncMock()),
    ):
        response = client.post("/api/v1/integrations/plugins/claude-code/provision")

    body = response.json()
    assert body["created"] is False
    assert body["agentId"] == str(AGENT_ID)
    assert body["apiKey"] == "rotated-key"

    # Old key revoked, new labeled key minted — rotation is the re-provision.
    delete_key.assert_awaited_once()
    assert delete_key.await_args.args[1] == old_key.id
    create_key.assert_awaited_once()
    assert create_key.await_args.args[1] == "claude-code"


def test_reprovision_preserves_original_provisioned_at(client):
    existing_config = [
        {
            "name": AGENT_CONFIG_NAME,
            "configuration": {
                "plugin": {"key": "claude-code", "provisioned_at": "2026-01-01T00:00:00+00:00"},
                "agents": {"keep": "me"},
            },
        }
    ]
    with (
        patch.object(
            _router_module,
            "create_agent",
            new=AsyncMock(return_value=(_fake_agent_user(), "key")),
        ),
        patch.object(
            _router_module,
            "get_principal_all_configuration",
            new=AsyncMock(return_value=existing_config),
        ),
        patch.object(
            _router_module, "store_principal_configuration", new=AsyncMock()
        ) as store_config,
        patch.object(_router_module, "register_agent_connection", new=AsyncMock()),
    ):
        response = client.post("/api/v1/integrations/plugins/claude-code/provision")

    assert response.status_code == 200
    configuration = store_config.await_args.kwargs["configuration"]
    # Rotation isn't a new install: the first provisioned_at sticks, and the
    # registry's own "agents" entries in the shared blob survive the write.
    assert configuration["plugin"]["provisioned_at"] == "2026-01-01T00:00:00+00:00"
    assert configuration["agents"] == {"keep": "me"}


def test_disconnect_revokes_keys_and_deactivates_connection(client):
    keys = [SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())]
    with (
        patch.object(
            _router_module,
            "list_agents",
            new=AsyncMock(return_value=[_fake_agent_info()]),
        ),
        patch.object(_router_module, "get_api_keys", new=AsyncMock(return_value=keys)),
        patch.object(_router_module, "delete_api_key", new=AsyncMock()) as delete_key,
        patch.object(_router_module, "deactivate_agent_connection", new=AsyncMock()) as deactivate,
    ):
        response = client.delete("/api/v1/integrations/plugins/claude-code")

    assert response.json() == {"disconnected": True}
    assert delete_key.await_count == 2
    expected_connection_id = build_agent_connection_id(
        agent_session_name="plugin:claude-code", user_id=str(AGENT_ID)
    )
    deactivate.assert_awaited_once_with(AGENT_ID, expected_connection_id)


def test_disconnect_without_provisioned_agent_reports_false(client):
    with patch.object(_router_module, "list_agents", new=AsyncMock(return_value=[])):
        response = client.delete("/api/v1/integrations/plugins/claude-code")
    assert response.json() == {"disconnected": False}


# --------------------------------------------------------------------------- #
# UserApiKey.last_used_at throttle (UserManager.get_by_token)
# --------------------------------------------------------------------------- #


class _FakeSession:
    """Async-context-manager session whose execute() returns queued results."""

    def __init__(self, results):
        self._results = list(results)
        self.execute = AsyncMock(side_effect=self._results)
        self.commit = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _result(scalar_value):
    res = MagicMock()
    res.scalar.return_value = scalar_value
    return res


@pytest.mark.asyncio
async def test_last_used_at_throttle_two_auths_in_window_one_write():
    user_id = uuid4()
    user = SimpleNamespace(id=user_id, is_active=True)

    # First auth: last_used_at is unset → gets written (one commit).
    key = SimpleNamespace(id=uuid4(), user_id=user_id, last_used_at=None)
    # Second auth: the key comes back with the fresh timestamp the first auth
    # persisted → inside the throttle window, no second write.
    fresh_key = SimpleNamespace(id=key.id, user_id=user_id, last_used_at=datetime.now(timezone.utc))

    first_session = _FakeSession([_result(key), _result(user)])
    second_session = _FakeSession([_result(fresh_key), _result(user)])
    engine = MagicMock()
    engine.get_async_session = MagicMock(side_effect=[first_session, second_session])
    manager = UserManager(MagicMock())

    with patch(f"{USER_MANAGER_MODULE}.get_relational_engine", return_value=engine):
        assert await manager.get_by_token("raw-key") is user
        assert await manager.get_by_token("raw-key") is user

    assert key.last_used_at is not None
    first_session.commit.assert_awaited_once()
    second_session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_last_used_at_write_failure_never_breaks_auth():
    user_id = uuid4()
    user = SimpleNamespace(id=user_id, is_active=True)
    key = SimpleNamespace(id=uuid4(), user_id=user_id, last_used_at=None)

    session = _FakeSession([_result(key), _result(user)])
    session.commit = AsyncMock(side_effect=RuntimeError("db went away"))
    engine = MagicMock()
    engine.get_async_session = MagicMock(return_value=session)
    manager = UserManager(MagicMock())

    with patch(f"{USER_MANAGER_MODULE}.get_relational_engine", return_value=engine):
        assert await manager.get_by_token("raw-key") is user
