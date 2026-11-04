"""Unit tests for the /plugins/{plugin_key} routes in get_integrations_router.

Agent plugins (claude-code, codex, ...) are not OAuth providers — each
connected plugin gets its own agent sub-user, with one API key per machine. These tests
exercise the router's provisioning logic (idempotent get-or-create that
resolves an existing agent — tagged or pre-tag plain email — before creating
one, one added key per provision, any well-formed key provisions and a
malformed one is a 422 (SDK-646), disconnect-revokes-keys) with
the DB-facing collaborators mocked at module level, the same way
test_get_integrations_router.py mocks the OAuth credential layer. The
throttled UserApiKey.last_used_at write that provisioned keys rely on for
"last seen" is covered here too, via UserManager directly.
"""

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_users.exceptions import UserAlreadyExists
from sqlalchemy.exc import IntegrityError

from cognee.api.v1.integrations.routers.get_integrations_router import get_integrations_router
from cognee.modules.agents.registry import AGENT_CONFIG_NAME, build_agent_connection_id
from cognee.modules.integrations.agent_identity import agent_email, plugin_agent_email
from cognee.modules.users.get_user_manager import UserManager
from cognee.modules.users.methods import get_authenticated_user

USER_ID = uuid4()
AGENT_ID = uuid4()
KEY_ID = uuid4()

# The methods package rebinds get_authenticated_user to the function; import
# the submodule explicitly to reach the resolved auth-posture constant.
_auth_module = importlib.import_module("cognee.modules.users.methods.get_authenticated_user")

# Import the submodule explicitly: the router package's __init__.py rebinds
# get_integrations_router to the *function*, shadowing the submodule (see the
# fuller explanation in test_get_integrations_router.py).
_router_module = importlib.import_module(
    "cognee.api.v1.integrations.routers.get_integrations_router"
)

USER_MANAGER_MODULE = "cognee.modules.users.get_user_manager"


class _FakeUser:
    id = USER_ID
    tenant_id = None


def _fake_agent_user(plugin_key: str = "claude-code", agent_id=AGENT_ID, *, tagged=True):
    """A plugin agent; ``tagged=False`` is the plain email form of agents
    provisioned before the provision route tagged its emails."""
    return SimpleNamespace(
        id=agent_id,
        email=(
            plugin_agent_email(plugin_key, USER_ID) if tagged else agent_email(plugin_key, USER_ID)
        ),
        tenant_id=None,
    )


def _fake_agent_info(plugin_key: str = "claude-code", agent_id=AGENT_ID, *, tagged=True):
    return SimpleNamespace(
        user=_fake_agent_user(plugin_key, agent_id, tagged=tagged), api_key_label="abcd1234****"
    )


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: _FakeUser()
    yield TestClient(app)


@pytest.fixture(autouse=True)
def no_child_agents():
    """Provisioning resolves the user's child agents before creating one; by
    default there are none. Tests with existing agents patch ``list_agents``
    again inside their own block, which takes precedence while it is active."""
    with patch.object(_router_module, "list_agents", new=AsyncMock(return_value=[])):
        yield


@pytest.mark.parametrize(
    "plugin_key",
    ["Not%20A%20Plugin", "Hermes", "a+b", "-leading", "trailing-", "a--b", "x" * 65],
)
def test_malformed_plugin_key_422s_on_every_route(client, plugin_key):
    """The server keeps no plugin list (SDK-646): the only check is the key grammar."""
    base = f"/api/v1/integrations/plugins/{plugin_key}"
    assert client.post(f"{base}/provision").status_code == 422
    assert client.get(f"{base}/keys").status_code == 422
    assert client.delete(f"{base}/keys/{uuid4()}").status_code == 422
    assert client.delete(base).status_code == 422


def test_any_well_formed_key_provisions_with_its_self_description(client):
    """A plugin the server has never heard of names and describes itself."""
    agent_user = _fake_agent_user("hermes-agent")
    with (
        patch.object(
            _router_module, "create_agent", new=AsyncMock(return_value=(agent_user, None))
        ) as create_agent,
        patch.object(
            _router_module,
            "add_plugin_agent_key",
            new=AsyncMock(return_value=SimpleNamespace(id=KEY_ID, api_key="raw-key")),
        ) as add_key,
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
        response = client.post(
            "/api/v1/integrations/plugins/hermes-agent/provision",
            json={"label": "  Hermes Agent ", "connection_type": "hermes_agent"},
        )

    assert response.status_code == 200
    assert response.json()["pluginKey"] == "hermes-agent"
    # The agent is minted as a plugin identity (tagged email), named after the key.
    assert create_agent.await_args.args[0] == "hermes-agent"
    assert create_agent.await_args.kwargs == {"issue_api_key": False, "plugin_identity": True}
    add_key.assert_awaited_once_with(AGENT_ID, "hermes-agent")
    # The self-description is what the status page will show, label stripped.
    plugin_entry = store_config.await_args.kwargs["configuration"]["plugin"]
    assert plugin_entry["key"] == "hermes-agent"
    assert plugin_entry["label"] == "Hermes Agent"
    assert plugin_entry["connection_type"] == "hermes_agent"
    # The connection carries the declared type and names the plugin.
    register_kwargs = register_connection.await_args.kwargs
    assert register_kwargs["agent_session_name"] == "plugin:hermes-agent"
    assert register_kwargs["connection_type"] == "hermes_agent"
    assert register_kwargs["metadata"] == {"plugin_key": "hermes-agent"}


@pytest.mark.parametrize(
    "body",
    [
        {"label": "x" * 65},
        {"connection_type": "Hermes Agent"},
        {"connection_type": "hermes-agent"},
    ],
)
def test_provision_rejects_a_malformed_self_description(client, body):
    with patch.object(_router_module, "create_agent", new=AsyncMock()) as create_agent:
        response = client.post("/api/v1/integrations/plugins/hermes-agent/provision", json=body)

    assert response.status_code == 422
    create_agent.assert_not_awaited()


@pytest.mark.skipif(
    not _auth_module.REQUIRE_AUTHENTICATION,
    reason="auth disabled in this environment (single-user posture)",
)
def test_provision_rejects_unauthenticated_callers():
    # No dependency override: the real auth stack must gate the route that
    # mints raw API keys.
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    response = TestClient(app).post("/api/v1/integrations/plugins/claude-code/provision")
    assert response.status_code == 401


def test_provision_creates_agent_key_and_plugin_record(client):
    agent_user = _fake_agent_user()
    with (
        patch.object(
            _router_module,
            "create_agent",
            new=AsyncMock(return_value=(agent_user, None)),
        ) as create_agent,
        patch.object(
            _router_module,
            "add_plugin_agent_key",
            new=AsyncMock(return_value=SimpleNamespace(id=KEY_ID, api_key="raw-key-shown-once")),
        ) as add_key,
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
    assert body["keyId"] == str(KEY_ID)
    assert body["created"] is True

    # Provisioned under the plugin key AND the authenticated principal — the
    # agent must be a child of the requesting user, not of some other user.
    create_agent.assert_awaited_once()
    assert create_agent.await_args.args[0] == "claude-code"
    assert create_agent.await_args.args[1].id == USER_ID
    # Every key goes through add_plugin_agent_key, on the agent, named after the plugin.
    assert create_agent.await_args.kwargs == {"issue_api_key": False, "plugin_identity": True}
    add_key.assert_awaited_once_with(AGENT_ID, "claude-code")

    # plugin_key recorded in the agent's principal configuration blob. No body
    # (a plugin from before the self-description): no label, and the
    # connection type is derived from the key.
    store_config.assert_awaited_once()
    store_kwargs = store_config.await_args.kwargs
    assert store_kwargs["principal_id"] == AGENT_ID
    assert store_kwargs["name"] == AGENT_CONFIG_NAME
    plugin_entry = store_kwargs["configuration"]["plugin"]
    assert plugin_entry["key"] == "claude-code"
    assert plugin_entry["provisioned_at"]
    assert plugin_entry["label"] is None
    assert plugin_entry["connection_type"] == "claude_code"

    # Registered so /agents/connections sees the plugin before first traffic.
    register_connection.assert_awaited_once()
    register_kwargs = register_connection.await_args.kwargs
    assert register_kwargs["agent_session_name"] == "plugin:claude-code"
    assert register_kwargs["connection_type"] == "claude_code"
    assert register_kwargs["user_id"] == AGENT_ID
    assert register_kwargs["metadata"] == {"plugin_key": "claude-code"}


def test_provision_registers_parent_tenant_id():
    """The registered connection carries the parent's tenant, not the stale agent object's.

    create_agent sets the agent's tenant_id via a raw UPDATE without
    refreshing the returned ORM instance, so on first provision the agent
    object still reads tenant_id=None. The router must therefore source the
    tenant from the authenticated parent (they share it by construction).
    """
    tenant_id = uuid4()
    tenanted_user = SimpleNamespace(id=USER_ID, tenant_id=tenant_id)
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: tenanted_user
    tenant_client = TestClient(app)

    with (
        patch.object(
            _router_module,
            "create_agent",
            # Stale first-provision shape: agent object still has tenant_id=None.
            new=AsyncMock(return_value=(_fake_agent_user(), None)),
        ),
        patch.object(
            _router_module,
            "add_plugin_agent_key",
            new=AsyncMock(return_value=SimpleNamespace(id=KEY_ID, api_key="key")),
        ),
        patch.object(
            _router_module, "get_principal_all_configuration", new=AsyncMock(return_value=[])
        ),
        patch.object(_router_module, "store_principal_configuration", new=AsyncMock()),
        patch.object(
            _router_module, "register_agent_connection", new=AsyncMock()
        ) as register_connection,
    ):
        response = tenant_client.post("/api/v1/integrations/plugins/claude-code/provision")

    assert response.status_code == 200
    assert register_connection.await_args.kwargs["tenant_id"] == tenant_id


def test_second_provision_adds_a_key_to_the_same_agent_and_revokes_nothing(client):
    # list_agents returns another plugin's agent first: the router must pick
    # the claude-code agent by its deterministic email, not the first child.
    codex_agent_id = uuid4()
    with (
        patch.object(_router_module, "create_agent", new=AsyncMock()) as create_agent,
        patch.object(
            _router_module,
            "list_agents",
            new=AsyncMock(
                return_value=[_fake_agent_info("codex", codex_agent_id), _fake_agent_info()]
            ),
        ),
        patch.object(_router_module, "delete_api_key", new=AsyncMock()) as delete_key,
        patch.object(_router_module, "revoke_plugin_agent_keys", new=AsyncMock()) as revoke_keys,
        patch.object(
            _router_module,
            "add_plugin_agent_key",
            new=AsyncMock(return_value=SimpleNamespace(id=KEY_ID, api_key="added-key")),
        ) as add_key,
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
    assert body["apiKey"] == "added-key"

    # One key added to the claude-code agent (never the codex one), none revoked,
    # and no second agent created.
    add_key.assert_awaited_once_with(AGENT_ID, "claude-code")
    create_agent.assert_not_awaited()
    delete_key.assert_not_awaited()
    revoke_keys.assert_not_awaited()


def test_reprovision_finds_an_agent_provisioned_before_the_email_tag(client):
    """An agent minted before the provision route tagged its emails carries the
    plain form, which never collides with the tagged email a create would
    mint. The route must resolve it before creating, or an upgrade leaves the
    user with two agents for one plugin: keys, sessions and grants split."""
    with (
        patch.object(_router_module, "create_agent", new=AsyncMock()) as create_agent,
        patch.object(
            _router_module,
            "list_agents",
            new=AsyncMock(return_value=[_fake_agent_info("claude-code", tagged=False)]),
        ),
        patch.object(
            _router_module,
            "add_plugin_agent_key",
            new=AsyncMock(return_value=SimpleNamespace(id=KEY_ID, api_key="added-key")),
        ) as add_key,
        patch.object(
            _router_module, "get_principal_all_configuration", new=AsyncMock(return_value=[])
        ),
        patch.object(_router_module, "store_principal_configuration", new=AsyncMock()),
        patch.object(_router_module, "register_agent_connection", new=AsyncMock()),
    ):
        response = client.post("/api/v1/integrations/plugins/claude-code/provision")

    assert response.status_code == 200
    assert response.json()["created"] is False
    assert response.json()["agentId"] == str(AGENT_ID)
    add_key.assert_awaited_once_with(AGENT_ID, "claude-code")
    create_agent.assert_not_awaited()


def test_reprovision_prefers_the_tagged_agent_when_both_forms_exist(client):
    """Both a plain-email and a tagged agent under one built-in key: the
    tagged one is the one this route minted, so it wins — never whichever
    child happens to be listed first."""
    legacy_id = uuid4()
    with (
        patch.object(_router_module, "create_agent", new=AsyncMock()) as create_agent,
        patch.object(
            _router_module,
            "list_agents",
            new=AsyncMock(
                return_value=[
                    _fake_agent_info("claude-code", legacy_id, tagged=False),
                    _fake_agent_info("claude-code", AGENT_ID),
                ]
            ),
        ),
        patch.object(
            _router_module,
            "add_plugin_agent_key",
            new=AsyncMock(return_value=SimpleNamespace(id=KEY_ID, api_key="added-key")),
        ) as add_key,
        patch.object(
            _router_module, "get_principal_all_configuration", new=AsyncMock(return_value=[])
        ),
        patch.object(_router_module, "store_principal_configuration", new=AsyncMock()),
        patch.object(_router_module, "register_agent_connection", new=AsyncMock()),
        patch.object(
            _router_module, "revoke_plugin_agent_keys", new=AsyncMock(return_value=1)
        ) as revoke_keys,
        patch.object(_router_module, "deactivate_agent_connection", new=AsyncMock()),
    ):
        provision = client.post("/api/v1/integrations/plugins/claude-code/provision")
        disconnect = client.delete("/api/v1/integrations/plugins/claude-code")

    assert provision.json()["agentId"] == str(AGENT_ID)
    add_key.assert_awaited_once_with(AGENT_ID, "claude-code")
    create_agent.assert_not_awaited()
    assert disconnect.json() == {"disconnected": True}
    revoke_keys.assert_awaited_once_with(AGENT_ID)


def test_custom_agent_named_after_a_plugin_is_never_the_plugin(client):
    """A custom agent the user created as ``hermes-agent`` carries the plain
    email form. Only built-in keys could provision before the tag, so under
    any other key the plain form is a custom agent: provisioning creates the
    plugin's own agent, and disconnecting the plugin before it exists revokes
    nothing — least of all the custom agent's keys."""
    custom_agent_id = uuid4()
    plugin_agent = _fake_agent_user("hermes-agent")
    with (
        patch.object(
            _router_module, "create_agent", new=AsyncMock(return_value=(plugin_agent, None))
        ) as create_agent,
        patch.object(
            _router_module,
            "list_agents",
            new=AsyncMock(
                return_value=[_fake_agent_info("hermes-agent", custom_agent_id, tagged=False)]
            ),
        ),
        patch.object(
            _router_module,
            "add_plugin_agent_key",
            new=AsyncMock(return_value=SimpleNamespace(id=KEY_ID, api_key="raw-key")),
        ) as add_key,
        patch.object(
            _router_module, "get_principal_all_configuration", new=AsyncMock(return_value=[])
        ),
        patch.object(_router_module, "store_principal_configuration", new=AsyncMock()),
        patch.object(_router_module, "register_agent_connection", new=AsyncMock()),
        patch.object(_router_module, "revoke_plugin_agent_keys", new=AsyncMock()) as revoke_keys,
        patch.object(_router_module, "deactivate_agent_connection", new=AsyncMock()),
    ):
        disconnect = client.delete("/api/v1/integrations/plugins/hermes-agent")
        provision = client.post("/api/v1/integrations/plugins/hermes-agent/provision")

    assert disconnect.json() == {"disconnected": False}
    revoke_keys.assert_not_awaited()
    assert provision.status_code == 200
    assert provision.json()["created"] is True
    assert provision.json()["agentId"] == str(AGENT_ID)
    create_agent.assert_awaited_once()
    add_key.assert_awaited_once_with(AGENT_ID, "hermes-agent")


def test_provision_race_integrity_error_takes_the_add_path(client):
    """Two concurrent first provisions: the loser's insert hits the unique-email
    constraint (IntegrityError, not UserAlreadyExists) and must recover by
    adding a key to the existing agent instead of 500ing."""
    with (
        patch.object(
            _router_module,
            "create_agent",
            new=AsyncMock(side_effect=IntegrityError("INSERT INTO users", {}, Exception("dup"))),
        ),
        patch.object(
            _router_module,
            "list_agents",
            # Nothing to resolve before the create; the winner's agent after it.
            new=AsyncMock(side_effect=[[], [_fake_agent_info()]]),
        ),
        patch.object(
            _router_module,
            "add_plugin_agent_key",
            new=AsyncMock(return_value=SimpleNamespace(id=KEY_ID, api_key="added-key")),
        ) as add_key,
        patch.object(
            _router_module, "get_principal_all_configuration", new=AsyncMock(return_value=[])
        ),
        patch.object(_router_module, "store_principal_configuration", new=AsyncMock()),
        patch.object(_router_module, "register_agent_connection", new=AsyncMock()),
    ):
        response = client.post("/api/v1/integrations/plugins/claude-code/provision")

    assert response.status_code == 200
    body = response.json()
    assert body["created"] is False
    assert body["apiKey"] == "added-key"
    add_key.assert_awaited_once_with(AGENT_ID, "claude-code")


def test_reprovision_409s_when_agent_not_resolvable(client):
    """UserAlreadyExists but the agent isn't among this user's children: the
    router must refuse (409) rather than mint a key for another
    plugin's — or another owner's — agent."""
    with (
        patch.object(
            _router_module,
            "create_agent",
            new=AsyncMock(side_effect=UserAlreadyExists()),
        ),
        patch.object(
            _router_module,
            "list_agents",
            # Only a different plugin's agent exists under this user.
            new=AsyncMock(return_value=[_fake_agent_info("codex", uuid4())]),
        ),
        patch.object(_router_module, "add_plugin_agent_key", new=AsyncMock()) as add_key,
        patch.object(_router_module, "register_agent_connection", new=AsyncMock()),
    ):
        response = client.post("/api/v1/integrations/plugins/claude-code/provision")

    assert response.status_code == 409
    add_key.assert_not_awaited()


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
            new=AsyncMock(return_value=(_fake_agent_user(), None)),
        ),
        patch.object(
            _router_module,
            "add_plugin_agent_key",
            new=AsyncMock(return_value=SimpleNamespace(id=KEY_ID, api_key="key")),
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
    # Another key isn't a new install: the first provisioned_at sticks, and the
    # registry's own "agents" entries in the shared blob survive the write.
    assert configuration["plugin"]["provisioned_at"] == "2026-01-01T00:00:00+00:00"
    assert configuration["agents"] == {"keep": "me"}


def test_disconnect_revokes_keys_and_deactivates_connection(client):
    with (
        patch.object(
            _router_module,
            "list_agents",
            new=AsyncMock(return_value=[_fake_agent_info()]),
        ),
        patch.object(
            _router_module, "revoke_plugin_agent_keys", new=AsyncMock(return_value=2)
        ) as revoke_keys,
        patch.object(_router_module, "deactivate_agent_connection", new=AsyncMock()) as deactivate,
    ):
        response = client.delete("/api/v1/integrations/plugins/claude-code")

    assert response.json() == {"disconnected": True}
    revoke_keys.assert_awaited_once_with(AGENT_ID)
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
async def test_last_used_at_refreshes_once_stale():
    """The throttle is a throttle, not a one-shot: a timestamp older than the
    window must be rewritten, or "last seen" freezes after the first auth."""
    user_id = uuid4()
    user = SimpleNamespace(id=user_id, is_active=True)
    stale = datetime.now(timezone.utc) - timedelta(minutes=10)
    key = SimpleNamespace(id=uuid4(), user_id=user_id, last_used_at=stale)

    session = _FakeSession([_result(key), _result(user)])
    engine = MagicMock()
    engine.get_async_session = MagicMock(return_value=session)
    manager = UserManager(MagicMock())

    with patch(f"{USER_MANAGER_MODULE}.get_relational_engine", return_value=engine):
        assert await manager.get_by_token("raw-key") is user

    session.commit.assert_awaited_once()
    assert key.last_used_at > stale


@pytest.mark.asyncio
async def test_naive_fresh_last_used_at_skips_write_without_error():
    """SQLite hands back tz-naive datetimes; the throttle must normalize them
    instead of raising TypeError on the aware-naive subtraction. Called
    directly (not through get_by_token) so a TypeError fails the test instead
    of being swallowed by the auth path's best-effort except."""
    session = MagicMock()
    session.commit = AsyncMock()
    fresh_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    key = SimpleNamespace(id=uuid4(), last_used_at=fresh_naive)

    await UserManager._touch_api_key_last_used(session, key)

    session.commit.assert_not_awaited()
    assert key.last_used_at == fresh_naive


@pytest.mark.asyncio
async def test_naive_stale_last_used_at_refreshes():
    session = MagicMock()
    session.commit = AsyncMock()
    stale_naive = (datetime.now(timezone.utc) - timedelta(minutes=10)).replace(tzinfo=None)
    key = SimpleNamespace(id=uuid4(), last_used_at=stale_naive)

    await UserManager._touch_api_key_last_used(session, key)

    session.commit.assert_awaited_once()
    assert key.last_used_at.tzinfo is not None


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
