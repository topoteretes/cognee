"""Unit tests for the agents SDK namespace (``cognee.agents``) and the
``AgentsCommand`` CLI command.

These tests run WITHOUT a live LLM, network, or relational database. The
backend ``cognee.modules.agents.*`` functions and permission helpers are
monkeypatched at the names imported into the SDK module, so the SDK's own
glue logic (display-email stripping, permission ordering, error translation,
return-shape) is exercised against fakes.

If the SDK module is not yet importable (it is built alongside these tests)
the whole module is skipped rather than erroring at collection time.
"""

import argparse
import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

agents_sdk = pytest.importorskip(
    "cognee.api.v1.agents.agents",
    reason="agents SDK namespace not yet available",
)

agents = agents_sdk.agents


# --------------------------------------------------------------------------- #
# Fakes / helpers
# --------------------------------------------------------------------------- #


def _make_user(user_id=None):
    """A minimal stand-in for a cognee User object."""
    return SimpleNamespace(id=user_id or uuid4(), tenant_id=None)


def _make_agent_info(parent_id, slug="support", api_key_label="support"):
    """Build an AgentInfo-like object whose email mirrors create_agent output."""
    agent_id = uuid4()
    email = f"{slug}+{parent_id}@cognee.agent"
    agent_user = SimpleNamespace(id=agent_id, email=email, parent_user_id=parent_id)
    return SimpleNamespace(user=agent_user, api_key_label=api_key_label)


def _patch(monkeypatch, name, value):
    """Patch ``name`` on the agents SDK module if it is referenced there."""
    if hasattr(agents_sdk, name):
        monkeypatch.setattr(agents_sdk, name, value)


# --------------------------------------------------------------------------- #
# create -> list -> get -> delete round trip
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_list_get_delete_round_trip(monkeypatch):
    parent = _make_user()
    store = {}  # agent_id -> AgentInfo

    async def fake_create_agent(name, parent_user):
        slug = name.lower().replace(" ", "-")
        info = _make_agent_info(parent_user.id, slug=slug, api_key_label=slug)
        store[info.user.id] = info
        return info.user, "agent-api-key-123"

    async def fake_list_agents(owner_id):
        return [i for i in store.values() if i.user.parent_user_id == owner_id]

    async def fake_get_agent(agent_id, owner_id):
        info = store.get(agent_id)
        if info is None:
            raise LookupError("Agent not found")
        if info.user.parent_user_id != owner_id:
            raise PermissionError("Not authorized to view this agent")
        return info

    async def fake_delete_agent(agent_id, owner_id):
        info = store.get(agent_id)
        if info is None:
            raise LookupError("Agent not found")
        if info.user.parent_user_id != owner_id:
            raise PermissionError("Not authorized to delete this agent")
        del store[agent_id]

    _patch(monkeypatch, "create_agent", fake_create_agent)
    _patch(monkeypatch, "list_agents", fake_list_agents)
    _patch(monkeypatch, "get_agent", fake_get_agent)
    _patch(monkeypatch, "delete_agent", fake_delete_agent)

    # create (no datasets -> no permission helpers touched)
    created = await agents.create("Support Bot", user=parent)
    assert created["agent_api_key"] == "agent-api-key-123"
    # display email strips the '+{parent_id}' segment
    assert created["agent_email"] == "support-bot@cognee.agent"
    assert "+" not in created["agent_email"]
    agent_id = created["agent_id"]
    assert agent_id == str(next(iter(store)))

    # list shows the created agent
    listed = await agents.list(user=parent)
    assert len(listed) == 1
    assert listed[0]["agent_id"] == agent_id
    assert listed[0]["agent_email"] == "support-bot@cognee.agent"
    assert listed[0]["api_key_label"] == "support-bot"

    # get returns it
    got = await agents.get(agent_id, user=parent)
    assert got["agent_id"] == agent_id
    assert got["agent_email"] == "support-bot@cognee.agent"

    # delete removes it
    result = await agents.delete(agent_id, user=parent)
    assert result is None
    assert await agents.list(user=parent) == []


@pytest.mark.asyncio
async def test_create_grants_agent_read_write_after_caller_check(monkeypatch):
    parent = _make_user()
    dataset_id = uuid4()
    call_order = []

    async def fake_create_agent(name, parent_user):
        info = _make_agent_info(parent_user.id, slug="bot")
        return info.user, "key"

    async def fake_get_authorized_dataset(user, ds_id, permission="read"):
        call_order.append(("authorize", user.id, ds_id, permission))
        return SimpleNamespace(id=ds_id)

    granted = []

    async def fake_give_permission_on_dataset(user, ds_id, permission):
        call_order.append(("grant", user.id, ds_id, permission))
        granted.append((user.id, ds_id, permission))

    _patch(monkeypatch, "create_agent", fake_create_agent)
    _patch(monkeypatch, "get_authorized_dataset", fake_get_authorized_dataset)
    _patch(monkeypatch, "give_permission_on_dataset", fake_give_permission_on_dataset)

    await agents.create("Bot", datasets=[dataset_id], user=parent)

    # caller authorization happens before any grant to the agent
    assert call_order[0][0] == "authorize"
    assert call_order[0][1] == parent.id
    grant_perms = {(g[2]) for g in granted}
    assert grant_perms == {"read", "write"}
    # the agent (not the parent) receives the grants
    assert all(g[0] != parent.id for g in granted)


@pytest.mark.asyncio
async def test_create_resolves_dataset_name(monkeypatch):
    parent = _make_user()
    resolved_id = uuid4()

    async def fake_create_agent(name, parent_user):
        info = _make_agent_info(parent_user.id, slug="bot")
        return info.user, "key"

    async def fake_get_datasets_by_name(name, owner_id):
        if name == "known":
            return [SimpleNamespace(id=resolved_id, name="known")]
        return []

    authorized = []

    async def fake_get_authorized_dataset(user, ds_id, permission="read"):
        authorized.append(ds_id)
        return SimpleNamespace(id=ds_id)

    async def fake_give_permission_on_dataset(user, ds_id, permission):
        pass

    _patch(monkeypatch, "create_agent", fake_create_agent)
    _patch(monkeypatch, "get_datasets_by_name", fake_get_datasets_by_name)
    _patch(monkeypatch, "get_authorized_dataset", fake_get_authorized_dataset)
    _patch(monkeypatch, "give_permission_on_dataset", fake_give_permission_on_dataset)

    await agents.create("Bot", datasets=["known"], user=parent)
    assert resolved_id in authorized


# --------------------------------------------------------------------------- #
# Permissioning
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_unauthorized_dataset_raises(monkeypatch):
    from cognee.modules.users.exceptions import PermissionDeniedError

    parent = _make_user()
    dataset_id = uuid4()

    async def fake_create_agent(name, parent_user):
        info = _make_agent_info(parent_user.id, slug="bot")
        return info.user, "key"

    async def fake_get_authorized_dataset(user, ds_id, permission="read"):
        # Either raising or returning None signals "no access".
        return None

    grant_calls = []

    async def fake_give_permission_on_dataset(user, ds_id, permission):
        grant_calls.append((ds_id, permission))

    _patch(monkeypatch, "create_agent", fake_create_agent)
    _patch(monkeypatch, "get_authorized_dataset", fake_get_authorized_dataset)
    _patch(monkeypatch, "give_permission_on_dataset", fake_give_permission_on_dataset)

    with pytest.raises((PermissionDeniedError, ValueError)):
        await agents.create("Bot", datasets=[dataset_id], user=parent)

    # the agent must never be granted access to a dataset the caller can't read
    assert grant_calls == []


@pytest.mark.asyncio
async def test_get_other_users_agent_is_denied(monkeypatch):
    from cognee.modules.users.exceptions import PermissionDeniedError

    owner = _make_user()
    other_agent_id = uuid4()

    async def fake_get_agent(agent_id, owner_id):
        # backend raises PermissionError when parent_user_id != owner_id
        raise PermissionError("Not authorized to view this agent")

    _patch(monkeypatch, "get_agent", fake_get_agent)

    with pytest.raises((PermissionDeniedError, PermissionError)):
        await agents.get(other_agent_id, user=owner)


@pytest.mark.asyncio
async def test_delete_other_users_agent_is_denied(monkeypatch):
    from cognee.modules.users.exceptions import PermissionDeniedError

    owner = _make_user()
    other_agent_id = uuid4()

    async def fake_delete_agent(agent_id, owner_id):
        raise PermissionError("Not authorized to delete this agent")

    _patch(monkeypatch, "delete_agent", fake_delete_agent)

    with pytest.raises((PermissionDeniedError, PermissionError)):
        await agents.delete(other_agent_id, user=owner)


@pytest.mark.asyncio
async def test_get_missing_agent_raises_value_error(monkeypatch):
    owner = _make_user()
    missing_id = uuid4()

    async def fake_get_agent(agent_id, owner_id):
        raise LookupError("Agent not found")

    _patch(monkeypatch, "get_agent", fake_get_agent)

    with pytest.raises(ValueError):
        await agents.get(missing_id, user=owner)


# --------------------------------------------------------------------------- #
# register -> list_connections -> unregister
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_register_list_connections_unregister(monkeypatch):
    from cognee.modules.agents.models import (
        AgentConnection,
        AgentsListResponse,
    )

    user = _make_user()
    session_name = "session-A"

    captured_request = {}

    async def fake_register_agent(u, request):
        captured_request["request"] = request
        return AgentConnection(
            id=f"{u.id}:{request.agent_session_name}",
            agent_session_name=request.agent_session_name,
            type=request.type,
            memory_mode=request.memory_mode,
            user_id=u.id,
        )

    async def fake_list_agent_connections(**kwargs):
        conn = AgentConnection(
            id=f"{user.id}:{session_name}",
            agent_session_name=session_name,
            user_id=user.id,
            status="active",
        )
        return AgentsListResponse(
            agents=[conn],
            memory_sources=[],
            total=1,
            limit=kwargs.get("limit", 50),
            offset=kwargs.get("offset", 0),
            has_more=False,
        )

    async def fake_unregister_agent(u, request):
        # mimic active-count drop to zero after unregister
        return 0

    _patch(monkeypatch, "register_agent", fake_register_agent)
    _patch(monkeypatch, "list_agent_connections", fake_list_agent_connections)
    _patch(monkeypatch, "unregister_agent", fake_unregister_agent)

    # register
    registered = await agents.register(session_name, user=user, memory_mode="hybrid")
    assert isinstance(registered, dict)
    assert registered["agent_session_name"] == session_name
    # the SDK should have forwarded the kwargs into the request model
    assert captured_request["request"].memory_mode == "hybrid"

    # list_connections shows the connection (range_key kwarg, not range)
    listing = await agents.list_connections(user=user, range_key="30d")
    assert isinstance(listing, dict)
    assert listing["total"] == 1
    assert listing["agents"][0]["agent_session_name"] == session_name

    # unregister deactivates it -> active count int
    count = await agents.unregister(session_name, user=user)
    assert count == 0


@pytest.mark.asyncio
async def test_register_unauthorized_dataset_id_raises(monkeypatch):
    from cognee.modules.users.exceptions import PermissionDeniedError

    user = _make_user()
    bad_dataset = str(uuid4())

    register_called = []

    async def fake_get_authorized_dataset(u, ds_id, permission="read"):
        return None  # no access

    async def fake_register_agent(u, request):
        register_called.append(request)
        return SimpleNamespace(model_dump=lambda mode="json": {})

    _patch(monkeypatch, "get_authorized_dataset", fake_get_authorized_dataset)
    _patch(monkeypatch, "register_agent", fake_register_agent)

    with pytest.raises((PermissionDeniedError, ValueError)):
        await agents.register("sess", user=user, dataset_ids=[bad_dataset])

    # register must NOT be reached if a dataset is not accessible
    assert register_called == []


@pytest.mark.asyncio
async def test_list_connections_drops_unscoped_connections(monkeypatch):
    """Connections with no owning user AND no datasets are visible-to-all in the
    backend; the SDK must defensively filter them out and adjust the totals."""
    from cognee.modules.agents.models import AgentConnection, AgentsListResponse

    user = _make_user()

    async def fake_list_agent_connections(**kwargs):
        scoped = AgentConnection(
            id="scoped",
            agent_session_name="mine",
            user_id=user.id,
            status="active",
        )
        # ownerless + datasetless -> backend marks visible to everyone
        leaked = AgentConnection(
            id="leaked",
            agent_session_name="someone-elses",
            user_id=None,
            status="active",
        )
        return AgentsListResponse(
            agents=[scoped, leaked],
            memory_sources=[],
            total=2,
            limit=kwargs.get("limit", 50),
            offset=kwargs.get("offset", 0),
            has_more=False,
        )

    _patch(monkeypatch, "list_agent_connections", fake_list_agent_connections)

    listing = await agents.list_connections(user=user)
    session_names = [a["agent_session_name"] for a in listing["agents"]]
    assert session_names == ["mine"]
    assert "someone-elses" not in session_names
    # total is decremented for the removed leaked connection
    assert listing["total"] == 1


@pytest.mark.asyncio
async def test_get_connection_returns_none_when_missing(monkeypatch):
    user = _make_user()
    agent_id = uuid4()

    async def fake_get_agent_connection_detail(**kwargs):
        return None

    _patch(monkeypatch, "get_agent_connection_detail", fake_get_agent_connection_detail)

    result = await agents.get_connection(agent_id, user=user, agent_session_name="x")
    assert result is None


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #


cli_module = pytest.importorskip(
    "cognee.cli.commands.agents_command",
    reason="AgentsCommand CLI not yet available",
)


def test_agents_command_parser_configures_actions():
    command = cli_module.AgentsCommand()
    assert command.command_string == "agents"

    parser = argparse.ArgumentParser()
    command.configure_parser(parser)

    # parsing a "list" invocation succeeds and selects the list action
    args = parser.parse_args(["list"])
    assert getattr(args, "agents_action") == "list"


def test_agents_command_execute_list(monkeypatch):
    command = cli_module.AgentsCommand()
    parser = argparse.ArgumentParser()
    command.configure_parser(parser)
    args = parser.parse_args(["list"])

    # avoid any DB work: resolve_cli_user and the SDK list call are mocked
    fake_user = _make_user()

    async def fake_resolve_cli_user(_user_id, strict=False):
        return fake_user

    monkeypatch.setattr("cognee.cli.user_resolution.resolve_cli_user", fake_resolve_cli_user)

    listed_rows = [
        {
            "agent_id": str(uuid4()),
            "agent_email": "bot@cognee.agent",
            "api_key_label": "bot",
        }
    ]

    async def fake_list(user=None):
        assert user is fake_user
        return listed_rows

    import cognee

    # The top-level ``cognee.agents`` export may be wired in a separate edit.
    # Ensure the CLI can resolve it regardless, then patch ``list`` on the
    # SDK class object that the CLI calls through.
    if not hasattr(cognee, "agents"):
        monkeypatch.setattr(cognee, "agents", agents, raising=False)
    monkeypatch.setattr(cognee.agents, "list", staticmethod(fake_list))

    captured = []
    monkeypatch.setattr("cognee.cli.echo.echo", lambda *a, **k: captured.append(a))

    # execute() runs the list action without raising
    command.execute(args)
    assert any("bot@cognee.agent" in str(c) for c in captured)


def test_agents_command_execute_no_action_raises(monkeypatch):
    from cognee.cli.exceptions import CliCommandException

    command = cli_module.AgentsCommand()
    parser = argparse.ArgumentParser()
    command.configure_parser(parser)
    args = parser.parse_args([])  # no subcommand

    monkeypatch.setattr("cognee.cli.echo.error", lambda *a, **k: None)

    with pytest.raises(CliCommandException):
        command.execute(args)


# --------------------------------------------------------------------------- #
# Strict --user-id resolution (no silent fallback for agent commands)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resolve_cli_user_strict_unknown_uuid_raises(monkeypatch):
    """A valid-but-unknown --user-id must be a hard error in strict mode rather
    than silently falling back to the default user."""
    from cognee.cli import user_resolution

    async def fake_get_user(uid):
        raise Exception("user not found")

    async def fake_get_default_user():
        raise AssertionError("strict mode must not fall back to the default user")

    monkeypatch.setattr("cognee.modules.users.methods.get_user", fake_get_user, raising=False)
    monkeypatch.setattr(
        "cognee.modules.users.methods.get_default_user", fake_get_default_user, raising=False
    )

    with pytest.raises(ValueError):
        await user_resolution.resolve_cli_user(str(uuid4()), strict=True)


@pytest.mark.asyncio
async def test_resolve_cli_user_non_strict_falls_back(monkeypatch):
    """Default (non-strict) behaviour still warns and falls back, preserving the
    existing contract for other commands (e.g. datasets)."""
    from cognee.cli import user_resolution

    sentinel = _make_user()

    async def fake_get_user(uid):
        raise Exception("user not found")

    async def fake_get_default_user():
        return sentinel

    monkeypatch.setattr("cognee.modules.users.methods.get_user", fake_get_user, raising=False)
    monkeypatch.setattr(
        "cognee.modules.users.methods.get_default_user", fake_get_default_user, raising=False
    )
    monkeypatch.setattr("cognee.cli.echo.warning", lambda *a, **k: None)

    resolved = await user_resolution.resolve_cli_user(str(uuid4()))
    assert resolved is sentinel
