"""Unit tests for GET /integrations/status and its plugin_status sources.

Two layers, matching how the endpoint is built:

* Router tests exercise the aggregation/degradation logic with every
  status source mocked at module level (the same pattern as
  test_get_integrations_router.py), including the guarantee that no token
  material ever reaches the serialized JSON.
* plugin_status tests run the real SQL against an in-memory SQLite engine
  seeded with UserApiKey / SessionRecord rows, so the grouped queries —
  and in particular the LIKE-escape regression for the legacy prefixes —
  are tested against a real dialect, not mocks.
"""

import importlib
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.api.v1.integrations.routers.get_integrations_router import get_integrations_router
from cognee.infrastructure.databases.relational import Base
from cognee.modules.agents.models import AgentConnection
from cognee.modules.agents.registry import AGENT_CONFIG_NAME
from cognee.modules.integrations.base import OAuthInstallation, OAuthIntegration
from cognee.modules.integrations.plugin_status import (
    SOURCE_IDENTITY,
    SOURCE_REGISTRY,
    PluginStatusRow,
    identity_plugin_statuses,
    legacy_plugin_statuses,
    merge_plugin_statuses,
    registry_plugin_statuses,
)
from cognee.modules.integrations.plugins import KNOWN_PLUGINS
from cognee.modules.integrations.registry import supported_integrations, use_integration
from cognee.modules.session_lifecycle.models import SessionRecord
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models.UserApiKey import UserApiKey

USER_ID = uuid4()

# Import the submodules explicitly: the packages' __init__.py rebind these
# names to functions, shadowing the submodules (see the fuller explanation in
# test_get_integrations_router.py).
_router_module = importlib.import_module(
    "cognee.api.v1.integrations.routers.get_integrations_router"
)
_status_module = importlib.import_module("cognee.modules.integrations.plugin_status")


class _FakeUser:
    id = USER_ID
    tenant_id = None


class _FakeIntegration(OAuthIntegration):
    provider = "fake"
    settings_cls = None

    def authorize_url(self, state):
        return f"https://fake.example/authorize?state={state}"

    async def exchange_code(self, code):
        return {"code": code}

    def parse_installation(self, token_response):
        return OAuthInstallation(provider_account_id="ACC1", token_payload={})

    def state_signing_secret(self):
        return "fake-secret"

    def frontend_base_url(self):
        return "https://app.example.com"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: _FakeUser()

    before = dict(supported_integrations)
    supported_integrations.clear()
    use_integration(_FakeIntegration())

    yield TestClient(app)

    supported_integrations.clear()
    supported_integrations.update(before)


@contextmanager
def _quiet_plugin_sources():
    """Blank every plugin-status source so router tests isolate one at a time."""
    with (
        patch.object(_router_module, "identity_plugin_statuses", new=AsyncMock(return_value={})),
        patch.object(_router_module, "legacy_plugin_statuses", new=AsyncMock(return_value={})),
        patch.object(_router_module, "registry_plugin_statuses", new=AsyncMock(return_value={})),
        patch.object(_router_module, "visible_user_ids", new=AsyncMock(return_value=[USER_ID])),
    ):
        yield


# --------------------------------------------------------------------------- #
# Router: integrations half
# --------------------------------------------------------------------------- #


def test_registered_provider_without_credential_reports_disconnected(client):
    with (
        patch.object(
            _router_module, "list_active_credentials_for_user", new=AsyncMock(return_value={})
        ),
        _quiet_plugin_sources(),
    ):
        response = client.get("/api/v1/integrations/status")

    assert response.status_code == 200
    integrations = response.json()["integrations"]
    assert integrations == [
        {
            "provider": "fake",
            "connected": False,
            "accountLabel": None,
            "providerAccountId": None,
            "connectedAt": None,
        }
    ]


def test_connected_provider_exposes_display_fields_and_no_token_material(client):
    # A realistic credential row: display fields AND encrypted token
    # material. Only the former may ever reach the wire.
    credential = SimpleNamespace(
        provider="fake",
        account_label="Acme Workspace",
        provider_account_id="ACC1",
        created_at=datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc),
        ciphertext=b"secret-ciphertext",
        nonce=b"secret-nonce",
        encryption_version=1,
        key_id="k1",
    )
    with (
        patch.object(
            _router_module,
            "list_active_credentials_for_user",
            new=AsyncMock(return_value={"fake": credential}),
        ),
        _quiet_plugin_sources(),
    ):
        response = client.get("/api/v1/integrations/status")

    (row,) = response.json()["integrations"]
    assert row["connected"] is True
    assert row["accountLabel"] == "Acme Workspace"
    assert row["providerAccountId"] == "ACC1"
    # Offset required: SQLite hands back tz-naive datetimes, and an
    # offset-less ISO string gets parsed as local time by JS Date.
    assert row["connectedAt"] in ("2026-08-01T12:00:00Z", "2026-08-01T12:00:00+00:00")
    # Whitelist, not blacklist: the serialized row is exactly the display
    # fields — nothing token-shaped can leak through renames.
    assert set(row) == {"provider", "connected", "accountLabel", "providerAccountId", "connectedAt"}
    body_text = response.text.lower()
    for forbidden in ("ciphertext", "nonce", "token", "apikey", "api_key"):
        assert forbidden not in body_text


# --------------------------------------------------------------------------- #
# Router: plugins half (sources mocked)
# --------------------------------------------------------------------------- #


def test_every_known_plugin_appears_disconnected_by_default(client):
    with (
        patch.object(
            _router_module, "list_active_credentials_for_user", new=AsyncMock(return_value={})
        ),
        _quiet_plugin_sources(),
    ):
        response = client.get("/api/v1/integrations/status")

    plugins = response.json()["plugins"]
    assert [p["key"] for p in plugins] == list(KNOWN_PLUGINS)
    for plugin in plugins:
        assert plugin["connected"] is False
        assert plugin["agentId"] is None
        assert plugin["sessionCount"] == 0
        assert plugin["source"] is None


def test_identity_row_surfaces_through_the_dto(client):
    agent_id = uuid4()
    identity_row = PluginStatusRow(
        key="claude-code",
        connected=True,
        agent_id=agent_id,
        provisioned_at=datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc),
        last_active_at=datetime(2026, 8, 18, 9, 30, tzinfo=timezone.utc),
        session_count=14,
        source=SOURCE_IDENTITY,
    )
    with (
        patch.object(
            _router_module, "list_active_credentials_for_user", new=AsyncMock(return_value={})
        ),
        patch.object(
            _router_module,
            "identity_plugin_statuses",
            new=AsyncMock(return_value={"claude-code": identity_row}),
        ),
        patch.object(_router_module, "legacy_plugin_statuses", new=AsyncMock(return_value={})),
        patch.object(_router_module, "registry_plugin_statuses", new=AsyncMock(return_value={})),
        patch.object(_router_module, "visible_user_ids", new=AsyncMock(return_value=[USER_ID])),
    ):
        response = client.get("/api/v1/integrations/status")

    by_key = {p["key"]: p for p in response.json()["plugins"]}
    claude = by_key["claude-code"]
    assert claude["connected"] is True
    assert claude["agentId"] == str(agent_id)
    assert claude["provisionedAt"].startswith("2026-08-10T08:00:00")
    assert claude["lastActiveAt"].startswith("2026-08-18T09:30:00")
    assert claude["sessionCount"] == 14
    assert claude["source"] == "identity"


def test_legacy_lookup_excludes_identity_provisioned_keys(client):
    identity_row = PluginStatusRow(key="claude-code", connected=True, source=SOURCE_IDENTITY)
    legacy_lookup = AsyncMock(return_value={})
    with (
        patch.object(
            _router_module, "list_active_credentials_for_user", new=AsyncMock(return_value={})
        ),
        patch.object(
            _router_module,
            "identity_plugin_statuses",
            new=AsyncMock(return_value={"claude-code": identity_row}),
        ),
        patch.object(_router_module, "legacy_plugin_statuses", new=legacy_lookup),
        patch.object(_router_module, "registry_plugin_statuses", new=AsyncMock(return_value={})),
        patch.object(_router_module, "visible_user_ids", new=AsyncMock(return_value=[USER_ID])),
    ):
        client.get("/api/v1/integrations/status")

    legacy_lookup.assert_awaited_once_with([USER_ID], exclude_keys={"claude-code"})


def test_each_failing_source_degrades_its_section_never_500s(client):
    boom = AsyncMock(side_effect=RuntimeError("db went away"))
    with (
        patch.object(_router_module, "list_active_credentials_for_user", new=boom),
        patch.object(_router_module, "identity_plugin_statuses", new=boom),
        patch.object(_router_module, "legacy_plugin_statuses", new=boom),
        patch.object(_router_module, "registry_plugin_statuses", new=boom),
        patch.object(_router_module, "visible_user_ids", new=boom),
    ):
        response = client.get("/api/v1/integrations/status")

    assert response.status_code == 200
    body = response.json()
    # Every section is present at its degraded default, not missing.
    assert [i["provider"] for i in body["integrations"]] == ["fake"]
    assert body["integrations"][0]["connected"] is False
    assert [p["key"] for p in body["plugins"]] == list(KNOWN_PLUGINS)
    assert all(p["connected"] is False for p in body["plugins"])


def test_failing_credentials_source_leaves_plugin_section_intact(client):
    """Partial failure: only the integrations half degrades; a healthy
    identity source still surfaces its connected plugin row."""
    identity_row = PluginStatusRow(key="claude-code", connected=True, source=SOURCE_IDENTITY)
    with (
        patch.object(
            _router_module,
            "list_active_credentials_for_user",
            new=AsyncMock(side_effect=RuntimeError("credentials db went away")),
        ),
        patch.object(
            _router_module,
            "identity_plugin_statuses",
            new=AsyncMock(return_value={"claude-code": identity_row}),
        ),
        patch.object(_router_module, "legacy_plugin_statuses", new=AsyncMock(return_value={})),
        patch.object(_router_module, "registry_plugin_statuses", new=AsyncMock(return_value={})),
        patch.object(_router_module, "visible_user_ids", new=AsyncMock(return_value=[USER_ID])),
    ):
        response = client.get("/api/v1/integrations/status")

    assert response.status_code == 200
    body = response.json()
    assert body["integrations"][0]["connected"] is False  # degraded half
    by_key = {p["key"]: p for p in body["plugins"]}
    assert by_key["claude-code"]["connected"] is True  # healthy half untouched
    assert by_key["claude-code"]["source"] == "identity"


def test_failing_identity_source_leaves_integrations_section_intact(client):
    """Partial failure, other direction: identity blows up, but a healthy
    credential still reports its provider connected."""
    credential = SimpleNamespace(
        provider="fake",
        account_label="Acme Workspace",
        provider_account_id="ACC1",
        created_at=datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    with (
        patch.object(
            _router_module,
            "list_active_credentials_for_user",
            new=AsyncMock(return_value={"fake": credential}),
        ),
        patch.object(
            _router_module,
            "identity_plugin_statuses",
            new=AsyncMock(side_effect=RuntimeError("identity lookup went away")),
        ),
        patch.object(_router_module, "legacy_plugin_statuses", new=AsyncMock(return_value={})),
        patch.object(_router_module, "registry_plugin_statuses", new=AsyncMock(return_value={})),
        patch.object(_router_module, "visible_user_ids", new=AsyncMock(return_value=[USER_ID])),
    ):
        response = client.get("/api/v1/integrations/status")

    assert response.status_code == 200
    body = response.json()
    assert body["integrations"][0]["connected"] is True  # healthy half untouched
    assert all(p["connected"] is False for p in body["plugins"])  # degraded half


def test_status_rejects_unauthenticated_callers():
    # No dependency override: the real auth stack must gate the aggregate
    # status route just like the provisioning routes.
    app = FastAPI()
    app.include_router(get_integrations_router(), prefix="/api/v1/integrations")
    response = TestClient(app).get("/api/v1/integrations/status")
    assert response.status_code == 401


# --------------------------------------------------------------------------- #
# plugin_status sources against a real in-memory SQLite engine
# --------------------------------------------------------------------------- #


async def _sqlite_engine(*models):
    engine = create_async_engine("sqlite+aiosqlite://")
    tables = [model.__table__ for model in models]
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, tables=tables))
    maker = async_sessionmaker(engine, expire_on_commit=False)
    return SimpleNamespace(get_async_session=maker), engine


def _session_record(session_id, user_id, last_activity_at):
    return SessionRecord(
        session_id=session_id,
        user_id=user_id,
        status="running",
        started_at=last_activity_at - timedelta(minutes=5),
        last_activity_at=last_activity_at,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        error_count=0,
    )


def _identity_config(plugin_key, provisioned_at="2026-08-10T08:00:00+00:00"):
    return [
        {
            "name": AGENT_CONFIG_NAME,
            "configuration": {"plugin": {"key": plugin_key, "provisioned_at": provisioned_at}},
        }
    ]


def _agent_email(plugin_key, parent_id):
    """The deterministic internal email ``create_agent`` derives."""
    return f"{plugin_key}+{parent_id}@cognee.agent"


@pytest.mark.asyncio
async def test_identity_status_joins_keys_and_sessions():
    parent_id = uuid4()
    agent_id = uuid4()
    key_last_used = datetime(2026, 8, 17, 8, 0, tzinfo=timezone.utc)
    latest_session = datetime(2026, 8, 18, 9, 30, tzinfo=timezone.utc)

    fake_engine, engine = await _sqlite_engine(UserApiKey, SessionRecord)
    try:
        async with fake_engine.get_async_session() as session:
            session.add(
                UserApiKey(
                    id=uuid4(),
                    user_id=agent_id,
                    api_key="hashed",
                    label="cc-key",
                    last_used_at=key_last_used,
                )
            )
            session.add(_session_record("cc_first", agent_id, latest_session))
            session.add(_session_record("cc_second", agent_id, latest_session - timedelta(days=1)))
            await session.commit()

        with (
            patch.object(
                _status_module,
                "child_agent_emails",
                new=AsyncMock(return_value={agent_id: _agent_email("claude-code", parent_id)}),
            ),
            patch.object(
                _status_module,
                "get_principal_all_configuration",
                new=AsyncMock(return_value=_identity_config("claude-code")),
            ),
            patch.object(_status_module, "get_relational_engine", return_value=fake_engine),
        ):
            statuses = await identity_plugin_statuses(parent_id)
    finally:
        await engine.dispose()

    row = statuses["claude-code"]
    assert row.connected is True
    assert row.agent_id == agent_id
    assert row.provisioned_at == datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc)
    assert row.session_count == 2
    # Sessions are more recent than the key's last auth here — max wins.
    assert row.last_active_at == latest_session
    assert row.source == SOURCE_IDENTITY


@pytest.mark.asyncio
async def test_identity_status_revoked_key_reports_disconnected():
    """Disconnect deletes the agent's keys: no UserApiKey row → connected
    False, while the identity row itself (agent, provisioned_at) survives."""
    parent_id = uuid4()
    agent_id = uuid4()
    fake_engine, engine = await _sqlite_engine(UserApiKey, SessionRecord)
    try:
        with (
            patch.object(
                _status_module,
                "child_agent_emails",
                new=AsyncMock(return_value={agent_id: _agent_email("codex", parent_id)}),
            ),
            patch.object(
                _status_module,
                "get_principal_all_configuration",
                new=AsyncMock(return_value=_identity_config("codex")),
            ),
            patch.object(_status_module, "get_relational_engine", return_value=fake_engine),
        ):
            statuses = await identity_plugin_statuses(parent_id)
    finally:
        await engine.dispose()

    row = statuses["codex"]
    assert row.connected is False
    assert row.agent_id == agent_id
    assert row.session_count == 0
    assert row.last_active_at is None


@pytest.mark.asyncio
async def test_identity_status_derives_key_from_email_not_config_blob():
    """A child agent can rewrite its own configuration blob through the
    public store_user_configuration endpoint. The email is server-assigned,
    so a blob claiming another plugin's key must not let the agent
    impersonate (or clobber) that plugin's status row."""
    parent_id = uuid4()
    agent_id = uuid4()
    fake_engine, engine = await _sqlite_engine(UserApiKey, SessionRecord)
    try:
        with (
            patch.object(
                _status_module,
                "child_agent_emails",
                new=AsyncMock(return_value={agent_id: _agent_email("claude-code", parent_id)}),
            ),
            patch.object(
                _status_module,
                "get_principal_all_configuration",
                # Tampered blob claims to be the codex plugin.
                new=AsyncMock(return_value=_identity_config("codex")),
            ),
            patch.object(_status_module, "get_relational_engine", return_value=fake_engine),
        ):
            statuses = await identity_plugin_statuses(parent_id)
    finally:
        await engine.dispose()

    # The row lands on the email-derived key; the claimed key is ignored.
    assert set(statuses) == {"claude-code"}
    assert statuses["claude-code"].agent_id == agent_id


@pytest.mark.asyncio
async def test_identity_status_drops_unparseable_provisioned_at():
    """provisioned_at comes from the agent-writable blob: a non-datetime
    value degrades to None instead of 500ing the status page later at DTO
    validation."""
    parent_id = uuid4()
    agent_id = uuid4()
    fake_engine, engine = await _sqlite_engine(UserApiKey, SessionRecord)
    try:
        with (
            patch.object(
                _status_module,
                "child_agent_emails",
                new=AsyncMock(return_value={agent_id: _agent_email("claude-code", parent_id)}),
            ),
            patch.object(
                _status_module,
                "get_principal_all_configuration",
                new=AsyncMock(
                    return_value=_identity_config("claude-code", provisioned_at="not-a-date")
                ),
            ),
            patch.object(_status_module, "get_relational_engine", return_value=fake_engine),
        ):
            statuses = await identity_plugin_statuses(parent_id)
    finally:
        await engine.dispose()

    row = statuses["claude-code"]
    assert row.provisioned_at is None  # dropped, row itself survives
    assert row.agent_id == agent_id


@pytest.mark.asyncio
async def test_legacy_status_escapes_like_and_scopes_to_visible_users():
    """The LIKE-escape regression: ``ccx_...`` ids must not count toward the
    ``cc_`` prefix, and other users' sessions must not count at all."""
    me = uuid4()
    someone_else = uuid4()
    now = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)

    fake_engine, engine = await _sqlite_engine(SessionRecord)
    try:
        async with fake_engine.get_async_session() as session:
            session.add(_session_record("cc_mine", me, now))
            session.add(_session_record("ccx_not_claude", me, now))  # must NOT count as cc_
            session.add(_session_record("codex_mine", me, now - timedelta(hours=1)))
            session.add(_session_record("cc_theirs", someone_else, now))  # not visible to me
            session.add(_session_record("unprefixed", me, now))
            await session.commit()

        with patch.object(_status_module, "get_relational_engine", return_value=fake_engine):
            statuses = await legacy_plugin_statuses([me])
    finally:
        await engine.dispose()

    assert set(statuses) == {"claude-code", "codex"}
    claude = statuses["claude-code"]
    assert claude.connected is True
    assert claude.session_count == 1  # cc_mine only: ccx_ and cc_theirs excluded
    assert claude.last_active_at == now
    assert claude.source == "sessions-legacy"
    assert statuses["codex"].session_count == 1


@pytest.mark.asyncio
async def test_legacy_status_skips_identity_provisioned_plugins():
    me = uuid4()
    now = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)
    fake_engine, engine = await _sqlite_engine(SessionRecord)
    try:
        async with fake_engine.get_async_session() as session:
            session.add(_session_record("cc_old_install", me, now))
            session.add(_session_record("codex_old_install", me, now))
            await session.commit()

        with patch.object(_status_module, "get_relational_engine", return_value=fake_engine):
            statuses = await legacy_plugin_statuses([me], exclude_keys={"claude-code"})
    finally:
        await engine.dispose()

    # claude-code has its own identity now — prefix inference for it is off.
    assert set(statuses) == {"codex"}


@pytest.mark.asyncio
async def test_registry_status_buckets_by_metadata_and_connection_type():
    """Connections in the caller's OWN blob: claims are trusted as-is (only
    the caller can write that blob)."""
    seen_at = datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc)
    connections = [
        # Provisioned connection: metadata plugin_key wins over type.
        AgentConnection(
            id="a",
            agent_session_name="plugin:claude-code",
            type="claude_code",
            status="active",
            last_active_at=seen_at,
            metadata={"plugin_key": "claude-code"},
        ),
        # Bare MCP client, no metadata: bucketed by connection type.
        AgentConnection(
            id="b",
            agent_session_name="my-mcp",
            type="mcp",
            status="active",
            last_active_at=seen_at + timedelta(hours=1),
        ),
        # Generic sdk connection lands on the "api" card.
        AgentConnection(
            id="c",
            agent_session_name="script",
            type="sdk",
            status="active",
            last_active_at=seen_at,
        ),
        # Unmapped connection type, no metadata: no plugin bucket → skipped.
        AgentConnection(
            id="d",
            agent_session_name="pipeline",
            type="workflow",
            status="active",
            last_active_at=seen_at,
        ),
        # Bogus user-supplied metadata plugin_key: not a known plugin → skipped.
        AgentConnection(
            id="e",
            agent_session_name="impostor",
            type="unknown",
            status="active",
            last_active_at=seen_at,
            metadata={"plugin_key": "not-a-plugin"},
        ),
    ]
    with (
        patch.object(
            _status_module,
            "list_persisted_agent_connections",
            new=AsyncMock(return_value=connections),
        ),
        patch.object(_status_module, "child_agent_emails", new=AsyncMock(return_value={})),
    ):
        statuses = await registry_plugin_statuses(uuid4())

    # The workflow and bogus-metadata connections produce no rows at all.
    assert set(statuses) == {"claude-code", "mcp", "api"}
    assert all(row.connected for row in statuses.values())
    assert statuses["mcp"].last_active_at == seen_at + timedelta(hours=1)
    assert all(row.source == SOURCE_REGISTRY for row in statuses.values())


@pytest.mark.asyncio
async def test_registry_status_pins_child_blob_connections_to_email_identity():
    """A child agent's connection blob is agent-writable (public configuration
    endpoint), so its claimed plugin_key / connection type must not be able to
    mark ANOTHER plugin connected. Every connection found in a child blob is
    pinned to the plugin key derived from the agent's server-assigned email;
    non-plugin child agents contribute nothing."""
    parent_id = uuid4()
    claude_agent_id = uuid4()
    custom_agent_id = uuid4()
    seen_at = datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc)

    connections_by_owner = {
        parent_id: [],
        # claude-code agent's blob, tampered to claim codex on both signals.
        claude_agent_id: [
            AgentConnection(
                id="spoof",
                agent_session_name="plugin:codex",
                type="codex",
                status="active",
                last_active_at=seen_at,
                metadata={"plugin_key": "codex"},
            )
        ],
        # Manually created non-plugin agent claiming an MCP connection.
        custom_agent_id: [
            AgentConnection(
                id="custom",
                agent_session_name="my-bot",
                type="mcp",
                status="active",
                last_active_at=seen_at,
            )
        ],
    }

    async def _by_owner(user_ids, active_only=True):
        (owner,) = user_ids
        return connections_by_owner[owner]

    child_emails = {
        claude_agent_id: f"claude-code+{parent_id}@cognee.agent",
        custom_agent_id: f"my-bot+{parent_id}@cognee.agent",
    }
    with (
        patch.object(
            _status_module,
            "list_persisted_agent_connections",
            new=AsyncMock(side_effect=_by_owner),
        ),
        patch.object(
            _status_module, "child_agent_emails", new=AsyncMock(return_value=child_emails)
        ),
    ):
        statuses = await registry_plugin_statuses(parent_id)

    # The spoofed connection lands on the email-derived key, never on codex;
    # the non-plugin agent's mcp claim produces no row at all.
    assert set(statuses) == {"claude-code"}
    assert statuses["claude-code"].connected is True
    assert statuses["claude-code"].last_active_at == seen_at


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #


def test_merge_identity_and_registry_rows_for_same_key():
    agent_id = uuid4()
    provisioned_at = datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc)
    identity = {
        "claude-code": PluginStatusRow(
            key="claude-code",
            connected=False,  # every key revoked...
            agent_id=agent_id,
            provisioned_at=provisioned_at,
            last_active_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
            session_count=3,
            source=SOURCE_IDENTITY,
        )
    }
    registry = {
        "claude-code": PluginStatusRow(
            key="claude-code",
            connected=True,  # ...but a stale registry connection lingers
            last_active_at=datetime(2026, 8, 15, tzinfo=timezone.utc),
            source=SOURCE_REGISTRY,
        ),
        "mcp": PluginStatusRow(key="mcp", connected=True, source=SOURCE_REGISTRY),
    }

    merged = merge_plugin_statuses(identity, registry)

    assert set(merged) == {"claude-code", "mcp"}  # same key → one row
    row = merged["claude-code"]
    # Identity's key-existence check is authoritative: revoking every API
    # key means disconnected, no matter what the registry still claims.
    assert row.connected is False
    assert row.last_active_at == datetime(2026, 8, 15, tzinfo=timezone.utc)  # max
    assert row.agent_id == agent_id  # identity wins agentId...
    assert row.provisioned_at == provisioned_at  # ...and provisionedAt
    assert row.source == SOURCE_IDENTITY
    assert row.session_count == 3
    assert merged["mcp"].source == SOURCE_REGISTRY


def test_merge_ors_connected_for_non_identity_rows():
    """The identity-authoritative rule is scoped: a legacy (or default) base
    row still ORs connected in from the registry."""
    base = {"mcp": PluginStatusRow(key="mcp", connected=False)}
    extra = {"mcp": PluginStatusRow(key="mcp", connected=True, source=SOURCE_REGISTRY)}

    merged = merge_plugin_statuses(base, extra)

    assert merged["mcp"].connected is True
