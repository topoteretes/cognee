"""Unit tests for cognee.modules.session_lifecycle.agent_usage (CLO-434).

Covers the session_records <-> agent_connections join and the
per-(user, agent type) cost aggregation directly, independent of FastAPI —
the HTTP layer (get_sessions_router) is exercised separately as thin wiring.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.session_lifecycle import agent_usage
from cognee.modules.agents.models import AgentConnection, AgentsListResponse
from cognee.modules.session_lifecycle.metrics import SessionListPage, SessionRowWithStatus
from cognee.modules.session_lifecycle.models import SessionRecord
from cognee.modules.users.exceptions import PermissionDeniedError


def _session_row(session_id: str, user_id) -> SessionRowWithStatus:
    record = SessionRecord(
        session_id=session_id,
        user_id=user_id,
        status="completed",
        started_at=datetime.now(timezone.utc),
        last_activity_at=datetime.now(timezone.utc),
        tokens_in=10,
        tokens_out=20,
        cost_usd=0.01,
        error_count=0,
        last_model=None,
    )
    return SessionRowWithStatus(record=record, effective_status="completed")


def _row(user_id, session_id, tokens_in=10, tokens_out=20, cost_usd=0.01):
    return SimpleNamespace(
        user_id=user_id,
        session_id=session_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
    )


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _stmt):
        return _FakeResult(self._rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeEngine:
    def __init__(self, rows):
        self._rows = rows

    def get_async_session(self):
        return _FakeSession(self._rows)


# --------------------------------------------------------------------------- #
# build_sessions_with_agent_info_page
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_merges_matched_and_falls_back_for_unmatched(monkeypatch):
    user_id = uuid4()
    user = SimpleNamespace(id=user_id, tenant_id=None, email="me@example.com")

    matched_row = _session_row("claude-code-123", user_id)
    unmatched_row = _session_row("codex-456", user_id)

    async def fake_list_session_rows(**_kwargs):
        return SessionListPage(sessions=[matched_row, unmatched_row], total=2, limit=50, offset=0)

    async def fake_list_agent_connections(**_kwargs):
        conn = AgentConnection(
            id="conn-1",
            agent_session_name="my-claude-code",
            type="claude_code",
            source="mcp",
            session_id="claude-code-123",
            user_id=user_id,
            origin_function="remember",
        )
        return AgentsListResponse(
            agents=[conn], memory_sources=[], total=1, limit=10000, offset=0, has_more=False
        )

    monkeypatch.setattr(agent_usage, "list_session_rows", fake_list_session_rows)
    monkeypatch.setattr(agent_usage, "list_agent_connections", fake_list_agent_connections)

    result = await agent_usage.build_sessions_with_agent_info_page(
        user=user,
        permitted_dataset_ids=[],
        visible_user_ids=[user_id],
        since=None,
        status_filter=None,
        limit=50,
        offset=0,
        order_by="last_activity_at",
        descending=True,
    )

    assert result["total"] == 2
    sessions_by_id = {s["session_id"]: s for s in result["sessions"]}

    matched = sessions_by_id["claude-code-123"]
    assert matched["agent_type"] == "claude_code"
    assert matched["agent_source"] == "mcp"
    assert matched["agent_session_name"] == "my-claude-code"
    assert matched["origin_function"] == "remember"

    # No registered connection for this session -> prefix-derived fallback.
    unmatched = sessions_by_id["codex-456"]
    assert unmatched["agent_type"] == "codex"
    assert unmatched["agent_source"] is None
    assert unmatched["agent_session_name"] is None


@pytest.mark.asyncio
async def test_warns_when_agent_connection_page_is_capped(monkeypatch, caplog):
    """If a scope has more agent connections than the page cap, sessions past
    the cap silently fall back to the prefix heuristic — that should at least
    be visible in logs instead of failing silently."""
    user_id = uuid4()
    user = SimpleNamespace(id=user_id, tenant_id=None, email="me@example.com")

    row = _session_row("claude-code-123", user_id)

    async def fake_list_session_rows(**_kwargs):
        return SessionListPage(sessions=[row], total=1, limit=50, offset=0)

    async def fake_list_agent_connections(**_kwargs):
        return AgentsListResponse(
            agents=[],
            memory_sources=[],
            total=20_000,
            limit=agent_usage._AGENT_CONNECTIONS_PAGE_CAP,
            offset=0,
            has_more=True,
        )

    monkeypatch.setattr(agent_usage, "list_session_rows", fake_list_session_rows)
    monkeypatch.setattr(agent_usage, "list_agent_connections", fake_list_agent_connections)

    with caplog.at_level("WARNING"):
        await agent_usage.build_sessions_with_agent_info_page(
            user=user,
            permitted_dataset_ids=[],
            visible_user_ids=[user_id],
            since=None,
            status_filter=None,
            limit=50,
            offset=0,
            order_by="last_activity_at",
            descending=True,
        )

    assert any("capped" in record.message for record in caplog.records)


# --------------------------------------------------------------------------- #
# compute_cost_by_user_agent
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_single_user_mode_falls_back_to_caller_only(monkeypatch):
    user_id = uuid4()
    user = SimpleNamespace(id=user_id, tenant_id=None, email="solo@example.com")

    rows = [_row(user_id, "claude-code-1"), _row(user_id, "codex-1")]
    monkeypatch.setattr(agent_usage, "get_relational_engine", lambda: _FakeEngine(rows))

    async def fake_persisted(_user_ids, active_only=False):
        return []

    monkeypatch.setattr(agent_usage, "list_persisted_agent_connections", fake_persisted)
    monkeypatch.setattr(agent_usage, "list_registered_agent_connections", lambda: [])

    result = await agent_usage.compute_cost_by_user_agent(
        user=user, visible_user_ids=[user_id], permitted_dataset_ids=[], since=None
    )
    by_type = {r["agent_type"]: r for r in result}

    assert by_type["claude_code"]["user_email"] == "solo@example.com"
    assert by_type["claude_code"]["session_count"] == 1
    assert by_type["codex"]["session_count"] == 1


@pytest.mark.asyncio
async def test_solo_mode_includes_child_agent_sessions(monkeypatch):
    """No tenant, but the caller has a child agent (e.g. a delegated
    sub-account) — those sessions must still be included, same as
    every other /sessions endpoint does via ``get_visible_user_ids``."""
    user_id = uuid4()
    child_id = uuid4()
    user = SimpleNamespace(id=user_id, tenant_id=None, email="solo@example.com")

    rows = [_row(user_id, "claude-code-1"), _row(child_id, "codex-1")]
    monkeypatch.setattr(agent_usage, "get_relational_engine", lambda: _FakeEngine(rows))

    async def fake_persisted(_user_ids, active_only=False):
        return []

    monkeypatch.setattr(agent_usage, "list_persisted_agent_connections", fake_persisted)
    monkeypatch.setattr(agent_usage, "list_registered_agent_connections", lambda: [])

    result = await agent_usage.compute_cost_by_user_agent(
        user=user, visible_user_ids=[user_id, child_id], permitted_dataset_ids=[], since=None
    )
    by_type = {r["agent_type"]: r for r in result}

    assert by_type["claude_code"]["user_email"] == "solo@example.com"
    assert by_type["codex"]["user_id"] == str(child_id)
    assert by_type["codex"]["user_email"] == "unknown"


@pytest.mark.asyncio
async def test_tenant_mode_aggregates_across_users_and_sorts_by_cost(monkeypatch):
    caller_id = uuid4()
    other_id = uuid4()
    tenant_id = uuid4()
    user = SimpleNamespace(id=caller_id, tenant_id=tenant_id, email="admin@example.com")

    async def fake_get_users_in_tenant(_tenant_id, _user):
        return [
            {"id": str(caller_id), "email": "admin@example.com", "roles": []},
            {"id": str(other_id), "email": "member@example.com", "roles": []},
        ]

    rows = [
        _row(caller_id, "claude-code-1", cost_usd=0.01),
        _row(other_id, "codex-1", cost_usd=5.00),
    ]

    conn = AgentConnection(
        id="conn-1",
        agent_session_name="member-codex",
        type="codex",
        session_id="codex-1",
        user_id=other_id,
    )

    async def fake_persisted(_user_ids, active_only=False):
        return [conn]

    monkeypatch.setattr(agent_usage, "get_users_in_tenant", fake_get_users_in_tenant)
    monkeypatch.setattr(agent_usage, "get_relational_engine", lambda: _FakeEngine(rows))
    monkeypatch.setattr(agent_usage, "list_persisted_agent_connections", fake_persisted)
    monkeypatch.setattr(agent_usage, "list_registered_agent_connections", lambda: [])

    result = await agent_usage.compute_cost_by_user_agent(
        user=user, visible_user_ids=[caller_id], permitted_dataset_ids=[], since=None
    )

    # Highest spender first.
    assert result[0]["user_email"] == "member@example.com"
    assert result[0]["agent_type"] == "codex"
    assert result[0]["cost_usd"] == pytest.approx(5.00)
    assert result[1]["user_email"] == "admin@example.com"
    assert result[1]["agent_type"] == "claude_code"


@pytest.mark.asyncio
async def test_tenant_mode_member_falls_back_to_own_sessions(monkeypatch):
    """A non-admin member isn't denied outright — they just don't get the
    tenant-wide view; the function scopes down to their base visibility
    (self + child agents), same as every other /sessions endpoint."""
    caller_id = uuid4()
    tenant_id = uuid4()
    user = SimpleNamespace(id=caller_id, tenant_id=tenant_id, email="member@example.com")

    async def fake_get_users_in_tenant(_tenant_id, _user):
        raise PermissionDeniedError(message="nope")

    rows = [_row(caller_id, "claude-code-1", cost_usd=0.02)]

    async def fake_persisted(_user_ids, active_only=False):
        return []

    monkeypatch.setattr(agent_usage, "get_users_in_tenant", fake_get_users_in_tenant)
    monkeypatch.setattr(agent_usage, "get_relational_engine", lambda: _FakeEngine(rows))
    monkeypatch.setattr(agent_usage, "list_persisted_agent_connections", fake_persisted)
    monkeypatch.setattr(agent_usage, "list_registered_agent_connections", lambda: [])

    result = await agent_usage.compute_cost_by_user_agent(
        user=user, visible_user_ids=[caller_id], permitted_dataset_ids=[], since=None
    )

    assert len(result) == 1
    assert result[0]["user_id"] == str(caller_id)
    assert result[0]["user_email"] == "member@example.com"
    assert result[0]["agent_type"] == "claude_code"


@pytest.mark.asyncio
async def test_tenant_mode_member_keeps_child_agents_when_denied_tenant_view(monkeypatch):
    """A non-admin member denied the tenant-wide view must still keep
    their own child-agent sessions — the tenant lookup failing shouldn't
    collapse the base visibility down to just the caller."""
    caller_id = uuid4()
    child_id = uuid4()
    tenant_id = uuid4()
    user = SimpleNamespace(id=caller_id, tenant_id=tenant_id, email="member@example.com")

    async def fake_get_users_in_tenant(_tenant_id, _user):
        raise PermissionDeniedError(message="nope")

    rows = [
        _row(caller_id, "claude-code-1", cost_usd=0.02),
        _row(child_id, "codex-1", cost_usd=0.03),
    ]

    async def fake_persisted(_user_ids, active_only=False):
        return []

    monkeypatch.setattr(agent_usage, "get_users_in_tenant", fake_get_users_in_tenant)
    monkeypatch.setattr(agent_usage, "get_relational_engine", lambda: _FakeEngine(rows))
    monkeypatch.setattr(agent_usage, "list_persisted_agent_connections", fake_persisted)
    monkeypatch.setattr(agent_usage, "list_registered_agent_connections", lambda: [])

    result = await agent_usage.compute_cost_by_user_agent(
        user=user, visible_user_ids=[caller_id, child_id], permitted_dataset_ids=[], since=None
    )
    by_type = {r["agent_type"]: r for r in result}

    assert len(result) == 2
    assert by_type["claude_code"]["user_email"] == "member@example.com"
    assert by_type["codex"]["user_id"] == str(child_id)
    assert by_type["codex"]["user_email"] == "unknown"


# --------------------------------------------------------------------------- #
# get_sessions_with_agent_info / get_cost_by_user_agent (SDK-usable entry
# points — resolve visibility, then delegate; see review discussion on
# moving this out of the router so a plain SDK caller can use it too)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_sessions_with_agent_info_resolves_visibility_then_delegates(monkeypatch):
    user_id = uuid4()
    user = SimpleNamespace(id=user_id, tenant_id=None, email="me@example.com")

    async def fake_permitted(_user):
        return ["dataset-1"]

    async def fake_visible(_user):
        return [user_id]

    captured = {}

    async def fake_build_page(**kwargs):
        captured.update(kwargs)
        return {"sessions": [], "total": 0, "limit": 50, "offset": 0, "has_more": False}

    monkeypatch.setattr(agent_usage, "get_permitted_dataset_ids", fake_permitted)
    monkeypatch.setattr(agent_usage, "get_visible_user_ids", fake_visible)
    monkeypatch.setattr(agent_usage, "build_sessions_with_agent_info_page", fake_build_page)

    result = await agent_usage.get_sessions_with_agent_info(
        user=user,
        since=None,
        status_filter=None,
        limit=10,
        offset=5,
        order_by="last_activity_at",
        descending=True,
    )

    assert result["total"] == 0
    assert captured["user"] is user
    assert captured["permitted_dataset_ids"] == ["dataset-1"]
    assert captured["visible_user_ids"] == [user_id]
    assert captured["limit"] == 10
    assert captured["offset"] == 5


@pytest.mark.asyncio
async def test_get_cost_by_user_agent_resolves_visibility_then_delegates(monkeypatch):
    user_id = uuid4()
    user = SimpleNamespace(id=user_id, tenant_id=None, email="me@example.com")

    async def fake_permitted(_user):
        return ["dataset-1"]

    async def fake_visible(_user):
        return [user_id]

    captured = {}

    async def fake_compute(**kwargs):
        captured.update(kwargs)
        return [{"user_id": str(user_id)}]

    monkeypatch.setattr(agent_usage, "get_permitted_dataset_ids", fake_permitted)
    monkeypatch.setattr(agent_usage, "get_visible_user_ids", fake_visible)
    monkeypatch.setattr(agent_usage, "compute_cost_by_user_agent", fake_compute)

    result = await agent_usage.get_cost_by_user_agent(user=user, since=None)

    assert result == [{"user_id": str(user_id)}]
    assert captured["user"] is user
    assert captured["permitted_dataset_ids"] == ["dataset-1"]
    assert captured["visible_user_ids"] == [user_id]


@pytest.mark.asyncio
async def test_get_sessions_with_agent_info_defaults_user_when_omitted(monkeypatch):
    """Matches every other SDK-facing function in cognee.api.v1.agents.agents:
    a caller with no HTTP auth context can omit ``user`` entirely."""
    default_user = SimpleNamespace(id=uuid4(), tenant_id=None, email="default@example.com")

    async def fake_get_default_user():
        return default_user

    async def fake_permitted(_user):
        return []

    async def fake_visible(_user):
        return [default_user.id]

    captured = {}

    async def fake_build_page(**kwargs):
        captured.update(kwargs)
        return {"sessions": [], "total": 0, "limit": 50, "offset": 0, "has_more": False}

    monkeypatch.setattr(agent_usage, "get_default_user", fake_get_default_user)
    monkeypatch.setattr(agent_usage, "get_permitted_dataset_ids", fake_permitted)
    monkeypatch.setattr(agent_usage, "get_visible_user_ids", fake_visible)
    monkeypatch.setattr(agent_usage, "build_sessions_with_agent_info_page", fake_build_page)

    await agent_usage.get_sessions_with_agent_info(
        since=None,
        status_filter=None,
        limit=50,
        offset=0,
        order_by="last_activity_at",
        descending=True,
    )

    assert captured["user"] is default_user


@pytest.mark.asyncio
async def test_get_cost_by_user_agent_defaults_user_when_omitted(monkeypatch):
    default_user = SimpleNamespace(id=uuid4(), tenant_id=None, email="default@example.com")

    async def fake_get_default_user():
        return default_user

    async def fake_permitted(_user):
        return []

    async def fake_visible(_user):
        return [default_user.id]

    captured = {}

    async def fake_compute(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(agent_usage, "get_default_user", fake_get_default_user)
    monkeypatch.setattr(agent_usage, "get_permitted_dataset_ids", fake_permitted)
    monkeypatch.setattr(agent_usage, "get_visible_user_ids", fake_visible)
    monkeypatch.setattr(agent_usage, "compute_cost_by_user_agent", fake_compute)

    await agent_usage.get_cost_by_user_agent(since=None)

    assert captured["user"] is default_user
