"""Tests for parent-user visibility of child-agent sessions.

Parent users should see sessions created by their child agents via the
``user_ids`` parameter on ``list_session_rows`` and ``get_session_row``.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.modules.session_lifecycle.metrics import (
    list_session_rows,
    get_session_row,
)
from cognee.modules.session_lifecycle.models import SessionRecord
from cognee.infrastructure.databases.relational import get_relational_engine


@pytest_asyncio.fixture
async def parent_and_agent_sessions():
    """Create sessions for a parent user and a child agent, return their IDs."""
    parent_id = uuid4()
    agent_id = uuid4()
    other_id = uuid4()
    now = datetime.now(timezone.utc)

    engine = get_relational_engine()

    # Ensure session_records table exists
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)

    async with engine.get_async_session() as session:
        session.add(
            SessionRecord(
                session_id="parent-session",
                user_id=parent_id,
                status="completed",
                started_at=now,
                last_activity_at=now,
            )
        )
        session.add(
            SessionRecord(
                session_id="agent-session",
                user_id=agent_id,
                status="running",
                started_at=now,
                last_activity_at=now,
            )
        )
        session.add(
            SessionRecord(
                session_id="other-session",
                user_id=other_id,
                status="running",
                started_at=now,
                last_activity_at=now,
            )
        )
        await session.commit()

    yield {"parent_id": parent_id, "agent_id": agent_id, "other_id": other_id}

    # Cleanup
    async with engine.get_async_session() as session:
        for sid in ("parent-session", "agent-session", "other-session"):
            row = await session.get(SessionRecord, (sid, parent_id))
            if row:
                await session.delete(row)
            row = await session.get(SessionRecord, (sid, agent_id))
            if row:
                await session.delete(row)
            row = await session.get(SessionRecord, (sid, other_id))
            if row:
                await session.delete(row)
        await session.commit()


class TestListSessionRowsUserIds:
    """list_session_rows with user_ids includes child agent sessions."""

    @pytest.mark.asyncio
    async def test_single_user_id_only_sees_own(self, parent_and_agent_sessions):
        ids = parent_and_agent_sessions
        page = await list_session_rows(user_id=ids["parent_id"])
        session_ids = {r.record.session_id for r in page.sessions}
        assert "parent-session" in session_ids
        assert "agent-session" not in session_ids
        assert "other-session" not in session_ids

    @pytest.mark.asyncio
    async def test_user_ids_includes_agent_sessions(self, parent_and_agent_sessions):
        ids = parent_and_agent_sessions
        page = await list_session_rows(user_ids=[ids["parent_id"], ids["agent_id"]])
        session_ids = {r.record.session_id for r in page.sessions}
        assert "parent-session" in session_ids
        assert "agent-session" in session_ids
        assert "other-session" not in session_ids

    @pytest.mark.asyncio
    async def test_user_ids_excludes_unrelated_users(self, parent_and_agent_sessions):
        ids = parent_and_agent_sessions
        page = await list_session_rows(user_ids=[ids["parent_id"], ids["agent_id"]])
        session_ids = {r.record.session_id for r in page.sessions}
        assert "other-session" not in session_ids


class TestGetSessionRowUserIds:
    """get_session_row with user_ids can find agent-owned sessions."""

    @pytest.mark.asyncio
    async def test_parent_cannot_see_agent_session_without_user_ids(
        self, parent_and_agent_sessions
    ):
        ids = parent_and_agent_sessions
        row = await get_session_row(
            session_id="agent-session",
            user_id=ids["parent_id"],
        )
        assert row is None

    @pytest.mark.asyncio
    async def test_parent_can_see_agent_session_with_user_ids(self, parent_and_agent_sessions):
        ids = parent_and_agent_sessions
        row = await get_session_row(
            session_id="agent-session",
            user_id=ids["parent_id"],
            user_ids=[ids["parent_id"], ids["agent_id"]],
        )
        assert row is not None
        assert row.session_id == "agent-session"
        assert row.user_id == ids["agent_id"]

    @pytest.mark.asyncio
    async def test_user_ids_does_not_leak_unrelated_sessions(self, parent_and_agent_sessions):
        ids = parent_and_agent_sessions
        row = await get_session_row(
            session_id="other-session",
            user_id=ids["parent_id"],
            user_ids=[ids["parent_id"], ids["agent_id"]],
        )
        assert row is None
