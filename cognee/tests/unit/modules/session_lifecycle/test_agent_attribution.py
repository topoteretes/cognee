"""Tests for agent↔session attribution on SessionRecord.

``agent_id`` is stamped fill-if-null: the first agent connection to claim a
session wins, re-registration is a no-op, and a second agent cannot
overwrite the attribution.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.session_lifecycle.metrics import (
    ensure_and_touch_session,
    set_session_agent,
)
from cognee.modules.session_lifecycle.models import SessionRecord


@pytest_asyncio.fixture
async def session_table():
    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)
    created: list[tuple[str, object]] = []

    yield created

    async with engine.get_async_session() as session:
        for session_id, user_id in created:
            row = await session.get(SessionRecord, (session_id, user_id))
            if row:
                await session.delete(row)
        await session.commit()


async def _get_agent_id(session_id, user_id):
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        row = await session.get(SessionRecord, (session_id, user_id))
        return row.agent_id if row else None


@pytest.mark.asyncio
async def test_ensure_session_carries_agent_id(session_table):
    user_id = uuid4()
    session_table.append(("s-ensure", user_id))

    await ensure_and_touch_session(session_id="s-ensure", user_id=user_id, agent_id="copilot-abc")

    assert await _get_agent_id("s-ensure", user_id) == "copilot-abc"


@pytest.mark.asyncio
async def test_ensure_session_backfills_null_agent_id(session_table):
    user_id = uuid4()
    session_table.append(("s-backfill", user_id))

    await ensure_and_touch_session(session_id="s-backfill", user_id=user_id)
    assert await _get_agent_id("s-backfill", user_id) is None

    await ensure_and_touch_session(session_id="s-backfill", user_id=user_id, agent_id="copilot-abc")
    assert await _get_agent_id("s-backfill", user_id) == "copilot-abc"


@pytest.mark.asyncio
async def test_ensure_session_does_not_overwrite_agent_id(session_table):
    user_id = uuid4()
    session_table.append(("s-keep", user_id))

    await ensure_and_touch_session(session_id="s-keep", user_id=user_id, agent_id="first-agent")
    await ensure_and_touch_session(session_id="s-keep", user_id=user_id, agent_id="second-agent")

    assert await _get_agent_id("s-keep", user_id) == "first-agent"


@pytest.mark.asyncio
async def test_set_session_agent_fill_if_null(session_table):
    user_id = uuid4()
    session_table.append(("s-claim", user_id))

    await ensure_and_touch_session(session_id="s-claim", user_id=user_id)
    await set_session_agent(session_id="s-claim", user_id=user_id, agent_id="claimer-1")
    assert await _get_agent_id("s-claim", user_id) == "claimer-1"

    # A second agent cannot steal the attribution.
    await set_session_agent(session_id="s-claim", user_id=user_id, agent_id="claimer-2")
    assert await _get_agent_id("s-claim", user_id) == "claimer-1"


@pytest.mark.asyncio
async def test_set_session_agent_missing_row_is_noop(session_table):
    user_id = uuid4()
    # No ensure — the row does not exist; must not raise or create a row.
    await set_session_agent(session_id="s-ghost", user_id=user_id, agent_id="claimer")
    assert await _get_agent_id("s-ghost", user_id) is None


@pytest.mark.asyncio
async def test_to_dict_exposes_agent_id(session_table):
    user_id = uuid4()
    session_table.append(("s-dict", user_id))
    engine = get_relational_engine()
    now = datetime.now(timezone.utc)
    async with engine.get_async_session() as session:
        session.add(
            SessionRecord(
                session_id="s-dict",
                user_id=user_id,
                status="running",
                started_at=now,
                last_activity_at=now,
                agent_id="copilot-abc",
            )
        )
        await session.commit()

    async with engine.get_async_session() as session:
        row = await session.get(SessionRecord, ("s-dict", user_id))
        assert row.to_dict()["agent_id"] == "copilot-abc"
