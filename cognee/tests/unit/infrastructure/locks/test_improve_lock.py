"""The cross-worker improve lock on ``session_records`` (SDK-593).

Runs against a real temporary SQLite database holding only the session_records
table: the lock is one conditional UPDATE per transition, so the row's own
atomicity is what the tests exercise — no in-process registry is involved.
"""

import importlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.infrastructure.locks import (
    ImproveLockRelease,
    release_improve_lock,
    request_improve_rerun,
    try_acquire_improve_lock,
)
from cognee.modules.session_lifecycle.models import SessionRecord

relational_pkg = importlib.import_module("cognee.infrastructure.databases.relational")
metrics_mod = importlib.import_module("cognee.modules.session_lifecycle.metrics")


@pytest_asyncio.fixture
async def lock_engine(tmp_path, monkeypatch):
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="improve_lock_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )
    async with engine.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all, tables=[SessionRecord.__table__])

    # The lock module imports the getter lazily from the package; metrics binds it at import.
    monkeypatch.setattr(relational_pkg, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(metrics_mod, "get_relational_engine", lambda: engine)
    yield engine
    await engine.engine.dispose()


async def _row(engine, session_id, user_id) -> SessionRecord | None:
    async with engine.get_async_session() as session:
        return (
            await session.execute(
                select(SessionRecord).where(
                    SessionRecord.session_id == session_id, SessionRecord.user_id == user_id
                )
            )
        ).scalar_one_or_none()


async def _seed(engine, session_id, user_id, **overrides):
    now = datetime.now(timezone.utc)
    async with engine.get_async_session() as session:
        session.add(
            SessionRecord(
                session_id=session_id,
                user_id=user_id,
                status="running",
                started_at=now,
                last_activity_at=now,
                **overrides,
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_acquire_then_second_claim_is_busy_then_release_frees(lock_engine):
    user_id = uuid4()
    await _seed(lock_engine, "s1", user_id)

    token = await try_acquire_improve_lock("s1", user_id)
    assert token

    assert await try_acquire_improve_lock("s1", user_id) is None

    row = await _row(lock_engine, "s1", user_id)
    assert row.improve_lock_token == token
    assert row.improve_lock_acquired_at is not None

    assert await release_improve_lock("s1", user_id, token) is ImproveLockRelease.RELEASED
    row = await _row(lock_engine, "s1", user_id)
    assert row.improve_lock_token is None
    assert row.improve_lock_acquired_at is None

    assert await try_acquire_improve_lock("s1", user_id)


@pytest.mark.asyncio
async def test_release_with_a_foreign_token_is_a_noop(lock_engine):
    user_id = uuid4()
    await _seed(lock_engine, "s1", user_id)
    token = await try_acquire_improve_lock("s1", user_id)

    assert await release_improve_lock("s1", user_id, "not-the-holder") is ImproveLockRelease.LOST

    assert (await _row(lock_engine, "s1", user_id)).improve_lock_token == token


@pytest.mark.asyncio
async def test_lock_is_scoped_per_user_and_per_session(lock_engine):
    user_a, user_b = uuid4(), uuid4()
    await _seed(lock_engine, "s1", user_a)
    await _seed(lock_engine, "s1", user_b)
    await _seed(lock_engine, "s2", user_a)

    assert await try_acquire_improve_lock("s1", user_a)
    assert await try_acquire_improve_lock("s1", user_b)
    assert await try_acquire_improve_lock("s2", user_a)


@pytest.mark.asyncio
async def test_missing_session_row_is_created_before_locking(lock_engine):
    """SDK-only sessions never went through remember(); the lock creates the row."""
    user_id = uuid4()
    assert await _row(lock_engine, "fresh", user_id) is None

    token = await try_acquire_improve_lock("fresh", user_id)

    row = await _row(lock_engine, "fresh", user_id)
    assert token and row is not None and row.improve_lock_token == token
    assert row.status == "running"


@pytest.mark.asyncio
async def test_expired_lock_is_taken_over(lock_engine, monkeypatch):
    """A hung or killed holder can no longer wedge a session: past the TTL the lock is free."""
    user_id = uuid4()
    monkeypatch.setenv("IMPROVE_LOCK_TTL_SECONDS", "60")
    stale = datetime.now(timezone.utc) - timedelta(seconds=120)
    await _seed(
        lock_engine,
        "s1",
        user_id,
        improve_lock_token="dead-worker",
        improve_lock_acquired_at=stale,
        improve_rerun_requested=True,
    )

    token = await try_acquire_improve_lock("s1", user_id)

    row = await _row(lock_engine, "s1", user_id)
    assert token and token != "dead-worker"
    assert row.improve_lock_token == token
    # The new holder starts a full pass, which satisfies the stale rerun request.
    assert row.improve_rerun_requested is False


@pytest.mark.asyncio
async def test_live_lock_within_ttl_stays_busy(lock_engine, monkeypatch):
    user_id = uuid4()
    monkeypatch.setenv("IMPROVE_LOCK_TTL_SECONDS", "3600")
    await _seed(
        lock_engine,
        "s1",
        user_id,
        improve_lock_token="other-worker",
        improve_lock_acquired_at=datetime.now(timezone.utc) - timedelta(seconds=120),
    )

    assert await try_acquire_improve_lock("s1", user_id) is None


@pytest.mark.asyncio
async def test_rerun_request_reaches_the_holder_exactly_once(lock_engine):
    user_id = uuid4()
    await _seed(lock_engine, "s1", user_id)
    token = await try_acquire_improve_lock("s1", user_id)

    status = await request_improve_rerun("s1", user_id)
    assert status.busy is True
    assert status.rerun_requested is True
    assert status.holder_age_seconds is not None and status.holder_age_seconds >= 0

    # A second busy caller only re-asserts the same flag.
    assert (await request_improve_rerun("s1", user_id)).busy is True

    # The holder's release consumes the request exactly once and keeps the lock...
    assert await release_improve_lock("s1", user_id, token) is ImproveLockRelease.RERUN
    row = await _row(lock_engine, "s1", user_id)
    assert row.improve_lock_token == token
    assert row.improve_rerun_requested is False
    # ...and the next release lets go.
    assert await release_improve_lock("s1", user_id, token) is ImproveLockRelease.RELEASED


@pytest.mark.asyncio
async def test_rerun_request_on_a_free_session_is_not_recorded(lock_engine):
    """The flag only makes sense while someone holds the lock; a free session gets none."""
    user_id = uuid4()
    await _seed(lock_engine, "s1", user_id)

    status = await request_improve_rerun("s1", user_id)

    assert status.busy is False
    assert (await _row(lock_engine, "s1", user_id)).improve_rerun_requested is False


@pytest.mark.asyncio
async def test_only_the_holder_can_consume_a_pending_rerun(lock_engine):
    user_id = uuid4()
    await _seed(lock_engine, "s1", user_id)
    token = await try_acquire_improve_lock("s1", user_id)
    await request_improve_rerun("s1", user_id)

    assert await release_improve_lock("s1", user_id, "someone-else") is ImproveLockRelease.LOST
    assert (await _row(lock_engine, "s1", user_id)).improve_rerun_requested is True
    assert await release_improve_lock("s1", user_id, token) is ImproveLockRelease.RERUN


@pytest.mark.asyncio
async def test_empty_identity_is_a_noop_lock(lock_engine):
    assert await try_acquire_improve_lock("", uuid4())
    assert await try_acquire_improve_lock("s1", None)
    assert (await request_improve_rerun("", uuid4())).busy is False
    assert await release_improve_lock("", uuid4(), "t") is ImproveLockRelease.RELEASED


@pytest.mark.asyncio
async def test_release_is_refused_while_a_rerun_is_pending(lock_engine):
    """The check-and-release is one statement: a request that landed cannot be lost."""
    user_id = uuid4()
    await _seed(lock_engine, "s1", user_id)
    token = await try_acquire_improve_lock("s1", user_id)
    await request_improve_rerun("s1", user_id)

    assert await release_improve_lock("s1", user_id, token) is ImproveLockRelease.RERUN
    row = await _row(lock_engine, "s1", user_id)
    assert row.improve_lock_token == token  # still ours
    assert row.improve_rerun_requested is False  # the request is now ours to fulfil

    # The holder runs its pass and can then let go.
    assert await release_improve_lock("s1", user_id, token) is ImproveLockRelease.RELEASED
    assert (await _row(lock_engine, "s1", user_id)).improve_lock_token is None


@pytest.mark.asyncio
async def test_forced_release_lets_go_but_keeps_the_pending_flag_for_the_next_holder(
    lock_engine,
):
    user_id = uuid4()
    await _seed(lock_engine, "s1", user_id)
    token = await try_acquire_improve_lock("s1", user_id)
    await request_improve_rerun("s1", user_id)

    assert (
        await release_improve_lock("s1", user_id, token, force=True) is ImproveLockRelease.RELEASED
    )
    row = await _row(lock_engine, "s1", user_id)
    assert row.improve_lock_token is None
    assert row.improve_rerun_requested is True  # left for the next acquirer

    # ...whose full pass satisfies it.
    assert await try_acquire_improve_lock("s1", user_id)
    assert (await _row(lock_engine, "s1", user_id)).improve_rerun_requested is False


@pytest.mark.asyncio
async def test_release_after_a_ttl_takeover_reports_released_without_touching_the_new_holder(
    lock_engine, monkeypatch
):
    user_id = uuid4()
    monkeypatch.setenv("IMPROVE_LOCK_TTL_SECONDS", "60")
    await _seed(
        lock_engine,
        "s1",
        user_id,
        improve_lock_token="old-holder",
        improve_lock_acquired_at=datetime.now(timezone.utc) - timedelta(seconds=120),
    )
    new_token = await try_acquire_improve_lock("s1", user_id)
    await request_improve_rerun("s1", user_id)  # pending against the NEW holder

    assert await release_improve_lock("s1", user_id, "old-holder") is ImproveLockRelease.LOST
    row = await _row(lock_engine, "s1", user_id)
    assert row.improve_lock_token == new_token
    assert row.improve_rerun_requested is True  # the new holder's request, untouched
