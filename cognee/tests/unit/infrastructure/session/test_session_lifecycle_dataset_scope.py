from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.infrastructure.session.session_scope import get_storage_session_id
from cognee.modules.session_lifecycle import metrics
from cognee.modules.session_lifecycle.models import SessionModelUsage, SessionRecord


class _RelationalEngine:
    def __init__(self, engine):
        self.engine = engine
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    def get_async_session(self):
        return self._sessions()


@pytest_asyncio.fixture
async def lifecycle_engine(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(SessionRecord.metadata.create_all)
    wrapper = _RelationalEngine(engine)
    monkeypatch.setattr(metrics, "get_relational_engine", lambda: wrapper)
    yield wrapper
    await engine.dispose()


@pytest.mark.asyncio
async def test_same_public_session_isolated_per_dataset(lifecycle_engine):
    owner_id = uuid4()
    dataset_a = uuid4()
    dataset_b = uuid4()
    public_session_id = "shared-session"

    await metrics.ensure_and_touch_session(
        session_id=public_session_id,
        user_id=owner_id,
        dataset_id=dataset_a,
    )
    await metrics.ensure_and_touch_session(
        session_id=public_session_id,
        user_id=owner_id,
        dataset_id=dataset_b,
    )

    async with lifecycle_engine.get_async_session() as session:
        rows = list((await session.scalars(select(SessionRecord))).all())

    assert {row.session_id for row in rows} == {
        get_storage_session_id(public_session_id, dataset_a),
        get_storage_session_id(public_session_id, dataset_b),
    }
    assert {row.public_session_id for row in rows} == {public_session_id}

    row_a = await metrics.get_session_row(
        session_id=public_session_id,
        user_id=owner_id,
        dataset_id=dataset_a,
        permitted_dataset_ids=[dataset_a],
    )
    row_b = await metrics.get_session_row(
        session_id=public_session_id,
        user_id=owner_id,
        dataset_id=dataset_b,
        permitted_dataset_ids=[dataset_b],
    )
    assert row_a is not None and row_a.dataset_id == dataset_a
    assert row_b is not None and row_b.dataset_id == dataset_b


@pytest.mark.asyncio
async def test_delete_session_lifecycle_removes_only_exact_dataset_and_owner(lifecycle_engine):
    owner_id = uuid4()
    other_owner_id = uuid4()
    dataset_a = uuid4()
    dataset_b = uuid4()
    public_session_id = "shared-session"

    scopes = (
        (owner_id, dataset_a),
        (owner_id, dataset_b),
        (owner_id, None),
        (other_owner_id, dataset_a),
    )
    for scoped_owner_id, scoped_dataset_id in scopes:
        await metrics.ensure_and_touch_session(
            session_id=public_session_id,
            user_id=scoped_owner_id,
            dataset_id=scoped_dataset_id,
        )
        await metrics.accumulate_usage(
            session_id=public_session_id,
            user_id=scoped_owner_id,
            dataset_id=scoped_dataset_id,
            model="test-model",
            tokens_in=1,
        )

    assert await metrics.delete_session_lifecycle(
        session_id=public_session_id,
        user_id=str(owner_id),
        dataset_id=str(dataset_a),
    )

    deleted_storage_id = get_storage_session_id(public_session_id, dataset_a)
    expected_remaining = {
        (get_storage_session_id(public_session_id, dataset_b), owner_id),
        (public_session_id, owner_id),
        (deleted_storage_id, other_owner_id),
    }
    async with lifecycle_engine.get_async_session() as session:
        records = set(
            (await session.execute(select(SessionRecord.session_id, SessionRecord.user_id))).all()
        )
        usage = set(
            (
                await session.execute(
                    select(SessionModelUsage.session_id, SessionModelUsage.user_id)
                )
            ).all()
        )

    assert records == expected_remaining
    assert usage == expected_remaining
    assert not await metrics.delete_session_lifecycle(
        session_id=public_session_id,
        user_id=owner_id,
        dataset_id=dataset_a,
    )


@pytest.mark.asyncio
async def test_dataset_grant_lists_only_marked_requested_scope(lifecycle_engine):
    owner_id = uuid4()
    caller_id = uuid4()
    dataset_a = uuid4()
    dataset_b = uuid4()
    public_session_id = "shared-session"

    await metrics.ensure_and_touch_session(
        session_id=public_session_id,
        user_id=owner_id,
        dataset_id=dataset_a,
    )
    await metrics.ensure_and_touch_session(
        session_id=public_session_id,
        user_id=owner_id,
        dataset_id=dataset_b,
    )

    page = await metrics.list_session_rows(
        user_id=caller_id,
        permitted_dataset_ids=[dataset_a],
    )
    assert len(page.sessions) == 1
    assert page.sessions[0].record.dataset_id == dataset_a

    assert (
        await metrics.get_session_row(
            session_id=public_session_id,
            user_id=caller_id,
            permitted_dataset_ids=[dataset_a],
            dataset_id=dataset_a,
        )
        is not None
    )
    assert (
        await metrics.get_session_row(
            session_id=public_session_id,
            user_id=caller_id,
            permitted_dataset_ids=[dataset_a],
            dataset_id=dataset_b,
        )
        is None
    )


@pytest.mark.asyncio
async def test_owner_and_parent_cannot_bypass_scoped_dataset_permission(lifecycle_engine):
    parent_id = uuid4()
    child_id = uuid4()
    dataset_id = uuid4()
    other_dataset_id = uuid4()

    await metrics.ensure_and_touch_session(
        session_id="scoped-session",
        user_id=child_id,
        dataset_id=dataset_id,
    )
    await metrics.ensure_and_touch_session(
        session_id="legacy-session",
        user_id=child_id,
    )

    page_without_permission = await metrics.list_session_rows(
        user_ids=[parent_id, child_id],
        permitted_dataset_ids=[],
    )
    assert [row.record.session_id for row in page_without_permission.sessions] == ["legacy-session"]
    assert (
        await metrics.get_session_row(
            session_id="scoped-session",
            user_id=parent_id,
            user_ids=[parent_id, child_id],
            dataset_id=dataset_id,
            permitted_dataset_ids=[],
        )
        is None
    )
    assert (
        await metrics.get_session_row(
            session_id="scoped-session",
            user_id=parent_id,
            user_ids=[parent_id, child_id],
            dataset_id=dataset_id,
            permitted_dataset_ids=[other_dataset_id],
        )
        is None
    )

    legacy_row = await metrics.get_session_row(
        session_id="legacy-session",
        user_id=parent_id,
        user_ids=[parent_id, child_id],
    )
    assert legacy_row is not None and legacy_row.user_id == child_id

    scoped_row = await metrics.get_session_row(
        session_id="scoped-session",
        user_id=parent_id,
        user_ids=[parent_id, child_id],
        dataset_id=dataset_id,
        permitted_dataset_ids=[dataset_id],
    )
    assert scoped_row is not None and scoped_row.user_id == child_id

    permitted_page = await metrics.list_session_rows(
        user_ids=[parent_id, child_id],
        permitted_dataset_ids=[dataset_id],
    )
    assert {row.record.to_dict()["session_id"] for row in permitted_page.sessions} == {
        "legacy-session",
        "scoped-session",
    }


@pytest.mark.asyncio
async def test_legacy_sticky_dataset_row_remains_owner_only(lifecycle_engine):
    owner_id = uuid4()
    caller_id = uuid4()
    dataset_id = uuid4()
    now = datetime.now(timezone.utc)

    async with lifecycle_engine.get_async_session() as session:
        session.add(
            SessionRecord(
                session_id="legacy-session",
                public_session_id=None,
                user_id=owner_id,
                dataset_id=dataset_id,
                status="running",
                started_at=now,
                last_activity_at=now,
            )
        )
        await session.commit()

    page = await metrics.list_session_rows(
        user_id=caller_id,
        permitted_dataset_ids=[dataset_id],
    )
    assert page.sessions == []
    assert (
        await metrics.get_session_row(
            session_id="legacy-session",
            user_id=caller_id,
            permitted_dataset_ids=[dataset_id],
            dataset_id=dataset_id,
        )
        is None
    )

    owner_row = await metrics.get_session_row(
        session_id="legacy-session",
        user_id=owner_id,
    )
    assert owner_row is not None
    assert owner_row.to_dict()["dataset_id"] is None


@pytest.mark.asyncio
async def test_owner_scope_inference_requires_one_unambiguous_dataset(lifecycle_engine):
    owner_id = uuid4()
    dataset_a = uuid4()
    dataset_b = uuid4()

    await metrics.ensure_and_touch_session(
        session_id="unique", user_id=owner_id, dataset_id=dataset_a
    )
    assert (
        await metrics.get_owned_session_dataset_id(session_id="unique", user_id=owner_id)
        == dataset_a
    )

    await metrics.ensure_and_touch_session(
        session_id="ambiguous", user_id=owner_id, dataset_id=dataset_a
    )
    await metrics.ensure_and_touch_session(
        session_id="ambiguous", user_id=owner_id, dataset_id=dataset_b
    )
    with pytest.raises(ValueError, match="multiple datasets"):
        await metrics.get_owned_session_dataset_id(session_id="ambiguous", user_id=owner_id)


@pytest.mark.asyncio
async def test_cross_owner_dataset_collision_fails_closed_without_owner_selector(lifecycle_engine):
    caller_id = uuid4()
    dataset_id = uuid4()
    owner_a = uuid4()
    owner_b = uuid4()

    for owner_id in (owner_a, owner_b):
        await metrics.ensure_and_touch_session(
            session_id="shared", user_id=owner_id, dataset_id=dataset_id
        )

    assert (
        await metrics.get_session_row(
            session_id="shared",
            user_id=caller_id,
            permitted_dataset_ids=[dataset_id],
            dataset_id=dataset_id,
        )
        is None
    )
    selected = await metrics.get_session_row(
        session_id="shared",
        user_id=caller_id,
        permitted_dataset_ids=[dataset_id],
        dataset_id=dataset_id,
        owner_user_id=owner_b,
    )
    assert selected is not None and selected.user_id == owner_b
