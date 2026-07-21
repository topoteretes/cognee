"""Tests for dataset-scoped sessions.

A SessionManager can be bound to a dataset (explicitly or via the
current_dataset_id context variable). The binding:

- derives a per-dataset default session ID when session_id is omitted, so
  two datasets can never mix turns in one shared "default_session";
- attributes lifecycle rows (session_records.dataset_id) to the dataset.

Explicit session IDs are stored unchanged — existing sessions keep working.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import select

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.modules.session_lifecycle.metrics import (
    delete_session_lifecycle,
    record_session_activity,
)
from cognee.modules.session_lifecycle.models import SessionModelUsage, SessionRecord
from cognee.infrastructure.databases.relational import get_relational_engine


class TestDatasetBinding:
    """Default-session derivation from the manager's dataset binding."""

    def test_explicit_dataset_derives_default_session_id(self):
        dataset_id = uuid4()
        manager = SessionManager(cache_engine=None, dataset_id=dataset_id)
        assert manager._resolve_session_id(None) == f"default_session_{dataset_id}"

    def test_no_dataset_uses_plain_default(self):
        manager = SessionManager(cache_engine=None)
        assert manager._resolve_session_id(None) == "default_session"

    def test_explicit_session_id_unchanged(self):
        manager = SessionManager(cache_engine=None, dataset_id=uuid4())
        assert manager._resolve_session_id("my_session") == "my_session"

    def test_inherits_current_dataset_id_context(self):
        dataset_id = uuid4()
        token = current_dataset_id.set(str(dataset_id))
        try:
            manager = SessionManager(cache_engine=None)
        finally:
            current_dataset_id.reset(token)
        assert manager.dataset_id == str(dataset_id)
        assert manager._resolve_session_id(None) == f"default_session_{dataset_id}"

    def test_explicit_dataset_overrides_context(self):
        explicit_id = uuid4()
        token = current_dataset_id.set(str(uuid4()))
        try:
            manager = SessionManager(cache_engine=None, dataset_id=explicit_id)
        finally:
            current_dataset_id.reset(token)
        assert manager.dataset_id == str(explicit_id)


class TestActivityAttribution:
    """Writes report the manager's dataset to the lifecycle heartbeat."""

    @pytest.mark.asyncio
    async def test_add_qa_records_activity_with_dataset(self):
        dataset_id = uuid4()
        manager = SessionManager(cache_engine=AsyncMock(), dataset_id=dataset_id)

        with (
            patch(
                "cognee.infrastructure.session.session_manager.index_session_qa",
                new_callable=AsyncMock,
            ),
            patch(
                "cognee.infrastructure.session.session_manager.record_session_activity",
                new_callable=AsyncMock,
            ) as record_mock,
        ):
            await manager.add_qa(user_id="u1", question="q", context="", answer="a")

        record_mock.assert_awaited_once_with(
            "u1", f"default_session_{dataset_id}", dataset_id=str(dataset_id)
        )


class TestDeleteSessionLifecycle:
    """delete_session removes the lifecycle rows alongside the cache content."""

    @pytest.mark.asyncio
    async def test_delete_session_deletes_lifecycle_rows(self):
        cache = AsyncMock()
        cache.delete_session.return_value = True
        manager = SessionManager(cache_engine=cache)

        with (
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_qa_vectors",
                new_callable=AsyncMock,
            ),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_lifecycle",
                new_callable=AsyncMock,
                return_value=True,
            ) as lifecycle_mock,
        ):
            assert await manager.delete_session(user_id="u1", session_id="s1") is True

        lifecycle_mock.assert_awaited_once_with(session_id="s1", user_id="u1")

    @pytest.mark.asyncio
    async def test_delete_session_true_when_only_lifecycle_row_existed(self):
        """An orphan lifecycle row (no cache content) still reports deletion."""
        cache = AsyncMock()
        cache.delete_session.return_value = False
        manager = SessionManager(cache_engine=cache)

        with (
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_qa_vectors",
                new_callable=AsyncMock,
            ),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_lifecycle",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            assert await manager.delete_session(user_id="u1", session_id="s1") is True


@pytest.mark.asyncio
async def test_record_session_activity_fills_dataset_id():
    """The heartbeat upsert attributes the session row to the dataset."""
    user_id = uuid4()
    dataset_id = uuid4()
    session_id = f"scope-test-{uuid4()}"

    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)

    await record_session_activity(str(user_id), session_id, dataset_id=str(dataset_id))

    async with engine.get_async_session() as session:
        row = (
            await session.execute(
                select(SessionRecord).where(SessionRecord.session_id == session_id)
            )
        ).scalar_one()
    assert row.user_id == user_id
    assert row.dataset_id == dataset_id


@pytest.mark.asyncio
async def test_delete_session_lifecycle_removes_rows():
    user_id = uuid4()
    session_id = f"scope-test-{uuid4()}"
    now = datetime.now(timezone.utc)

    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)

    async with engine.get_async_session() as session:
        session.add(
            SessionRecord(
                session_id=session_id,
                user_id=user_id,
                status="running",
                started_at=now,
                last_activity_at=now,
            )
        )
        session.add(
            SessionModelUsage(
                session_id=session_id,
                user_id=user_id,
                model="gpt-test",
                tokens_in=1,
                tokens_out=1,
                cost_usd=0.0,
            )
        )
        await session.commit()

    assert await delete_session_lifecycle(session_id=session_id, user_id=user_id) is True

    async with engine.get_async_session() as session:
        record = (
            await session.execute(
                select(SessionRecord).where(SessionRecord.session_id == session_id)
            )
        ).scalar_one_or_none()
        usage = (
            await session.execute(
                select(SessionModelUsage).where(SessionModelUsage.session_id == session_id)
            )
        ).scalar_one_or_none()
    assert record is None
    assert usage is None


@pytest.mark.asyncio
async def test_delete_session_lifecycle_missing_row_returns_false():
    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)

    assert await delete_session_lifecycle(session_id=f"missing-{uuid4()}", user_id=uuid4()) is False


@pytest.mark.asyncio
async def test_delete_session_lifecycle_invalid_user_returns_false():
    assert await delete_session_lifecycle(session_id="s1", user_id="not-a-uuid") is False
