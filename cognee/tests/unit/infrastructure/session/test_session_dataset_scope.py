"""Tests for dataset-scoped sessions.

A SessionManager can be bound to a dataset (explicitly or via the
current_dataset_id context variable). The binding:

- derives a per-dataset default session ID when session_id is omitted, so
  two datasets can never mix turns in one shared "default_session";
- attributes lifecycle rows (session_records.dataset_id) to the dataset.

Explicit session IDs are stored unchanged — existing sessions keep working.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import select

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.session.session_manager import SessionManager
from cognee.modules.session_lifecycle.metrics import (
    claim_session_dataset,
    delete_session_lifecycle,
    get_session_dataset,
    get_session_dataset_id,
    record_session_activity,
)
from cognee.modules.session_lifecycle.models import SessionModelUsage, SessionRecord
from cognee.infrastructure.databases.relational import get_relational_engine


class TestDatasetBinding:
    """Default-session derivation from the manager's dataset binding."""

    @pytest.mark.asyncio
    async def test_explicit_dataset_derives_default_session_id(self):
        dataset_id = uuid4()
        manager = SessionManager(cache_engine=None, dataset_id=dataset_id)
        assert await manager._resolve_session_id(None) == f"default_session_{dataset_id}"

    @pytest.mark.asyncio
    async def test_no_dataset_and_no_default_uses_plain_default(self):
        """Without a resolvable main_dataset there is nothing to scope to."""
        manager = SessionManager(cache_engine=None)
        with patch.object(SessionManager, "_effective_dataset_id", AsyncMock(return_value=None)):
            assert await manager._resolve_session_id(None) == "default_session"

    @pytest.mark.asyncio
    async def test_explicit_session_id_unchanged(self):
        manager = SessionManager(cache_engine=None, dataset_id=uuid4())
        assert await manager._resolve_session_id("my_session") == "my_session"

    @pytest.mark.asyncio
    async def test_inherits_current_dataset_id_context(self):
        dataset_id = uuid4()
        token = current_dataset_id.set(str(dataset_id))
        try:
            manager = SessionManager(cache_engine=None)
        finally:
            current_dataset_id.reset(token)
        assert manager.dataset_id == str(dataset_id)
        assert await manager._resolve_session_id(None) == f"default_session_{dataset_id}"

    def test_explicit_dataset_overrides_context(self):
        explicit_id = uuid4()
        token = current_dataset_id.set(str(uuid4()))
        try:
            manager = SessionManager(cache_engine=None, dataset_id=explicit_id)
        finally:
            current_dataset_id.reset(token)
        assert manager.dataset_id == str(explicit_id)

    def test_non_uuid_dataset_context_is_ignored(self):
        """A name-valued context cannot scope a session; it must not be used as-is."""
        token = current_dataset_id.set("main_dataset")
        try:
            manager = SessionManager(cache_engine=None)
        finally:
            current_dataset_id.reset(token)
        assert manager.dataset_id is None


class TestDefaultDatasetFallback:
    """An omitted dataset falls back to main_dataset, like everywhere else in Cognee."""

    @pytest.mark.asyncio
    async def test_falls_back_to_main_dataset(self):
        user_id, dataset_id = uuid4(), uuid4()
        dataset = MagicMock()
        dataset.id = dataset_id
        manager = SessionManager(cache_engine=None)

        with patch(
            "cognee.modules.data.methods.get_datasets_by_name",
            AsyncMock(return_value=[dataset]),
        ) as by_name:
            resolved = await manager._resolve_session_id(None, str(user_id))
            # Second call must not re-query — the lookup is memoised per instance.
            await manager._resolve_session_id(None, str(user_id))

        assert resolved == f"default_session_{dataset_id}"
        by_name.assert_awaited_once_with(["main_dataset"], user_id)

    @pytest.mark.asyncio
    async def test_missing_main_dataset_does_not_create_one(self):
        """Reads must stay side-effect free before any dataset exists."""
        manager = SessionManager(cache_engine=None)
        with patch(
            "cognee.modules.data.methods.get_datasets_by_name",
            AsyncMock(return_value=[]),
        ):
            assert await manager._resolve_session_id(None, str(uuid4())) == "default_session"

    @pytest.mark.asyncio
    async def test_explicit_dataset_skips_the_fallback(self):
        manager = SessionManager(cache_engine=None, dataset_id=uuid4())
        with patch(
            "cognee.modules.data.methods.get_datasets_by_name",
            AsyncMock(return_value=[]),
        ) as by_name:
            await manager._resolve_session_id(None, str(uuid4()))
        by_name.assert_not_awaited()


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


async def _create_attributed_session(dataset_owner_id=None):
    """Insert a Dataset and a SessionRecord attributed to it; return the ids."""
    from cognee.modules.data.models import Dataset

    user_id = uuid4()
    dataset_id = uuid4()
    owner_id = dataset_owner_id or user_id
    session_id = f"scope-test-{uuid4()}"
    now = datetime.now(timezone.utc)

    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(SessionRecord.metadata.create_all)

    async with engine.get_async_session() as session:
        session.add(Dataset(id=dataset_id, name="scope_ds", owner_id=owner_id))
        session.add(
            SessionRecord(
                session_id=session_id,
                user_id=user_id,
                dataset_id=dataset_id,
                status="running",
                started_at=now,
                last_activity_at=now,
            )
        )
        await session.commit()
    return user_id, dataset_id, owner_id, session_id


@pytest.mark.asyncio
async def test_get_session_dataset_returns_dataset_and_owner():
    dataset_owner = uuid4()
    user_id, dataset_id, owner_id, session_id = await _create_attributed_session(dataset_owner)

    resolved = await get_session_dataset(session_id=session_id, user_id=user_id)
    assert resolved == (dataset_id, dataset_owner)


@pytest.mark.asyncio
async def test_get_session_dataset_none_without_attribution():
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
        await session.commit()

    assert await get_session_dataset(session_id=session_id, user_id=user_id) is None


@pytest.mark.asyncio
async def test_get_session_dataset_invalid_user_returns_none():
    assert await get_session_dataset(session_id="s1", user_id="not-a-uuid") is None


class TestScopedVectorCleanup:
    """Vector cleanup runs in the session's attributed dataset store AND the ambient store."""

    @staticmethod
    def _db_context_mock():
        context = MagicMock()
        context.return_value.__aenter__ = AsyncMock(return_value=None)
        context.return_value.__aexit__ = AsyncMock(return_value=False)
        return context

    @pytest.mark.asyncio
    async def test_delete_session_cleans_dataset_store(self):
        dataset_id, owner_id = uuid4(), uuid4()
        cache = AsyncMock()
        cache.delete_session.return_value = True
        manager = SessionManager(cache_engine=cache)
        db_context = self._db_context_mock()
        call_order = []

        with (
            patch(
                "cognee.infrastructure.session.session_manager.get_session_dataset",
                new_callable=AsyncMock,
                side_effect=lambda **kw: call_order.append("resolve") or (dataset_id, owner_id),
            ),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_lifecycle",
                new_callable=AsyncMock,
                side_effect=lambda **kw: call_order.append("lifecycle") or True,
            ),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_qa_vectors",
                new_callable=AsyncMock,
            ) as vectors_mock,
            patch(
                "cognee.context_global_variables.set_database_global_context_variables",
                db_context,
            ),
        ):
            assert await manager.delete_session(user_id="u1", session_id="s1") is True

        # Attribution must be read before the lifecycle row is deleted.
        assert call_order == ["resolve", "lifecycle"]
        db_context.assert_called_once_with(dataset_id, owner_id)
        # Once inside the dataset store, once in the ambient store.
        assert vectors_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_delete_session_without_attribution_uses_ambient_store_only(self):
        cache = AsyncMock()
        cache.delete_session.return_value = True
        manager = SessionManager(cache_engine=cache)
        db_context = self._db_context_mock()

        with (
            patch(
                "cognee.infrastructure.session.session_manager.get_session_dataset",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_lifecycle",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_qa_vectors",
                new_callable=AsyncMock,
            ) as vectors_mock,
            patch(
                "cognee.context_global_variables.set_database_global_context_variables",
                db_context,
            ),
        ):
            assert await manager.delete_session(user_id="u1", session_id="s1") is True

        db_context.assert_not_called()
        assert vectors_mock.await_count == 1

    @pytest.mark.asyncio
    async def test_delete_qa_cleans_dataset_store(self):
        dataset_id, owner_id = uuid4(), uuid4()
        cache = AsyncMock()
        cache.delete_qa_entry.return_value = True
        manager = SessionManager(cache_engine=cache)
        db_context = self._db_context_mock()

        with (
            patch(
                "cognee.infrastructure.session.session_manager.get_session_dataset",
                new_callable=AsyncMock,
                return_value=(dataset_id, owner_id),
            ),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_qa_vector",
                new_callable=AsyncMock,
            ) as vector_mock,
            patch(
                "cognee.context_global_variables.set_database_global_context_variables",
                db_context,
            ),
        ):
            assert await manager.delete_qa(user_id="u1", session_id="s1", qa_id="q1") is True

        db_context.assert_called_once_with(dataset_id, owner_id)
        assert vector_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_delete_session_scoped_cleanup_failure_still_cleans_ambient(self):
        """A broken dataset context must not block the ambient cleanup."""
        cache = AsyncMock()
        cache.delete_session.return_value = True
        manager = SessionManager(cache_engine=cache)
        db_context = MagicMock(side_effect=RuntimeError("dataset database is gone"))

        with (
            patch(
                "cognee.infrastructure.session.session_manager.get_session_dataset",
                new_callable=AsyncMock,
                return_value=(uuid4(), uuid4()),
            ),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_lifecycle",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "cognee.infrastructure.session.session_manager.delete_session_qa_vectors",
                new_callable=AsyncMock,
            ) as vectors_mock,
            patch(
                "cognee.context_global_variables.set_database_global_context_variables",
                db_context,
            ),
        ):
            assert await manager.delete_session(user_id="u1", session_id="s1") is True

        assert vectors_mock.await_count == 1


class TestOneDatasetBinding:
    """Sessions live in exactly one dataset: writes must match the binding."""

    @pytest.mark.asyncio
    async def test_same_dataset_passes(self):
        from cognee.modules.session_lifecycle.metrics import check_session_dataset_binding

        user_id, dataset_id, _, session_id = await _create_attributed_session()

        await check_session_dataset_binding(
            session_id=session_id, user_id=user_id, dataset_id=dataset_id
        )

    @pytest.mark.asyncio
    async def test_other_dataset_raises(self):
        from cognee.modules.session_lifecycle.exceptions import SessionDatasetMismatchError
        from cognee.modules.session_lifecycle.metrics import check_session_dataset_binding

        user_id, dataset_id, _, session_id = await _create_attributed_session()

        with pytest.raises(SessionDatasetMismatchError):
            await check_session_dataset_binding(
                session_id=session_id, user_id=user_id, dataset_id=uuid4()
            )

    @pytest.mark.asyncio
    async def test_unbound_session_passes_for_any_dataset(self):
        from cognee.modules.session_lifecycle.metrics import (
            check_session_dataset_binding,
            record_session_activity,
        )

        user_id = uuid4()
        session_id = f"scope-test-{uuid4()}"
        engine = get_relational_engine()
        async with engine.engine.begin() as conn:
            await conn.run_sync(SessionRecord.metadata.create_all)
        await record_session_activity(str(user_id), session_id)

        await check_session_dataset_binding(
            session_id=session_id, user_id=user_id, dataset_id=uuid4()
        )

    @pytest.mark.asyncio
    async def test_no_dataset_context_is_a_noop(self):
        from cognee.modules.session_lifecycle.metrics import check_session_dataset_binding

        await check_session_dataset_binding(session_id="s1", user_id=uuid4(), dataset_id=None)


class TestBindingEnforcementWiring:
    """The one-dataset guard is wired into the real write paths, not just the helper."""

    @pytest.mark.asyncio
    async def test_add_qa_raises_for_wrong_dataset_context(self):
        from cognee.modules.session_lifecycle.exceptions import SessionDatasetMismatchError

        user_id, _, _, session_id = await _create_attributed_session()
        manager = SessionManager(cache_engine=MagicMock(), dataset_id=uuid4())

        with pytest.raises(SessionDatasetMismatchError):
            await manager.add_qa(
                user_id=str(user_id), session_id=session_id, question="q", context="", answer="a"
            )

    @pytest.mark.asyncio
    async def test_delete_sessions_for_dataset_removes_sessions(self):
        import importlib

        from cognee.modules.session_lifecycle.metrics import delete_sessions_for_dataset

        user_id, dataset_id, _, session_id = await _create_attributed_session()

        manager = MagicMock()
        manager.delete_session = AsyncMock(return_value=True)
        # patch.object on the importlib-resolved module: the package __init__
        # re-exports a function named like the submodule, so a string target
        # resolves to the function (import-order dependent) and fails in CI.
        gsm_module = importlib.import_module("cognee.infrastructure.session.get_session_manager")
        with patch.object(gsm_module, "get_session_manager", return_value=manager):
            await delete_sessions_for_dataset(dataset_id)

        # Cache-side deletion was requested for the session's owner...
        manager.delete_session.assert_awaited_once_with(user_id=str(user_id), session_id=session_id)
        # ...and the lifecycle row is gone even though the manager was a stub
        # (the explicit fallback covers cache-less deployments).
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            row = (
                await session.execute(
                    select(SessionRecord).where(SessionRecord.session_id == session_id)
                )
            ).scalar_one_or_none()
        assert row is None

    @pytest.mark.asyncio
    async def test_run_session_turn_checks_binding_before_answer(self):
        from cognee.modules.session_lifecycle.exceptions import SessionDatasetMismatchError

        user_id, _, _, session_id = await _create_attributed_session()
        manager = SessionManager(cache_engine=MagicMock(), dataset_id=uuid4())

        with (
            patch.object(SessionManager, "is_session_available_for_completion", return_value=True),
            patch(
                "cognee.infrastructure.session.session_manager.generate_session_answer",
                AsyncMock(),
            ) as generate_answer,
        ):
            with pytest.raises(SessionDatasetMismatchError):
                await manager._run_session_turn(
                    user_id=str(user_id),
                    session_id=session_id,
                    query="q",
                    context="",
                    user_prompt_path="",
                    system_prompt_path="",
                )

        generate_answer.assert_not_awaited()


class TestAtomicClaim:
    """The bind is atomic: concurrent *first* writes cannot all win.

    This is the property a bare check-then-write cannot provide. On the search
    path the gap between "check" and "bind" spans a whole LLM completion, so N
    datasets fanned out over one session id would all observe "unbound" and all
    write. These tests hit the real relational engine, since the guarantee is a
    property of the upsert, not of the Python around it.
    """

    @pytest.mark.asyncio
    async def test_concurrent_claims_elect_exactly_one_dataset(self):
        from cognee.modules.session_lifecycle.exceptions import SessionDatasetMismatchError

        user_id = uuid4()
        session_id = f"scope-test-{uuid4()}"
        dataset_ids = [uuid4() for _ in range(6)]

        engine = get_relational_engine()
        async with engine.engine.begin() as conn:
            await conn.run_sync(SessionRecord.metadata.create_all)

        results = await asyncio.gather(
            *(
                claim_session_dataset(session_id=session_id, user_id=user_id, dataset_id=dataset_id)
                for dataset_id in dataset_ids
            ),
            return_exceptions=True,
        )

        winners = [r for r in results if not isinstance(r, Exception)]
        losers = [r for r in results if isinstance(r, SessionDatasetMismatchError)]
        assert len(winners) == 1, f"expected exactly one winner, got {results!r}"
        assert len(losers) == len(dataset_ids) - 1, f"unexpected results: {results!r}"

        bound = await get_session_dataset_id(session_id=session_id, user_id=user_id)
        assert bound in dataset_ids
        # Every loser must have been told the same surviving binding.
        assert {loser.bound_dataset_id for loser in losers} == {bound}

    @pytest.mark.asyncio
    async def test_concurrent_add_qa_writes_one_turn(self):
        """One session id fanned out over N datasets stores a single turn."""
        from cognee.modules.session_lifecycle.exceptions import SessionDatasetMismatchError

        user_id = uuid4()
        session_id = f"scope-test-{uuid4()}"

        engine = get_relational_engine()
        async with engine.engine.begin() as conn:
            await conn.run_sync(SessionRecord.metadata.create_all)

        cache = AsyncMock()
        managers = [SessionManager(cache_engine=cache, dataset_id=uuid4()) for _ in range(5)]

        with patch(
            "cognee.infrastructure.session.session_manager.index_session_qa",
            new_callable=AsyncMock,
        ) as index_mock:
            results = await asyncio.gather(
                *(
                    manager.add_qa(
                        user_id=str(user_id),
                        session_id=session_id,
                        question="q",
                        context="",
                        answer="a",
                    )
                    for manager in managers
                ),
                return_exceptions=True,
            )

        assert sum(not isinstance(r, Exception) for r in results) == 1
        assert all(
            isinstance(r, SessionDatasetMismatchError) for r in results if isinstance(r, Exception)
        ), f"unexpected failures: {results!r}"
        # The rejected writes must not have touched the cache or the vector store.
        assert cache.create_qa_entry.await_count == 1
        assert index_mock.await_count == 1

    @pytest.mark.asyncio
    async def test_terminal_session_can_still_be_bound(self):
        """A completed session that never got a dataset must not stay open to all."""
        from cognee.modules.session_lifecycle.exceptions import SessionDatasetMismatchError
        from cognee.modules.session_lifecycle.metrics import SessionStatus, mark_ended

        user_id = uuid4()
        session_id = f"scope-test-{uuid4()}"
        dataset_id = uuid4()

        engine = get_relational_engine()
        async with engine.engine.begin() as conn:
            await conn.run_sync(SessionRecord.metadata.create_all)

        await record_session_activity(str(user_id), session_id)
        await mark_ended(session_id=session_id, user_id=user_id, status=SessionStatus.COMPLETED)
        async with engine.get_async_session() as session:
            before = (
                await session.execute(
                    select(SessionRecord).where(SessionRecord.session_id == session_id)
                )
            ).scalar_one()
            frozen_activity = before.last_activity_at

        await claim_session_dataset(session_id=session_id, user_id=user_id, dataset_id=dataset_id)
        assert await get_session_dataset_id(session_id=session_id, user_id=user_id) == dataset_id

        with pytest.raises(SessionDatasetMismatchError):
            await claim_session_dataset(session_id=session_id, user_id=user_id, dataset_id=uuid4())

        # Binding a terminal session must not resurrect its activity clock.
        async with engine.get_async_session() as session:
            after = (
                await session.execute(
                    select(SessionRecord).where(SessionRecord.session_id == session_id)
                )
            ).scalar_one()
        assert after.last_activity_at == frozen_activity
        assert after.status == SessionStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_dangling_binding_is_still_enforced(self):
        """Enforcement reads the raw column, so a deleted dataset still binds.

        get_session_dataset() returns None here (its JOIN finds no Dataset row);
        if enforcement used that, the session would silently rebind elsewhere.
        """
        user_id = uuid4()
        session_id = f"scope-test-{uuid4()}"
        dangling_dataset_id = uuid4()

        engine = get_relational_engine()
        async with engine.engine.begin() as conn:
            await conn.run_sync(SessionRecord.metadata.create_all)
        await record_session_activity(str(user_id), session_id, dataset_id=str(dangling_dataset_id))

        assert await get_session_dataset(session_id=session_id, user_id=user_id) is None
        assert (
            await get_session_dataset_id(session_id=session_id, user_id=user_id)
            == dangling_dataset_id
        )

        from cognee.modules.session_lifecycle.exceptions import SessionDatasetMismatchError

        with pytest.raises(SessionDatasetMismatchError):
            await claim_session_dataset(session_id=session_id, user_id=user_id, dataset_id=uuid4())
