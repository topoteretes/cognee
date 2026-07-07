"""Tests for operation-level token usage tracking (COG-5563).

Mirrors the SessionRecord test patterns in
``cognee/tests/unit/api/v1/session/test_session_visibility.py`` but
exercises the operation lifecycle: ``ensure_and_touch_operation``,
``accumulate_operation_usage``, ``mark_operation_ended``, and the
``track_operation_usage`` context manager, including the rule that
session scope takes priority over operation scope so tokens are never
double-counted.
"""

from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.session_lifecycle.metrics import (
    SessionStatus,
    accumulate_operation_usage,
    ensure_and_touch_operation,
    mark_operation_ended,
)
from cognee.modules.session_lifecycle.models import OperationModelUsage, OperationRecord
from cognee.modules.session_lifecycle.usage_tracking import (
    record_llm_call,
    track_operation_usage,
    track_session_usage,
)


@pytest_asyncio.fixture
async def operation_tables():
    """Ensure operation_usage_records / operation_model_usage exist."""
    engine = get_relational_engine()
    async with engine.engine.begin() as conn:
        await conn.run_sync(OperationRecord.metadata.create_all)
    yield


async def _get_operation(operation_id: str, user_id) -> OperationRecord | None:
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        return await session.get(OperationRecord, (operation_id, user_id))


async def _get_model_usage(operation_id: str, user_id, model: str) -> OperationModelUsage | None:
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        return await session.get(OperationModelUsage, (operation_id, user_id, model))


async def _cleanup(operation_id: str, user_id):
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        row = await session.get(OperationRecord, (operation_id, user_id))
        if row:
            await session.delete(row)
        await session.commit()


class TestEnsureAndTouchOperation:
    @pytest.mark.asyncio
    async def test_creates_running_row(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            await ensure_and_touch_operation(
                operation_id=operation_id, user_id=user_id, operation_type="cognify"
            )
            row = await _get_operation(operation_id, user_id)
            assert row is not None
            assert row.status == SessionStatus.RUNNING.value
            assert row.operation_type == "cognify"
            assert row.tokens_in == 0
            assert row.tokens_out == 0
        finally:
            await _cleanup(operation_id, user_id)

    @pytest.mark.asyncio
    async def test_backfills_dataset_id_when_null(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        dataset_id = uuid4()
        try:
            await ensure_and_touch_operation(
                operation_id=operation_id, user_id=user_id, operation_type="search"
            )
            await ensure_and_touch_operation(
                operation_id=operation_id,
                user_id=user_id,
                operation_type="search",
                dataset_id=dataset_id,
            )
            row = await _get_operation(operation_id, user_id)
            assert row.dataset_id == dataset_id
        finally:
            await _cleanup(operation_id, user_id)

    @pytest.mark.asyncio
    async def test_does_not_resurrect_terminal_operation(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            await ensure_and_touch_operation(
                operation_id=operation_id, user_id=user_id, operation_type="memify"
            )
            await mark_operation_ended(
                operation_id=operation_id, user_id=user_id, status=SessionStatus.COMPLETED
            )
            before = await _get_operation(operation_id, user_id)

            await ensure_and_touch_operation(
                operation_id=operation_id, user_id=user_id, operation_type="memify"
            )
            after = await _get_operation(operation_id, user_id)
            assert after.status == SessionStatus.COMPLETED.value
            assert after.last_activity_at == before.last_activity_at
        finally:
            await _cleanup(operation_id, user_id)


class TestAccumulateOperationUsage:
    @pytest.mark.asyncio
    async def test_adds_tokens_and_cost_to_running_operation(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            await ensure_and_touch_operation(
                operation_id=operation_id, user_id=user_id, operation_type="cognify"
            )
            await accumulate_operation_usage(
                operation_id=operation_id,
                user_id=user_id,
                tokens_in=100,
                tokens_out=50,
                cost_usd=0.01,
                model="gpt-4o-mini",
            )
            await accumulate_operation_usage(
                operation_id=operation_id,
                user_id=user_id,
                tokens_in=10,
                tokens_out=5,
                cost_usd=0.001,
                model="gpt-4o-mini",
            )
            row = await _get_operation(operation_id, user_id)
            assert row.tokens_in == 110
            assert row.tokens_out == 55
            assert row.cost_usd == pytest.approx(0.011)
            assert row.last_model == "gpt-4o-mini"

            model_row = await _get_model_usage(operation_id, user_id, "gpt-4o-mini")
            assert model_row is not None
            assert model_row.tokens_in == 110
            assert model_row.tokens_out == 55
        finally:
            await _cleanup(operation_id, user_id)

    @pytest.mark.asyncio
    async def test_does_not_mutate_terminal_operation(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            await ensure_and_touch_operation(
                operation_id=operation_id, user_id=user_id, operation_type="cognify"
            )
            await mark_operation_ended(
                operation_id=operation_id, user_id=user_id, status=SessionStatus.COMPLETED
            )
            await accumulate_operation_usage(
                operation_id=operation_id,
                user_id=user_id,
                tokens_in=999,
                tokens_out=999,
                cost_usd=9.99,
                model="gpt-4o-mini",
            )
            row = await _get_operation(operation_id, user_id)
            assert row.tokens_in == 0
            assert row.tokens_out == 0
        finally:
            await _cleanup(operation_id, user_id)

    @pytest.mark.asyncio
    async def test_concurrent_calls_do_not_lose_updates(self, operation_tables):
        """Many LLM calls finishing concurrently within one operation (e.g. a
        cognify run's asyncio.gather'd chunk extraction) must not lose updates
        to the shared OperationRecord row."""
        import asyncio

        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            await ensure_and_touch_operation(
                operation_id=operation_id, user_id=user_id, operation_type="cognify"
            )
            await asyncio.gather(
                *[
                    accumulate_operation_usage(
                        operation_id=operation_id,
                        user_id=user_id,
                        tokens_in=1,
                        tokens_out=1,
                        model="gpt-4o-mini",
                    )
                    for _ in range(20)
                ]
            )
            row = await _get_operation(operation_id, user_id)
            assert row.tokens_in == 20
            assert row.tokens_out == 20

            model_row = await _get_model_usage(operation_id, user_id, "gpt-4o-mini")
            assert model_row.tokens_in == 20
            assert model_row.tokens_out == 20
        finally:
            await _cleanup(operation_id, user_id)


class TestMarkOperationEnded:
    @pytest.mark.asyncio
    async def test_transitions_to_completed(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            await ensure_and_touch_operation(
                operation_id=operation_id, user_id=user_id, operation_type="search"
            )
            await mark_operation_ended(
                operation_id=operation_id, user_id=user_id, status=SessionStatus.COMPLETED
            )
            row = await _get_operation(operation_id, user_id)
            assert row.status == SessionStatus.COMPLETED.value
            assert row.ended_at is not None
        finally:
            await _cleanup(operation_id, user_id)

    @pytest.mark.asyncio
    async def test_rejects_non_terminal_status(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        with pytest.raises(ValueError):
            await mark_operation_ended(
                operation_id=operation_id, user_id=user_id, status=SessionStatus.RUNNING
            )
        with pytest.raises(ValueError):
            await mark_operation_ended(
                operation_id=operation_id, user_id=user_id, status=SessionStatus.ABANDONED
            )


class TestTrackOperationUsage:
    @pytest.mark.asyncio
    async def test_marks_completed_on_success(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            async with track_operation_usage(operation_id, user_id, "cognify"):
                pass
            row = await _get_operation(operation_id, user_id)
            assert row.status == SessionStatus.COMPLETED.value
        finally:
            await _cleanup(operation_id, user_id)

    @pytest.mark.asyncio
    async def test_marks_failed_on_exception(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            with pytest.raises(RuntimeError):
                async with track_operation_usage(operation_id, user_id, "search"):
                    raise RuntimeError("boom")
            row = await _get_operation(operation_id, user_id)
            assert row.status == SessionStatus.FAILED.value
        finally:
            await _cleanup(operation_id, user_id)

    @pytest.mark.asyncio
    async def test_background_scope_leaves_operation_running(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            async with track_operation_usage(operation_id, user_id, "memify", background=True):
                pass
            row = await _get_operation(operation_id, user_id)
            assert row.status == SessionStatus.RUNNING.value
        finally:
            await _cleanup(operation_id, user_id)

    @pytest.mark.asyncio
    async def test_background_scope_still_marks_failed_on_exception(self, operation_tables):
        """A failure before the background task is ever spawned (e.g. during
        argument validation) must not leave the row stuck at 'running' forever."""
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            with pytest.raises(RuntimeError):
                async with track_operation_usage(operation_id, user_id, "cognify", background=True):
                    raise RuntimeError("boom before backgrounding")
            row = await _get_operation(operation_id, user_id)
            assert row.status == SessionStatus.FAILED.value
        finally:
            await _cleanup(operation_id, user_id)

    @pytest.mark.asyncio
    async def test_mark_failed_records_failure_without_exception(self, operation_tables):
        """Pipeline calls can fail via an error-sentinel return value instead of
        raising; the caller must be able to signal that via the yielded outcome."""
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            async with track_operation_usage(operation_id, user_id, "cognify") as outcome:
                pipeline_result = {"errored": True}
                if pipeline_result["errored"]:
                    outcome.mark_failed()
            row = await _get_operation(operation_id, user_id)
            assert row.status == SessionStatus.FAILED.value
        finally:
            await _cleanup(operation_id, user_id)

    @pytest.mark.asyncio
    async def test_mark_failed_ignored_when_ensure_failed(self, operation_tables, monkeypatch):
        """When ensure_and_touch_operation fails, tracking is disabled and a
        bare OperationOutcome is yielded — mark_failed() must not raise."""
        import cognee.modules.session_lifecycle.usage_tracking as usage_tracking_module

        async def _boom(**kwargs):
            raise RuntimeError("db unavailable")

        monkeypatch.setattr(
            "cognee.modules.session_lifecycle.metrics.ensure_and_touch_operation", _boom
        )
        operation_id = str(uuid4())
        user_id = uuid4()
        async with usage_tracking_module.track_operation_usage(
            operation_id, user_id, "cognify"
        ) as outcome:
            outcome.mark_failed()
        assert outcome.success is False

    @pytest.mark.asyncio
    async def test_llm_call_inside_scope_accumulates_to_operation(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            async with track_operation_usage(operation_id, user_id, "cognify"):
                await record_llm_call(
                    input_text="hello",
                    output_text="world",
                    model="gpt-4o-mini",
                    tokens_in_override=42,
                    tokens_out_override=7,
                )
            row = await _get_operation(operation_id, user_id)
            assert row.tokens_in == 42
            assert row.tokens_out == 7
        finally:
            await _cleanup(operation_id, user_id)

    @pytest.mark.asyncio
    async def test_llm_call_outside_scope_does_not_accumulate(self, operation_tables):
        operation_id = str(uuid4())
        user_id = uuid4()
        try:
            async with track_operation_usage(operation_id, user_id, "cognify"):
                pass
            # Scope has exited — the ContextVar is reset, so this call is a no-op.
            await record_llm_call(
                input_text="hello",
                output_text="world",
                model="gpt-4o-mini",
                tokens_in_override=42,
                tokens_out_override=7,
            )
            row = await _get_operation(operation_id, user_id)
            assert row.tokens_in == 0
            assert row.tokens_out == 0
        finally:
            await _cleanup(operation_id, user_id)


class TestSessionOperationPriority:
    """Session scope must win over operation scope so tokens are never double-counted."""

    @pytest.mark.asyncio
    async def test_session_wins_when_both_scopes_active(self, operation_tables):
        engine = get_relational_engine()
        from cognee.modules.session_lifecycle.models import SessionRecord

        async with engine.engine.begin() as conn:
            await conn.run_sync(SessionRecord.metadata.create_all)

        operation_id = str(uuid4())
        session_id = str(uuid4())
        user_id = uuid4()
        try:
            async with track_operation_usage(operation_id, user_id, "remember"):
                async with track_session_usage(session_id, user_id):
                    from cognee.modules.session_lifecycle.metrics import ensure_and_touch_session

                    await ensure_and_touch_session(session_id=session_id, user_id=user_id)
                    await record_llm_call(
                        input_text="hello",
                        output_text="world",
                        model="gpt-4o-mini",
                        tokens_in_override=42,
                        tokens_out_override=7,
                    )

            operation_row = await _get_operation(operation_id, user_id)
            assert operation_row.tokens_in == 0
            assert operation_row.tokens_out == 0

            async with engine.get_async_session() as session:
                session_row = await session.get(SessionRecord, (session_id, user_id))
            assert session_row.tokens_in == 42
            assert session_row.tokens_out == 7
        finally:
            await _cleanup(operation_id, user_id)
            async with engine.get_async_session() as session:
                row = await session.get(SessionRecord, (session_id, user_id))
                if row:
                    await session.delete(row)
                await session.commit()
