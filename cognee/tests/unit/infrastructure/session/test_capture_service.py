"""Exercise the capture service with real SQL cache and session-binding transactions."""

import asyncio
from contextlib import asynccontextmanager
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.infrastructure.databases.cache.sql.capture import CaptureConflictError
from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.infrastructure.databases.cache.sql.tables import (
    cache_capture_receipts,
    cache_trace_entries,
)
from cognee.infrastructure.session import capture
from cognee.memory import QAEntry, TraceEntry
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.session_lifecycle.models import SessionRecord
from cognee.modules.users.exceptions import PermissionDeniedError


@pytest_asyncio.fixture
async def service(tmp_path, monkeypatch):
    adapter = SqlCacheAdapter(f"sqlite+aiosqlite:///{tmp_path}/cache.db")
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/sessions.db")
    async with engine.begin() as connection:
        await connection.run_sync(SessionRecord.__table__.create)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
    dataset = SimpleNamespace(id=uuid4(), name="project")
    resolver = AsyncMock(return_value=[dataset])
    index = AsyncMock()
    monkeypatch.setattr(capture, "get_cache_engine", lambda: adapter)
    monkeypatch.setattr(capture, "supported", lambda: True)
    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", AsyncMock())
    monkeypatch.setattr(
        import_module("cognee.modules.data.methods.get_authorized_existing_datasets"),
        "get_authorized_existing_datasets",
        resolver,
    )
    relational = SimpleNamespace(get_async_session=sessions)
    monkeypatch.setattr(
        "cognee.infrastructure.databases.relational.get_relational_engine", lambda: relational
    )
    monkeypatch.setattr(
        "cognee.modules.session_lifecycle.metrics.get_relational_engine", lambda: relational
    )
    monkeypatch.setattr("cognee.infrastructure.session.session_embeddings.index_session_qa", index)

    @asynccontextmanager
    async def context(*args, **kwargs):
        yield

    monkeypatch.setattr(
        "cognee.context_global_variables.set_database_global_context_variables", context
    )
    monkeypatch.setattr(
        "cognee.modules.session_lifecycle.usage_tracking.track_session_usage", context
    )
    args = {
        "entry": QAEntry(question="q", answer="a"),
        "client_event_id": uuid4(),
        "dataset_name": dataset.name,
        "dataset_id": None,
        "session_id": "session",
        "user": user,
    }
    yield SimpleNamespace(
        adapter=adapter,
        sessions=sessions,
        user=user,
        dataset=dataset,
        resolver=resolver,
        index=index,
        args=args,
    )
    await adapter.close()
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("by_id", [False, True])
async def test_receipt_retry_survives_dataset_deletion_and_binding_changes(
    service, monkeypatch, by_id
):
    args = {**service.args, "dataset_id": service.dataset.id if by_id else None}
    receipt = await capture.remember_capture(**args)
    service.resolver.side_effect = DatasetNotFoundError("deleted")
    async with service.sessions() as session, session.begin():
        await session.execute(update(SessionRecord).values(dataset_id=uuid4()))
    monkeypatch.setattr(
        "cognee.modules.session_lifecycle.metrics.ensure_and_touch_session",
        AsyncMock(side_effect=OSError("relational unavailable")),
    )
    assert await capture.remember_capture(**args) == receipt
    assert service.resolver.await_count == 1
    assert service.index.await_count == 1
    with pytest.raises(CaptureConflictError):
        await capture.remember_capture(**{**args, "entry": QAEntry(question="q", answer="changed")})


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [DatasetNotFoundError, PermissionDeniedError])
async def test_first_admission_requires_existing_writable_dataset(service, error):
    service.resolver.side_effect = error("unavailable")
    with pytest.raises(error):
        await capture.remember_capture(**service.args)
    service.resolver.assert_awaited_once_with(["project"], "write", service.user, strict=True)
    async with service.adapter.sessionmaker() as session:
        assert not (await session.execute(select(cache_capture_receipts))).all()


@pytest.mark.asyncio
async def test_first_admission_checks_binding_and_binding_failures_are_retryable(
    service, monkeypatch
):
    async with service.sessions() as session, session.begin():
        session.add(
            SessionRecord(session_id="session", user_id=service.user.id, dataset_id=uuid4())
        )
    with pytest.raises(capture.CaptureRequestError) as error:
        await capture.remember_capture(**service.args)
    assert error.value.code == "capture_session_dataset_mismatch"
    monkeypatch.setattr(
        "cognee.modules.session_lifecycle.metrics.ensure_and_touch_session",
        AsyncMock(side_effect=OSError("relational unavailable")),
    )
    with pytest.raises(RuntimeError, match="binding temporarily unavailable"):
        await capture.remember_capture(**service.args)


@pytest.mark.asyncio
async def test_pending_index_retries_without_rebinding(service, monkeypatch):
    service.index.side_effect = OSError("lost index ACK")
    with pytest.raises(OSError):
        await capture.remember_capture(**service.args)
    service.index.side_effect = None
    monkeypatch.setattr(
        "cognee.modules.session_lifecycle.metrics.ensure_and_touch_session",
        AsyncMock(side_effect=AssertionError("retry must not rebind")),
    )
    result = await capture.remember_capture(**service.args)
    assert result["status"] == "session_stored"
    assert service.index.await_count == 2
    assert service.resolver.await_args.args[0] == [service.dataset.id]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [DatasetNotFoundError, PermissionDeniedError])
async def test_pending_index_never_writes_after_dataset_loss(service, error):
    service.index.side_effect = OSError("lost ACK")
    with pytest.raises(OSError):
        await capture.remember_capture(**service.args)
    service.resolver.side_effect = error("unavailable")
    receipt = await capture.remember_capture(**service.args)
    assert receipt["status"] == "session_stored"
    assert service.index.await_count == 1


@pytest.mark.asyncio
async def test_trace_feedback_does_not_block_cache_writer_or_repeat_on_retry(service, monkeypatch):
    from cognee.infrastructure.session.session_agent_trace import fallback_agent_trace_feedback

    started, release = asyncio.Event(), asyncio.Event()

    async def feedback(**kwargs):
        started.set()
        await release.wait()
        return fallback_agent_trace_feedback(
            origin_function="tool", status="success", error_message=""
        )

    generator = AsyncMock(side_effect=feedback)
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_agent_trace.generate_agent_trace_feedback", generator
    )
    monkeypatch.setattr(
        "cognee.infrastructure.session.session_manager.SessionManager.is_auto_feedback_enabled",
        lambda _: True,
    )
    args = {
        **service.args,
        "entry": TraceEntry(origin_function="tool", generate_feedback_with_llm=True),
    }
    task = asyncio.create_task(capture.remember_capture(**args))
    try:
        await asyncio.wait_for(started.wait(), 3)
        peer = SqlCacheAdapter(service.adapter.db_uri)
        try:
            await asyncio.wait_for(peer.set_value("unrelated", "value"), 3)
        finally:
            await peer.close()
    finally:
        release.set()
        result = await task
    assert await capture.remember_capture(**args) == result
    generator.assert_awaited_once()
    async with service.adapter.sessionmaker() as session:
        assert len((await session.execute(select(cache_trace_entries))).all()) == 1


@pytest.mark.parametrize(
    "backend,enabled,expected",
    [
        ("sqlite", True, True),
        ("turso", True, True),
        ("postgres", True, True),
        ("redis", True, False),
        ("fs", True, False),
        ("tapes", True, False),
        ("sqlite", False, False),
    ],
)
def test_capability_uses_config_without_constructing_broken_cache(
    monkeypatch, backend, enabled, expected
):
    monkeypatch.setattr(
        "cognee.infrastructure.databases.cache.config.get_cache_config",
        lambda: SimpleNamespace(caching=enabled, cache_backend=backend),
    )
    factory = Mock(side_effect=OSError("broken cache"))
    monkeypatch.setattr(capture, "get_cache_engine", factory)
    assert capture.supported() is expected
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_account_deletion_hook_cleans_receipts(service, monkeypatch):
    from cognee.modules.users.get_user_manager import UserManager

    await capture.remember_capture(**service.args)
    monkeypatch.setattr(
        "cognee.infrastructure.databases.cache.get_cache_engine", lambda: service.adapter
    )
    monkeypatch.setattr(
        "cognee.infrastructure.databases.cache.config.get_cache_config",
        lambda: SimpleNamespace(cache_backend="sqlite"),
    )
    agents = AsyncMock()
    monkeypatch.setattr(
        import_module("cognee.modules.agents.delete_agents_of_user"),
        "delete_agents_of_user",
        agents,
    )
    manager = UserManager(SimpleNamespace(session=object()))
    await manager.on_before_delete(service.user)
    agents.assert_awaited_once()
    async with service.adapter.sessionmaker() as session:
        assert len((await session.execute(select(cache_capture_receipts))).all()) == 1
    await manager.on_after_delete(service.user)
    async with service.adapter.sessionmaker() as session:
        assert not (await session.execute(select(cache_capture_receipts))).all()


@pytest.mark.asyncio
async def test_real_dataset_resolver_does_not_create_unknown_names(service, monkeypatch):
    module = import_module("cognee.modules.data.methods.get_authorized_existing_datasets")
    # Restore the real resolver while substituting only its backing reads.
    from cognee.modules.data.methods import get_authorized_existing_datasets as resolver

    monkeypatch.setattr(module, "get_authorized_existing_datasets", resolver)
    ids = import_module("cognee.modules.data.methods.get_dataset_ids")
    monkeypatch.setattr(ids, "get_datasets", AsyncMock(return_value=[]))
    with pytest.raises(DatasetNotFoundError):
        await capture.remember_capture(**service.args)
    denied = AsyncMock(side_effect=PermissionDeniedError("no write permission"))
    monkeypatch.setattr(module, "get_specific_user_permission_datasets", denied)
    with pytest.raises(PermissionDeniedError):
        await capture.remember_capture(**{**service.args, "dataset_id": service.dataset.id})
    denied.assert_awaited_once_with(service.user.id, "write", [service.dataset.id])
