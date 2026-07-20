import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.infrastructure.session.session_scope import get_storage_session_id
from cognee.modules.session_lifecycle.models import SessionRecord


def _detail_endpoint(router_module):
    router = router_module.get_sessions_router()
    return next(route.endpoint for route in router.routes if route.path == "/{session_id}")


def _row(*, session_id, owner_id, dataset_id, public_session_id):
    row = SimpleNamespace(
        session_id=session_id,
        user_id=owner_id,
        dataset_id=dataset_id,
        public_session_id=public_session_id,
    )
    row.to_dict = lambda: {
        "session_id": session_id,
        "user_id": str(owner_id),
        "dataset_id": str(dataset_id) if public_session_id is not None else None,
    }
    return row


@pytest.mark.asyncio
async def test_dashboard_visibility_requires_live_grant_for_scoped_owner_rows():
    router_module = importlib.import_module("cognee.api.v1.sessions.routers.get_sessions_router")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(SessionRecord.__table__.create)

    owner_id = uuid4()
    other_owner_id = uuid4()
    dataset_a = uuid4()
    dataset_b = uuid4()
    now = datetime.now(timezone.utc)
    rows = [
        SessionRecord(
            session_id="legacy",
            public_session_id=None,
            user_id=owner_id,
            dataset_id=dataset_a,
            status="running",
            started_at=now,
            last_activity_at=now,
        ),
        SessionRecord(
            session_id=get_storage_session_id("owned-a", dataset_a),
            public_session_id="owned-a",
            user_id=owner_id,
            dataset_id=dataset_a,
            status="running",
            started_at=now,
            last_activity_at=now,
        ),
        SessionRecord(
            session_id=get_storage_session_id("shared-a", dataset_a),
            public_session_id="shared-a",
            user_id=other_owner_id,
            dataset_id=dataset_a,
            status="running",
            started_at=now,
            last_activity_at=now,
        ),
        SessionRecord(
            session_id=get_storage_session_id("owned-b", dataset_b),
            public_session_id="owned-b",
            user_id=owner_id,
            dataset_id=dataset_b,
            status="running",
            started_at=now,
            last_activity_at=now,
        ),
    ]
    try:
        async with sessions() as session:
            session.add_all(rows)
            await session.commit()

            revoked = list(
                (
                    await session.scalars(
                        select(SessionRecord).where(
                            router_module._session_visibility_sql([owner_id], [])
                        )
                    )
                ).all()
            )
            permitted_a = list(
                (
                    await session.scalars(
                        select(SessionRecord).where(
                            router_module._session_visibility_sql([owner_id], [dataset_a])
                        )
                    )
                ).all()
            )
    finally:
        await engine.dispose()

    assert {row.to_dict()["session_id"] for row in revoked} == {"legacy"}
    assert {row.to_dict()["session_id"] for row in permitted_a} == {
        "legacy",
        "owned-a",
        "shared-a",
    }


@pytest.mark.asyncio
async def test_detail_selects_exact_dataset_and_owner(monkeypatch):
    router_module = importlib.import_module("cognee.api.v1.sessions.routers.get_sessions_router")
    manager_module = importlib.import_module("cognee.infrastructure.session.get_session_manager")

    caller_id = uuid4()
    owner_id = uuid4()
    dataset_id = uuid4()
    user = SimpleNamespace(id=caller_id)
    row = _row(
        session_id="shared",
        owner_id=owner_id,
        dataset_id=dataset_id,
        public_session_id="shared",
    )
    get_row = AsyncMock(return_value=row)
    manager = SimpleNamespace(
        is_available=True,
        get_session=AsyncMock(return_value=[]),
        get_agent_trace_session=AsyncMock(return_value=[]),
    )
    get_manager = MagicMock(return_value=manager)

    monkeypatch.setattr(
        router_module, "_permitted_dataset_ids_for", AsyncMock(return_value=[dataset_id])
    )
    monkeypatch.setattr(router_module, "_visible_user_ids", AsyncMock(return_value=[caller_id]))
    monkeypatch.setattr(router_module, "get_session_row", get_row)
    monkeypatch.setattr(manager_module, "get_session_manager", get_manager)

    result = await _detail_endpoint(router_module)(
        session_id="shared",
        dataset_id=dataset_id,
        owner_user_id=owner_id,
        user=user,
    )

    get_row.assert_awaited_once_with(
        session_id="shared",
        user_id=caller_id,
        user_ids=[caller_id],
        permitted_dataset_ids=[dataset_id],
        dataset_id=dataset_id,
        owner_user_id=owner_id,
    )
    get_manager.assert_called_once_with(dataset_id=dataset_id)
    manager.get_session.assert_awaited_once_with(
        user_id=str(owner_id), session_id="shared", formatted=False
    )
    assert result["dataset_id"] == str(dataset_id)


@pytest.mark.asyncio
async def test_detail_quarantines_legacy_row_with_stale_dataset_id(monkeypatch):
    router_module = importlib.import_module("cognee.api.v1.sessions.routers.get_sessions_router")
    manager_module = importlib.import_module("cognee.infrastructure.session.get_session_manager")

    owner_id = uuid4()
    stale_dataset_id = uuid4()
    user = SimpleNamespace(id=owner_id)
    row = _row(
        session_id="legacy",
        owner_id=owner_id,
        dataset_id=stale_dataset_id,
        public_session_id=None,
    )
    manager = SimpleNamespace(
        is_available=True,
        get_session=AsyncMock(return_value=[]),
        get_agent_trace_session=AsyncMock(return_value=[]),
    )
    get_manager = MagicMock(return_value=manager)

    monkeypatch.setattr(router_module, "_permitted_dataset_ids_for", AsyncMock(return_value=[]))
    monkeypatch.setattr(router_module, "_visible_user_ids", AsyncMock(return_value=[owner_id]))
    monkeypatch.setattr(router_module, "get_session_row", AsyncMock(return_value=row))
    monkeypatch.setattr(manager_module, "get_session_manager", get_manager)

    await _detail_endpoint(router_module)(
        session_id="legacy",
        dataset_id=None,
        owner_user_id=owner_id,
        user=user,
    )

    get_manager.assert_called_once_with(dataset_id=None)


@pytest.mark.asyncio
async def test_detail_rejects_unreadable_dataset_before_lookup(monkeypatch):
    router_module = importlib.import_module("cognee.api.v1.sessions.routers.get_sessions_router")
    dataset_id = uuid4()
    get_row = AsyncMock()

    monkeypatch.setattr(router_module, "_permitted_dataset_ids_for", AsyncMock(return_value=[]))
    monkeypatch.setattr(router_module, "_visible_user_ids", AsyncMock(return_value=[uuid4()]))
    monkeypatch.setattr(router_module, "get_session_row", get_row)

    with pytest.raises(HTTPException) as error:
        await _detail_endpoint(router_module)(
            session_id="shared",
            dataset_id=dataset_id,
            owner_user_id=uuid4(),
            user=SimpleNamespace(id=uuid4()),
        )

    assert error.value.status_code == 404
    get_row.assert_not_awaited()


@pytest.mark.asyncio
async def test_detail_requires_dataset_selector_for_scoped_row(monkeypatch):
    router_module = importlib.import_module("cognee.api.v1.sessions.routers.get_sessions_router")
    owner_id = uuid4()
    dataset_id = uuid4()

    monkeypatch.setattr(router_module, "_permitted_dataset_ids_for", AsyncMock(return_value=[]))
    monkeypatch.setattr(router_module, "_visible_user_ids", AsyncMock(return_value=[owner_id]))
    monkeypatch.setattr(
        router_module,
        "get_session_row",
        AsyncMock(
            return_value=_row(
                session_id="shared",
                owner_id=owner_id,
                dataset_id=dataset_id,
                public_session_id="shared",
            )
        ),
    )

    with pytest.raises(HTTPException) as error:
        await _detail_endpoint(router_module)(
            session_id="shared",
            dataset_id=None,
            owner_user_id=owner_id,
            user=SimpleNamespace(id=owner_id),
        )

    assert error.value.status_code == 409
