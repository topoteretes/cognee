"""Typed UUID writes must honor permissions and immutable session bindings."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from cognee.api.v1.remember import remember
from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.infrastructure.databases.relational import get_relational_config, get_relational_engine
from cognee.memory import QAEntry
from cognee.modules.data.models import Dataset
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import ACL, Permission, User


@pytest_asyncio.fixture
async def state(tmp_path, monkeypatch):
    cfg = get_relational_config()
    monkeypatch.setattr(cfg, "db_path", str(tmp_path))
    monkeypatch.setattr(cfg, "db_name", "typed.db")
    monkeypatch.setattr(cfg, "db_provider", "sqlite")
    engine = get_relational_engine()
    await engine.create_database()
    user = User(id=uuid4(), email="writer@test.example", hashed_password="unused")
    owned = Dataset(id=uuid4(), name="same-name", owner_id=user.id)
    shared = Dataset(id=uuid4(), name="same-name", owner_id=uuid4())
    write = Permission(id=uuid4(), name="write")
    async with engine.get_async_session() as session:
        session.add_all([user, owned, shared, write])
        await session.flush()
        session.add_all(
            [
                ACL(principal_id=user.id, dataset_id=ds.id, permission_id=write.id)
                for ds in (owned, shared)
            ]
        )
        await session.commit()
    sm = SimpleNamespace(is_available=True, add_qa=AsyncMock(return_value="qa-id"))
    monkeypatch.setattr(
        importlib.import_module("cognee.infrastructure.session.get_session_manager"),
        "get_session_manager",
        lambda: sm,
    )
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.engine.operations.setup"), "setup", AsyncMock()
    )
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.serve.state"), "get_remote_client", lambda: None
    )
    try:
        yield SimpleNamespace(**locals())
    finally:
        await engine.engine.dispose()


@pytest.mark.asyncio
async def test_typed_uuid_targets_shared_dataset_with_same_name(state):
    result = await remember(
        QAEntry(question="q", answer="a"),
        dataset_id=state.shared.id,
        session_id="shared-session",
        user=state.user,
    )
    assert result.dataset_id == str(state.shared.id)
    assert result.dataset_name == state.shared.name
    state.sm.add_qa.assert_awaited_once()


@pytest.mark.asyncio
async def test_typed_uuid_permission_denial_precedes_cache_write(state):
    with pytest.raises(PermissionDeniedError):
        await remember(
            QAEntry(question="q", answer="a"),
            dataset_id=uuid4(),
            session_id="denied-session",
            user=state.user,
        )
    state.sm.add_qa.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_datasets_cannot_claim_same_session(state):
    outcomes = await asyncio.gather(
        *[
            remember(
                QAEntry(question="q", answer="a"),
                dataset_id=dataset.id,
                session_id="concurrent-session",
                user=state.user,
            )
            for dataset in (state.owned, state.shared)
        ],
        return_exceptions=True,
    )
    assert sum(isinstance(result, ValueError) for result in outcomes) == 1
    state.sm.add_qa.assert_awaited_once()


@pytest.mark.asyncio
async def test_http_entry_really_accepts_advertised_uuid(state):
    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/api/v1/remember")
    app.dependency_overrides[get_authenticated_user] = lambda: state.user
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        spec = (await client.get("/openapi.json")).json()
        assert (
            spec["paths"]["/api/v1/remember/entry"]["post"]["x-cognee-session-dataset-ids"] is True
        )
        response = await client.post(
            "/api/v1/remember/entry",
            json={
                "entry": {"type": "qa", "question": "q", "answer": "a"},
                "dataset_id": str(state.shared.id),
                "session_id": "http-session",
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["dataset_id"] == str(state.shared.id)
    state.sm.add_qa.assert_awaited_once()


@pytest.mark.asyncio
async def test_remote_client_preserves_dataset_uuid(monkeypatch):
    from cognee.api.v1.serve.cloud_client import CloudClient

    captured = {}

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"status": "session_stored"}

    class Session:
        def post(self, url, *, json):
            captured.update(json)
            return Response()

    client = CloudClient("http://test", "test-key")
    monkeypatch.setattr(client, "_get_session", AsyncMock(return_value=Session()))
    ident = uuid4()
    await client.remember_entry(
        QAEntry(question="q", answer="a"), dataset_id=ident, session_id="remote-session"
    )
    assert captured["dataset_id"] == str(ident)
