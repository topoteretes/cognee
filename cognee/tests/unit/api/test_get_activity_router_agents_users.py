"""Tests for GET /activity/agents and GET /activity/users scoping.

``/agents`` runs against a private in-memory SQLite database seeded with two
unrelated users, each owning an agent. Replaying canned rows would pass no
matter what the queries filter on; real rows are what prove the caller never
sees the other user, their agent, or anything computed from their activity.
"""

import importlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.exceptions import CogneeApiError
from cognee.infrastructure.databases.relational.ModelBase import Base
from cognee.modules.data.models.Data import Data
from cognee.modules.search.models.Query import Query
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import User
from cognee.modules.users.models.UserApiKey import UserApiKey

# The routers package re-exports the factory under the module's own name, so
# import the module itself to reach the attributes these tests patch.
router_module = importlib.import_module("cognee.api.v1.activity.routers.get_activity_router")
visible_ids_module = importlib.import_module("cognee.modules.users.methods.get_visible_user_ids")


class _Engine:
    """The slice of the relational engine the router and helpers call."""

    def __init__(self, sessionmaker):
        self._sessionmaker = sessionmaker

    @asynccontextmanager
    async def get_async_session(self):
        async with self._sessionmaker() as session:
            yield session


def _user(email, parent_user_id=None) -> User:
    return User(
        id=uuid4(),
        email=email,
        hashed_password="!",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        parent_user_id=parent_user_id,
    )


@pytest_asyncio.fixture
async def two_users(monkeypatch):
    """Alice and Bob, each with one agent, an API key, data and a search."""
    sql_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with sql_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    engine = _Engine(async_sessionmaker(sql_engine, expire_on_commit=False))

    from cognee.infrastructure.databases import relational as relational_module

    monkeypatch.setattr(relational_module, "get_relational_engine", lambda: engine)
    monkeypatch.setattr(visible_ids_module, "get_relational_engine", lambda: engine)

    alice = _user("alice@example.com")
    bob = _user("bob@example.com")
    alice_agent = _user(f"helper+{alice.id}@cognee.agent", parent_user_id=alice.id)
    bob_agent = _user(f"helper+{bob.id}@cognee.agent", parent_user_id=bob.id)
    now = datetime.now(timezone.utc)

    async with engine.get_async_session() as session:
        session.add_all([alice, bob])
        await session.flush()
        session.add_all([alice_agent, bob_agent])
        await session.flush()
        for owner in (alice, alice_agent, bob, bob_agent):
            session.add(UserApiKey(user_id=owner.id, api_key=f"key-{owner.id}"))
            session.add(Data(name="doc", owner_id=owner.id, created_at=now))
            session.add(Query(text="q", query_type="CHUNKS", user_id=owner.id, created_at=now))
        await session.commit()

    yield SimpleNamespace(alice=alice, bob=bob, alice_agent=alice_agent, bob_agent=bob_agent)

    await sql_engine.dispose()


def _client(caller) -> TestClient:
    app = FastAPI()
    app.include_router(router_module.get_activity_router(), prefix="/activity")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: caller

    # Registered by the real app in cognee/api/client.py.
    @app.exception_handler(CogneeApiError)
    async def _cognee_error_handler(_request, exc: CogneeApiError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    return TestClient(app)


# --------------------------------------------------------------------------- #
# /agents
# --------------------------------------------------------------------------- #


def test_agents_lists_only_the_caller_and_their_own_agents(two_users):
    response = _client(two_users.alice).get("/activity/agents")

    assert response.status_code == 200
    by_id = {row["id"]: row for row in response.json()}
    assert set(by_id) == {str(two_users.alice.id), str(two_users.alice_agent.id)}
    assert by_id[str(two_users.alice_agent.id)]["is_agent"] is True
    assert by_id[str(two_users.alice_agent.id)]["api_key_count"] == 1
    assert by_id[str(two_users.alice.id)]["status"] == "LIVE"


def test_agents_never_includes_another_users_account_or_agents(two_users):
    body = _client(two_users.bob).get("/activity/agents").json()

    ids = {row["id"] for row in body}
    emails = {row["email"] for row in body}
    assert str(two_users.alice.id) not in ids
    assert str(two_users.alice_agent.id) not in ids
    assert not any(str(two_users.alice.id) in email for email in emails)
    assert emails == {"bob@example.com", f"helper+{two_users.bob.id}@cognee.agent"}


def test_agents_for_a_caller_with_no_agents_is_just_themselves(two_users):
    body = _client(two_users.alice_agent).get("/activity/agents").json()

    assert [row["id"] for row in body] == [str(two_users.alice_agent.id)]


# --------------------------------------------------------------------------- #
# /users
# --------------------------------------------------------------------------- #


def _stub_get_users_in_tenant(monkeypatch, fake):
    from cognee.modules.users.tenants import methods as tenant_methods

    monkeypatch.setattr(tenant_methods, "get_users_in_tenant", fake)


def test_users_passes_the_caller_and_returns_the_tenant_users(monkeypatch):
    tenant_id = uuid4()
    caller = SimpleNamespace(id=uuid4(), tenant_id=tenant_id, email="admin@example.com")
    tenant_users = [{"id": str(caller.id), "email": caller.email, "roles": []}]
    calls = []

    async def fake_get_users_in_tenant(tenant_id_arg, user_arg):
        calls.append((tenant_id_arg, user_arg))
        return tenant_users

    _stub_get_users_in_tenant(monkeypatch, fake_get_users_in_tenant)

    response = _client(caller).get("/activity/users")

    assert response.status_code == 200
    assert response.json() == tenant_users
    assert calls == [(tenant_id, caller)]


def test_users_denial_is_a_403_not_an_empty_list(monkeypatch):
    caller = SimpleNamespace(id=uuid4(), tenant_id=uuid4(), email="member@example.com")

    async def fake_get_users_in_tenant(_tenant_id, _user):
        raise PermissionDeniedError(message="User is not authorized to manage users")

    _stub_get_users_in_tenant(monkeypatch, fake_get_users_in_tenant)

    assert _client(caller).get("/activity/users").status_code == 403


def test_users_without_a_tenant_is_empty(monkeypatch):
    caller = SimpleNamespace(id=uuid4(), tenant_id=None, email="solo@example.com")

    async def fake_get_users_in_tenant(_tenant_id, _user):
        pytest.fail("a caller with no tenant has no tenant to list")

    _stub_get_users_in_tenant(monkeypatch, fake_get_users_in_tenant)

    response = _client(caller).get("/activity/users")

    assert response.status_code == 200
    assert response.json() == []
