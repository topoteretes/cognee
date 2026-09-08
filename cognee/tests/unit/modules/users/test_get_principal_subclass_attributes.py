"""get_principal must return a fully loaded principal (gh #4976).

`Principal` uses joined-table inheritance, so `Role.name` / `Role.tenant_id`
live in the `roles` table rather than in `principals`. A plain
select(Principal) loads the base row only, and reading a subclass column
afterwards refreshes a detached instance -> DetachedInstanceError.

That is why GET /permissions/principals/{id}/datasets returned 500 for role
principals while user and tenant principals worked: only the role branch of
authorized_get_principal_datasets reads subclass columns.

Runs against a real temporary SQLite database — mocks cannot catch this,
since the bug is in what SQLAlchemy loads.
"""

import importlib
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.users.models.Role import Role
from cognee.modules.users.models.Tenant import Tenant
from cognee.modules.users.models.User import User

get_principal_mod = importlib.import_module(
    "cognee.modules.users.permissions.methods.get_principal"
)


@pytest_asyncio.fixture
async def principals_engine(tmp_path, monkeypatch):
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="principals_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(get_principal_mod, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


@pytest.mark.asyncio
async def test_role_subclass_columns_readable_after_session_closes(principals_engine):
    """The regression: reading Role.tenant_id / Role.name must not raise."""
    tenant_id, role_id = uuid4(), uuid4()

    async with principals_engine.get_async_session() as session:
        session.add(Tenant(id=tenant_id, name="acme"))
        await session.flush()
        session.add(Role(id=role_id, name="editors", tenant_id=tenant_id))
        await session.commit()

    principal = await get_principal_mod.get_principal(role_id)

    # The session is closed; these are the two attributes
    # authorized_get_principal_datasets reads on the role branch.
    assert principal.type == "role"
    assert principal.tenant_id == tenant_id
    assert principal.name == "editors"


@pytest.mark.asyncio
async def test_user_principal_still_resolves(principals_engine):
    """The path that already worked must keep working."""
    tenant_id, user_id = uuid4(), uuid4()

    async with principals_engine.get_async_session() as session:
        session.add(Tenant(id=tenant_id, name="acme"))
        await session.flush()
        session.add(
            User(
                id=user_id,
                email="someone@example.com",
                hashed_password="x",
                tenant_id=tenant_id,
            )
        )
        await session.commit()

    principal = await get_principal_mod.get_principal(user_id)

    assert principal.type == "user"
    assert principal.id == user_id
    assert principal.tenant_id == tenant_id


@pytest.mark.asyncio
async def test_tenant_principal_still_resolves(principals_engine):
    tenant_id = uuid4()

    async with principals_engine.get_async_session() as session:
        session.add(Tenant(id=tenant_id, name="acme"))
        await session.commit()

    principal = await get_principal_mod.get_principal(tenant_id)

    assert principal.type == "tenant"
    assert principal.id == tenant_id


@pytest.mark.asyncio
async def test_get_principal_uses_exactly_one_session(principals_engine, monkeypatch):
    """One call, one pooled connection.

    The fix loads the subclass columns inside the existing session instead of
    handing the caller a live one to work inside. Nesting sessions would let a
    single request hold two pool connections and can deadlock the pool at its
    limit, so pin the count.
    """
    tenant_id, role_id = uuid4(), uuid4()

    async with principals_engine.get_async_session() as session:
        session.add(Tenant(id=tenant_id, name="acme"))
        await session.flush()
        session.add(Role(id=role_id, name="editors", tenant_id=tenant_id))
        await session.commit()

    opened = 0
    original = principals_engine.get_async_session

    def counting_session():
        nonlocal opened
        opened += 1
        return original()

    monkeypatch.setattr(principals_engine, "get_async_session", counting_session)

    principal = await get_principal_mod.get_principal(role_id)

    assert principal.name == "editors"
    assert opened == 1, f"expected a single session, got {opened}"
