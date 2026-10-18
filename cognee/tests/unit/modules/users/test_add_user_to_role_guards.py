"""add_user_to_role must report a missing user/role, not AttributeError.

The lookups used to be dereferenced above their own guards: the tenant
query read `role.tenant_id` and the membership load read `user`, both
before the `if not role` / `if not user` checks below them. An unknown id
therefore raised AttributeError on None, which the permissions router
turns into a 500 instead of the intended 404-shaped error.

Runs against a real temporary SQLite database — the failure is in
statement ordering against real query results, so mocks would not show it.
"""

import importlib
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.users.exceptions import RoleNotFoundError, UserNotFoundError
from cognee.modules.users.models.Role import Role
from cognee.modules.users.models.Tenant import Tenant
from cognee.modules.users.models.User import User

add_user_to_role_mod = importlib.import_module(
    "cognee.modules.users.roles.methods.add_user_to_role"
)


@pytest_asyncio.fixture
async def roles_engine(tmp_path, monkeypatch):
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="add_user_to_role_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(add_user_to_role_mod, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


async def _seed(engine, *, tenant_id, owner_id, user_id=None, role_id=None):
    async with engine.get_async_session() as session:
        session.add(Tenant(id=tenant_id, name="acme", owner_id=owner_id))
        await session.flush()
        if user_id is not None:
            session.add(
                User(
                    id=user_id,
                    email=f"{user_id}@example.com",
                    hashed_password="x",
                    tenant_id=tenant_id,
                )
            )
        if role_id is not None:
            session.add(Role(id=role_id, name="editors", tenant_id=tenant_id))
        await session.commit()


@pytest.mark.asyncio
async def test_unknown_role_raises_role_not_found(roles_engine):
    """Previously: AttributeError on None.tenant_id from the tenant query."""
    tenant_id, owner_id, user_id = uuid4(), uuid4(), uuid4()
    await _seed(roles_engine, tenant_id=tenant_id, owner_id=owner_id, user_id=user_id)

    with pytest.raises(RoleNotFoundError):
        await add_user_to_role_mod.add_user_to_role(
            user_id=user_id, role_id=uuid4(), owner_id=owner_id
        )


@pytest.mark.asyncio
async def test_unknown_user_raises_user_not_found(roles_engine):
    """Previously: AttributeError on None.awaitable_attrs."""
    tenant_id, owner_id, role_id = uuid4(), uuid4(), uuid4()
    await _seed(roles_engine, tenant_id=tenant_id, owner_id=owner_id, role_id=role_id)

    with pytest.raises(UserNotFoundError):
        await add_user_to_role_mod.add_user_to_role(
            user_id=uuid4(), role_id=role_id, owner_id=owner_id
        )


@pytest.mark.asyncio
async def test_unknown_user_and_role_reports_the_user_first(roles_engine):
    """Guard order is user, then role — pinned so a later edit cannot swap it."""
    tenant_id, owner_id = uuid4(), uuid4()
    await _seed(roles_engine, tenant_id=tenant_id, owner_id=owner_id)

    with pytest.raises(UserNotFoundError):
        await add_user_to_role_mod.add_user_to_role(
            user_id=uuid4(), role_id=uuid4(), owner_id=owner_id
        )
