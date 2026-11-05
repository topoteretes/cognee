"""Exercise tenant role lookups and authorization against an isolated SQLite database."""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.infrastructure.databases.relational import Base
from cognee.modules.users.exceptions import PermissionDeniedError, UserNotFoundError
from cognee.modules.users.models import (
    PrincipalCapability,
    Role,
    Tenant,
    User,
    UserRole,
    UserTenant,
)
from cognee.modules.users.models.Principal import Principal
from cognee.modules.users.tenants.methods.get_user_roles import get_user_roles


@pytest_asyncio.fixture
async def role_database(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    owner_id, user_id, tenant_id, other_tenant_id, role_id, other_role_id = (
        uuid4() for _ in range(6)
    )
    owner = User(id=owner_id, email="owner@example.test", hashed_password="unused")
    target = User(id=user_id, email="target@example.test", hashed_password="unused")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(
                    sync_connection,
                    tables=[
                        model.__table__
                        for model in (
                            Principal,
                            User,
                            Tenant,
                            Role,
                            UserRole,
                            UserTenant,
                            PrincipalCapability,
                        )
                    ],
                )
            )
        async with sessions() as session:
            session.add_all(
                [
                    owner,
                    target,
                    Tenant(id=tenant_id, name="Authorized tenant", owner_id=owner_id),
                    Tenant(id=other_tenant_id, name="Other tenant", owner_id=uuid4()),
                    Role(id=role_id, name="member", tenant_id=tenant_id),
                    Role(id=other_role_id, name="other-member", tenant_id=other_tenant_id),
                ]
            )
            await session.flush()
            session.add_all(
                [
                    UserTenant(user_id=owner_id, tenant_id=tenant_id),
                    UserTenant(user_id=user_id, tenant_id=other_tenant_id),
                ]
            )
            await session.commit()

        adapter = SimpleNamespace(get_async_session=sessions)
        # Keep the real permission checks; only redirect database access to SQLite.
        for module_path in (
            "cognee.modules.users.tenants.methods.get_user_roles",
            "cognee.modules.users.permissions.methods.get_tenant",
            "cognee.modules.users.permissions.methods.get_effective_capabilities",
            "cognee.modules.users.permissions.methods.get_user_role_names_in_tenant",
        ):
            monkeypatch.setattr(
                importlib.import_module(module_path), "get_relational_engine", lambda: adapter
            )

        yield SimpleNamespace(
            sessions=sessions,
            owner=owner,
            target=target,
            tenant_id=tenant_id,
            role_id=role_id,
            other_role_id=other_role_id,
        )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "in_tenant,own_role,other_role",
    [
        pytest.param(True, True, True, id="shared-user-roles-stay-in-tenant"),
        pytest.param(True, False, True, id="only-foreign-roles-returns-empty"),
        pytest.param(True, False, False, id="member-without-roles-returns-empty"),
        pytest.param(False, False, True, id="foreign-user-is-not-found"),
    ],
)
async def test_get_user_roles_is_tenant_scoped(role_database, in_tenant, own_role, other_role):
    db = role_database
    async with db.sessions() as session:
        if in_tenant:
            session.add(UserTenant(user_id=db.target.id, tenant_id=db.tenant_id))
        if own_role:
            session.add(UserRole(user_id=db.target.id, role_id=db.role_id))
        if other_role:
            session.add(UserRole(user_id=db.target.id, role_id=db.other_role_id))
        await session.commit()

    if not in_tenant:
        with pytest.raises(UserNotFoundError):
            await get_user_roles(db.tenant_id, db.target.id, db.owner)
    else:
        result = await get_user_roles(db.tenant_id, db.target.id, db.owner)
        assert result == ([{"id": str(db.role_id), "name": "member"}] if own_role else [])


@pytest.mark.asyncio
async def test_get_user_roles_rejects_unknown_user(role_database):
    db = role_database
    with pytest.raises(UserNotFoundError):
        await get_user_roles(db.tenant_id, uuid4(), db.owner)


@pytest.mark.asyncio
async def test_get_user_roles_rejects_unauthorized_requester(role_database):
    db = role_database
    with pytest.raises(PermissionDeniedError):
        await get_user_roles(db.tenant_id, db.owner.id, db.target)
