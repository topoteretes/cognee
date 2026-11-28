from importlib import import_module
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.modules.users.models import (
    PrincipalCapability,
    PrincipalConfiguration,
    Role,
    Tenant,
    User,
    UserRole,
    UserTenant,
)
from cognee.modules.users.models.Principal import Principal
from cognee.modules.users.permissions.permission_types import MANAGE_USERS

configuration_module = import_module("cognee.modules.users.methods.get_principal_configuration")


class _Engine:
    def __init__(self, engine):
        self.sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    def get_async_session(self):
        return self.sessionmaker()


@pytest_asyncio.fixture
async def relational_engine(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    tables = [
        model.__table__
        for model in (
            Principal,
            Tenant,
            User,
            Role,
            UserRole,
            UserTenant,
            PrincipalCapability,
            PrincipalConfiguration,
        )
    ]
    async with engine.begin() as connection:
        await connection.run_sync(Principal.metadata.create_all, tables=tables)

    relational_engine = _Engine(engine)
    for module_name in (
        "cognee.modules.users.methods.get_principal_configuration",
        "cognee.modules.users.methods.get_visible_user_ids",
        "cognee.modules.users.permissions.methods.get_effective_capabilities",
        "cognee.modules.users.permissions.methods.get_tenant",
        "cognee.modules.users.permissions.methods.get_user_role_names_in_tenant",
    ):
        monkeypatch.setattr(
            import_module(module_name), "get_relational_engine", lambda: relational_engine
        )

    yield relational_engine
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "grant,allowed",
    [
        ("user", True),
        ("role", True),
        ("tenant", True),
        ("owner", True),
        ("none", False),
        ("other_tenant", False),
        ("nonmember", False),
    ],
)
async def test_configuration_tenant_manager_access(relational_engine, grant, allowed):
    tenant_id, other_tenant_id, requester_id, owner_id, config_id, role_id = (
        uuid4() for _ in range(6)
    )
    async with relational_engine.get_async_session() as session:
        session.add_all(
            [
                Tenant(
                    id=tenant_id,
                    name="tenant",
                    owner_id=requester_id if grant == "owner" else owner_id,
                ),
                Tenant(id=other_tenant_id, name="other", owner_id=uuid4()),
                User(
                    id=requester_id,
                    email="manager@example.test",
                    hashed_password="unused",
                    tenant_id=tenant_id,
                ),
                User(
                    id=owner_id,
                    email="owner@example.test",
                    hashed_password="unused",
                    tenant_id=tenant_id,
                ),
                Role(id=role_id, name="operator", tenant_id=tenant_id),
                PrincipalConfiguration(
                    id=config_id, owner_id=owner_id, name="llm", configuration={"model": "m"}
                ),
            ]
        )
        if grant != "nonmember":
            session.add(UserTenant(user_id=requester_id, tenant_id=tenant_id))
        if grant in {"user", "role", "tenant", "other_tenant", "nonmember"}:
            grant_principal_id = {
                "role": role_id,
                "tenant": tenant_id,
            }.get(grant, requester_id)
            session.add(
                PrincipalCapability(
                    principal_id=grant_principal_id,
                    tenant_id=other_tenant_id if grant == "other_tenant" else tenant_id,
                    capability=MANAGE_USERS,
                )
            )
        if grant == "role":
            session.add(UserRole(user_id=requester_id, role_id=role_id))
        await session.commit()

    result = await configuration_module.get_principal_configuration(config_id, requester_id)

    assert result == ({"model": "m"} if allowed else {})
    assert await configuration_module.get_principal_configuration(uuid4(), requester_id) == {}


@pytest.mark.asyncio
async def test_configuration_tenant_manager_cannot_read_another_tenant(relational_engine):
    tenant_id, other_tenant_id, requester_id, owner_id, config_id = (uuid4() for _ in range(5))
    async with relational_engine.get_async_session() as session:
        session.add_all(
            [
                Tenant(id=tenant_id, name="tenant", owner_id=requester_id),
                Tenant(id=other_tenant_id, name="other", owner_id=owner_id),
                User(
                    id=requester_id,
                    email="manager@example.test",
                    hashed_password="unused",
                    tenant_id=tenant_id,
                ),
                User(
                    id=owner_id,
                    email="owner@example.test",
                    hashed_password="unused",
                    tenant_id=other_tenant_id,
                ),
                PrincipalConfiguration(
                    id=config_id, owner_id=owner_id, name="llm", configuration={"model": "secret"}
                ),
            ]
        )
        await session.commit()

    result = await configuration_module.get_principal_configuration(config_id, requester_id)

    assert result == {}
