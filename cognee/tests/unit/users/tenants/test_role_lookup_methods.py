import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.users.exceptions import RoleNotFoundError, UserNotFoundError
from cognee.modules.users.tenants.methods.get_user_roles import get_user_roles
from cognee.modules.users.tenants.methods.get_users_in_role import get_users_in_role

_get_users_in_role_mod = importlib.import_module(
    "cognee.modules.users.tenants.methods.get_users_in_role"
)
_get_user_roles_mod = importlib.import_module("cognee.modules.users.tenants.methods.get_user_roles")


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalars(self):
        return self

    def first(self):
        return self.value

    def all(self):
        return self.value


class FakeSession:
    def __init__(self, values):
        self.values = list(values)
        self.statements = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def execute(self, statement):
        self.statements.append(statement)
        return FakeResult(self.values.pop(0))


class FakeEngine:
    def __init__(self, session):
        self.session = session

    def get_async_session(self):
        return self.session


def statement_sql(session, index):
    return str(session.statements[index]).lower()


@pytest.mark.asyncio
async def test_get_users_in_role_filters_role_and_users_by_tenant(monkeypatch):
    tenant_id = uuid4()
    role_id = uuid4()
    requester_id = uuid4()
    target_user_id = uuid4()
    target_user = SimpleNamespace(id=target_user_id, email="member@example.com")
    session = FakeSession([SimpleNamespace(id=role_id), [target_user]])
    permission_calls = []

    async def fake_has_user_management_permission(user_id, checked_tenant_id):
        permission_calls.append((user_id, checked_tenant_id))
        return True

    monkeypatch.setattr(
        _get_users_in_role_mod, "get_relational_engine", lambda: FakeEngine(session)
    )
    monkeypatch.setattr(
        _get_users_in_role_mod,
        "has_user_management_permission",
        fake_has_user_management_permission,
    )

    result = await get_users_in_role(
        tenant_id=tenant_id, role_id=role_id, user=SimpleNamespace(id=requester_id)
    )

    assert result == [{"id": str(target_user_id), "name": "member@example.com"}]
    assert permission_calls == [(requester_id, tenant_id)]
    assert "roles.tenant_id" in statement_sql(session, 0)
    assert "user_roles.role_id" in statement_sql(session, 1)
    assert "user_tenants.tenant_id" in statement_sql(session, 1)


@pytest.mark.asyncio
async def test_get_users_in_role_raises_for_role_outside_tenant(monkeypatch):
    tenant_id = uuid4()
    role_id = uuid4()
    session = FakeSession([None])

    async def fake_has_user_management_permission(user_id, checked_tenant_id):  # noqa: ARG001
        return True

    monkeypatch.setattr(
        _get_users_in_role_mod, "get_relational_engine", lambda: FakeEngine(session)
    )
    monkeypatch.setattr(
        _get_users_in_role_mod,
        "has_user_management_permission",
        fake_has_user_management_permission,
    )

    with pytest.raises(RoleNotFoundError):
        await get_users_in_role(
            tenant_id=tenant_id, role_id=role_id, user=SimpleNamespace(id=uuid4())
        )

    assert len(session.statements) == 1
    assert "roles.tenant_id" in statement_sql(session, 0)


@pytest.mark.asyncio
async def test_get_user_roles_requires_user_membership_and_filters_roles_by_tenant(monkeypatch):
    tenant_id = uuid4()
    requester_id = uuid4()
    target_user_id = uuid4()
    role_id = uuid4()
    role = SimpleNamespace(id=role_id, name="admin")
    session = FakeSession([SimpleNamespace(user_id=target_user_id), [role]])
    permission_calls = []

    async def fake_has_user_management_permission(user_id, checked_tenant_id):
        permission_calls.append((user_id, checked_tenant_id))
        return True

    monkeypatch.setattr(_get_user_roles_mod, "get_relational_engine", lambda: FakeEngine(session))
    monkeypatch.setattr(
        _get_user_roles_mod,
        "has_user_management_permission",
        fake_has_user_management_permission,
    )

    result = await get_user_roles(
        tenant_id=tenant_id, user_id=target_user_id, user=SimpleNamespace(id=requester_id)
    )

    assert result == [{"id": str(role_id), "name": "admin"}]
    assert permission_calls == [(requester_id, tenant_id)]
    assert "user_tenants.tenant_id" in statement_sql(session, 0)
    assert "roles.tenant_id" in statement_sql(session, 1)


@pytest.mark.asyncio
async def test_get_user_roles_raises_for_user_outside_tenant(monkeypatch):
    tenant_id = uuid4()
    target_user_id = uuid4()
    session = FakeSession([None])

    async def fake_has_user_management_permission(user_id, checked_tenant_id):  # noqa: ARG001
        return True

    monkeypatch.setattr(_get_user_roles_mod, "get_relational_engine", lambda: FakeEngine(session))
    monkeypatch.setattr(
        _get_user_roles_mod,
        "has_user_management_permission",
        fake_has_user_management_permission,
    )

    with pytest.raises(UserNotFoundError):
        await get_user_roles(
            tenant_id=tenant_id, user_id=target_user_id, user=SimpleNamespace(id=uuid4())
        )

    assert len(session.statements) == 1
    assert "user_tenants.tenant_id" in statement_sql(session, 0)
