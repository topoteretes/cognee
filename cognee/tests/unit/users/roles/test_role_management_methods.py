import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.users.roles.methods.delete_role import delete_role
from cognee.modules.users.roles.methods.remove_user_from_role import remove_user_from_role

_delete_role_mod = importlib.import_module("cognee.modules.users.roles.methods.delete_role")
_remove_user_mod = importlib.import_module(
    "cognee.modules.users.roles.methods.remove_user_from_role"
)


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalars(self):
        return self

    def first(self):
        return self.value


class FakeSession:
    def __init__(self, select_values):
        self.select_values = list(select_values)
        self.statements = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def execute(self, statement):
        self.statements.append(statement)
        if getattr(statement, "is_delete", False):
            return FakeResult(None)
        return FakeResult(self.select_values.pop(0))

    async def commit(self):
        self.committed = True


class FakeEngine:
    def __init__(self, session):
        self.session = session

    def get_async_session(self):
        return self.session


def get_delete_tables(session):
    return [
        statement.table.name
        for statement in session.statements
        if getattr(statement, "is_delete", False)
    ]


@pytest.mark.asyncio
async def test_delete_role_uses_user_management_permission_and_deletes_principal(monkeypatch):
    role_id = uuid4()
    tenant_id = uuid4()
    requester_id = uuid4()
    session = FakeSession([SimpleNamespace(tenant_id=tenant_id)])
    permission_calls = []

    async def fake_has_user_management_permission(*, requester_id, tenant_id):
        permission_calls.append((requester_id, tenant_id))
        return True

    monkeypatch.setattr(_delete_role_mod, "get_relational_engine", lambda: FakeEngine(session))
    monkeypatch.setattr(
        _delete_role_mod,
        "has_user_management_permission",
        fake_has_user_management_permission,
    )

    await delete_role(role_id=role_id, owner_id=requester_id)

    assert permission_calls == [(requester_id, tenant_id)]
    assert get_delete_tables(session) == ["user_roles", "acls", "roles", "principals"]
    assert session.committed is True


@pytest.mark.asyncio
async def test_remove_user_from_role_uses_user_management_permission(monkeypatch):
    user_id = uuid4()
    role_id = uuid4()
    tenant_id = uuid4()
    requester_id = uuid4()
    session = FakeSession([SimpleNamespace(), SimpleNamespace(tenant_id=tenant_id)])
    permission_calls = []

    async def fake_has_user_management_permission(*, requester_id, tenant_id):
        permission_calls.append((requester_id, tenant_id))
        return True

    monkeypatch.setattr(_remove_user_mod, "get_relational_engine", lambda: FakeEngine(session))
    monkeypatch.setattr(
        _remove_user_mod,
        "has_user_management_permission",
        fake_has_user_management_permission,
    )

    await remove_user_from_role(user_id=user_id, role_id=role_id, owner_id=requester_id)

    assert permission_calls == [(requester_id, tenant_id)]
    assert get_delete_tables(session) == ["user_roles"]
    assert session.committed is True
