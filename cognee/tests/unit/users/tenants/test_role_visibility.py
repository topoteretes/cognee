"""Visibility rules for the role listing methods.

Role members may see the roles they belong to and who shares them, without
holding tenant-wide user management permission.
"""

import importlib
from uuid import uuid4

import pytest

from cognee.modules.users.exceptions import PermissionDeniedError, RoleNotFoundError
from cognee.modules.users.tenants.methods.get_tenant_roles import get_tenant_roles
from cognee.modules.users.tenants.methods.get_users_in_role import get_users_in_role

_roles_mod = importlib.import_module("cognee.modules.users.tenants.methods.get_tenant_roles")
_users_in_role_mod = importlib.import_module(
    "cognee.modules.users.tenants.methods.get_users_in_role"
)


class FakeUser:
    def __init__(self, user_id, email="member@example.com"):
        self.id = user_id
        self.email = email


class FakeRole:
    def __init__(self, role_id, name, users=None):
        self.id = role_id
        self.name = name
        self.users = users or []


class FakeResult:
    """Stands in for the object session.execute() returns."""

    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return self

    def all(self):
        return self._value


class FakeSession:
    """Returns queued results in order, and records the queries it was given."""

    def __init__(self, results):
        self._results = list(results)
        self.queries = []

    async def execute(self, query):
        self.queries.append(query)
        return FakeResult(self._results.pop(0))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class FakeEngine:
    def __init__(self, session):
        self._session = session

    def get_async_session(self):
        return self._session


def _patch_engine(monkeypatch, module, session):
    monkeypatch.setattr(module, "get_relational_engine", lambda: FakeEngine(session))


def _deny(monkeypatch, module):
    async def fake_has_permission(*_args, **_kwargs):
        raise PermissionDeniedError(
            message="User is not authorized to manage users for this tenant"
        )

    monkeypatch.setattr(module, "has_user_management_permission", fake_has_permission)


def _allow(monkeypatch, module):
    async def fake_has_permission(*_args, **_kwargs):
        return True

    monkeypatch.setattr(module, "has_user_management_permission", fake_has_permission)


@pytest.mark.asyncio
async def test_member_can_list_users_in_own_role(monkeypatch):
    """A role member sees the role's users without user management permission."""
    role_id, tenant_id = uuid4(), uuid4()
    requester = FakeUser(uuid4())
    peer = FakeUser(uuid4(), email="peer@example.com")

    session = FakeSession(
        [
            FakeRole(role_id, "engineering"),  # role lookup
            requester.id,  # membership lookup -> requester is a member
            [requester, peer],  # member list
        ]
    )
    _patch_engine(monkeypatch, _users_in_role_mod, session)
    _deny(monkeypatch, _users_in_role_mod)  # would raise if consulted

    result = await get_users_in_role(tenant_id=tenant_id, role_id=role_id, user=requester)

    assert result == [
        {"id": str(requester.id), "name": requester.email},
        {"id": str(peer.id), "name": peer.email},
    ]


@pytest.mark.asyncio
async def test_non_member_without_permission_is_denied(monkeypatch):
    """A non-member with no user management permission gets 403."""
    role_id, tenant_id = uuid4(), uuid4()
    requester = FakeUser(uuid4())

    session = FakeSession(
        [
            FakeRole(role_id, "engineering"),  # role lookup
            None,  # membership lookup -> not a member
        ]
    )
    _patch_engine(monkeypatch, _users_in_role_mod, session)
    _deny(monkeypatch, _users_in_role_mod)

    with pytest.raises(PermissionDeniedError):
        await get_users_in_role(tenant_id=tenant_id, role_id=role_id, user=requester)


@pytest.mark.asyncio
async def test_non_member_with_permission_is_allowed(monkeypatch):
    """An admin or owner still sees any role in the tenant."""
    role_id, tenant_id = uuid4(), uuid4()
    admin = FakeUser(uuid4(), email="admin@example.com")
    member = FakeUser(uuid4(), email="member@example.com")

    session = FakeSession(
        [
            FakeRole(role_id, "engineering"),
            None,  # admin is not a member of the role
            [member],
        ]
    )
    _patch_engine(monkeypatch, _users_in_role_mod, session)
    _allow(monkeypatch, _users_in_role_mod)

    result = await get_users_in_role(tenant_id=tenant_id, role_id=role_id, user=admin)

    assert result == [{"id": str(member.id), "name": member.email}]


@pytest.mark.asyncio
async def test_role_from_another_tenant_is_not_found(monkeypatch):
    """A role id that does not belong to the tenant raises 404, not a member list."""
    role_id, tenant_id = uuid4(), uuid4()
    requester = FakeUser(uuid4())

    session = FakeSession([None])  # role lookup scoped by tenant finds nothing
    _patch_engine(monkeypatch, _users_in_role_mod, session)
    _allow(monkeypatch, _users_in_role_mod)

    with pytest.raises(RoleNotFoundError):
        await get_users_in_role(tenant_id=tenant_id, role_id=role_id, user=requester)


@pytest.mark.asyncio
async def test_member_lists_only_own_roles(monkeypatch):
    """Without user management permission, the roles listing is filtered to the caller."""
    tenant_id = uuid4()
    requester = FakeUser(uuid4())
    own_role = FakeRole(uuid4(), "engineering", users=[requester])

    session = FakeSession([[own_role]])
    _patch_engine(monkeypatch, _roles_mod, session)
    _deny(monkeypatch, _roles_mod)

    result = await get_tenant_roles(tenant_id=tenant_id, user=requester)

    assert result == [
        {
            "id": str(own_role.id),
            "name": "engineering",
            "description": None,
            "user_count": 1,
        }
    ]
    # The filtered path joins user_roles; the unfiltered one does not.
    assert "user_roles" in str(session.queries[0])


@pytest.mark.asyncio
async def test_admin_lists_all_tenant_roles(monkeypatch):
    """With user management permission, no membership filter is applied."""
    tenant_id = uuid4()
    admin = FakeUser(uuid4(), email="admin@example.com")
    roles = [FakeRole(uuid4(), "engineering"), FakeRole(uuid4(), "support")]

    session = FakeSession([roles])
    _patch_engine(monkeypatch, _roles_mod, session)
    _allow(monkeypatch, _roles_mod)

    result = await get_tenant_roles(tenant_id=tenant_id, user=admin)

    assert [entry["name"] for entry in result] == ["engineering", "support"]
    assert "user_roles" not in str(session.queries[0])
