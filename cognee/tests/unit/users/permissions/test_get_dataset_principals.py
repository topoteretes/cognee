"""Reading back who has access to a dataset.

The inverse of get_principal_datasets: permissions could be granted and revoked
but never enumerated, so the UI had no way to show which teams hold access.
"""

import importlib
from uuid import uuid4

import pytest

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.permissions.methods.get_dataset_principals import (
    get_dataset_principals,
)

_module = importlib.import_module("cognee.modules.users.permissions.methods.get_dataset_principals")


class FakePermission:
    def __init__(self, name):
        self.name = name


class FakeACL:
    def __init__(self, principal, permission_name):
        self.principal = principal
        self.principal_id = principal.id
        self.permission = FakePermission(permission_name)


class FakeResult:
    def __init__(self, value):
        self._value = value

    def unique(self):
        return self

    def scalars(self):
        return self

    def all(self):
        return self._value


class FakeSession:
    def __init__(self, acls):
        self._acls = acls

    async def execute(self, _query):
        return FakeResult(self._acls)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class FakeEngine:
    def __init__(self, session):
        self._session = session

    def get_async_session(self):
        return self._session


def _patch(monkeypatch, acls, *, authorized=True):
    monkeypatch.setattr(_module, "get_relational_engine", lambda: FakeEngine(FakeSession(acls)))

    async def fake_authorize(_user_id, _permission, _dataset_ids):
        if not authorized:
            raise PermissionDeniedError(
                "Request owner does not have necessary permission: [share] for all datasets requested."
            )
        return []

    monkeypatch.setattr(_module, "get_specific_user_permission_datasets", fake_authorize)


class FakePrincipal:
    """Stands in for a User/Role/Tenant row — `type` is the discriminator column."""

    def __init__(self, kind, *, email=None, name=None):
        self.id = uuid4()
        self.type = kind
        self.email = email
        self.name = name


def _fake_user(email="member@example.com"):
    return FakePrincipal("user", email=email)


def _fake_role(name="engineering"):
    return FakePrincipal("role", name=name)


def _fake_tenant(name="acme"):
    return FakePrincipal("tenant", name=name)


@pytest.mark.asyncio
async def test_returns_each_kind_of_principal(monkeypatch):
    """Users, roles and the tenant principal are all reported, each labelled."""
    user, role, tenant = _fake_user(), _fake_role(), _fake_tenant()
    _patch(
        monkeypatch,
        [FakeACL(user, "read"), FakeACL(role, "write"), FakeACL(tenant, "read")],
    )

    result = await get_dataset_principals(uuid4(), uuid4())

    by_kind = {entry["kind"]: entry for entry in result}
    assert set(by_kind) == {"user", "role", "tenant"}
    assert by_kind["user"]["name"] == "member@example.com"
    assert by_kind["role"]["name"] == "engineering"
    assert by_kind["tenant"]["name"] == "acme"


@pytest.mark.asyncio
async def test_collects_permissions_per_principal(monkeypatch):
    """The ACL table stores a row per permission — one entry comes back, not three."""
    role = _fake_role()
    _patch(
        monkeypatch,
        [FakeACL(role, "read"), FakeACL(role, "write"), FakeACL(role, "share")],
    )

    result = await get_dataset_principals(uuid4(), uuid4())

    assert len(result) == 1
    assert result[0]["principal_id"] == str(role.id)
    assert result[0]["permissions"] == ["read", "share", "write"]


@pytest.mark.asyncio
async def test_requires_share_permission(monkeypatch):
    """A caller who cannot share the dataset cannot enumerate its principals."""
    _patch(monkeypatch, [FakeACL(_fake_user(), "read")], authorized=False)

    with pytest.raises(PermissionDeniedError):
        await get_dataset_principals(uuid4(), uuid4())


@pytest.mark.asyncio
async def test_skips_rows_with_a_missing_side(monkeypatch):
    """A dangling ACL row is dropped rather than crashing the listing."""
    dangling = FakeACL(_fake_user(), "read")
    dangling.principal = None
    _patch(monkeypatch, [dangling, FakeACL(_fake_role(), "read")])

    result = await get_dataset_principals(uuid4(), uuid4())

    assert [entry["kind"] for entry in result] == ["role"]


@pytest.mark.asyncio
async def test_returns_empty_when_nobody_has_access(monkeypatch):
    """A dataset only its owner reaches lists no shared principals."""
    _patch(monkeypatch, [])

    assert await get_dataset_principals(uuid4(), uuid4()) == []
