import importlib
from uuid import uuid4

import pytest

from cognee.modules.users.exceptions import PermissionDeniedError, RoleNotFoundError
from cognee.modules.users.permissions.methods import authorized_get_principal_datasets

# Module where the collaborators are looked up when the method runs.
_mod = importlib.import_module(authorized_get_principal_datasets.__module__)


class FakePrincipal:
    def __init__(self, principal_type, principal_id=None, tenant_id=None, name=None):
        self.type = principal_type
        self.id = principal_id or uuid4()
        self.tenant_id = tenant_id
        self.name = name


class FakeDataset:
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id


class FakeUser:
    def __init__(self, user_id, tenant_id):
        self.id = user_id
        self.tenant_id = tenant_id


def _patch(
    monkeypatch, requester_tenant_id, principal, datasets=(), role_names=(), may_manage=False
):
    """Wire the method's collaborators. get_user carries the requester's current
    tenant, and has_user_management_permission mirrors the real one: True when
    allowed, PermissionDeniedError when not."""

    async def fake_get_user(user_id):
        return FakeUser(user_id, requester_tenant_id)

    async def fake_get_principal(_):
        return principal

    async def fake_get_principal_datasets(_principal, _permission_name):
        return list(datasets)

    async def fake_get_user_role_names_in_tenant(_user_id, _tenant_id):
        return list(role_names)

    async def fake_has_user_management_permission(_requester_id, _tenant_id):
        if may_manage:
            return True
        raise PermissionDeniedError(message="not authorized")

    monkeypatch.setattr(_mod, "get_user", fake_get_user)
    monkeypatch.setattr(_mod, "get_principal", fake_get_principal)
    monkeypatch.setattr(_mod, "get_principal_datasets", fake_get_principal_datasets)
    monkeypatch.setattr(_mod, "get_all_user_permission_datasets", fake_get_principal_datasets)
    monkeypatch.setattr(_mod, "get_user_role_names_in_tenant", fake_get_user_role_names_in_tenant)
    monkeypatch.setattr(_mod, "has_user_management_permission", fake_has_user_management_permission)


@pytest.mark.asyncio
async def test_user_may_read_their_own_datasets(monkeypatch):
    """No user-management permission needed to ask about yourself."""
    requester_id, tenant_id = uuid4(), uuid4()
    _patch(
        monkeypatch,
        tenant_id,
        FakePrincipal("user", principal_id=requester_id),
        datasets=[FakeDataset(tenant_id)],
        may_manage=False,
    )

    datasets = await authorized_get_principal_datasets(requester_id, "read", requester_id)
    assert len(datasets) == 1


@pytest.mark.asyncio
async def test_another_users_datasets_need_user_management(monkeypatch):
    requester_id, tenant_id = uuid4(), uuid4()
    other = FakePrincipal("user")
    _patch(monkeypatch, tenant_id, other, datasets=[FakeDataset(tenant_id)], may_manage=False)

    with pytest.raises(PermissionDeniedError):
        await authorized_get_principal_datasets(other.id, "read", requester_id)


@pytest.mark.asyncio
async def test_user_manager_may_read_another_users_datasets(monkeypatch):
    requester_id, tenant_id = uuid4(), uuid4()
    other = FakePrincipal("user")
    _patch(monkeypatch, tenant_id, other, datasets=[FakeDataset(tenant_id)], may_manage=True)

    datasets = await authorized_get_principal_datasets(other.id, "read", requester_id)
    assert len(datasets) == 1


@pytest.mark.asyncio
async def test_role_member_may_read_their_teams_datasets(monkeypatch):
    """A member needs no user-management permission for their own role."""
    requester_id, tenant_id = uuid4(), uuid4()
    role = FakePrincipal("role", tenant_id=tenant_id, name="engineering")
    _patch(
        monkeypatch,
        tenant_id,
        role,
        datasets=[FakeDataset(tenant_id)],
        role_names=["engineering"],
        may_manage=False,
    )

    datasets = await authorized_get_principal_datasets(role.id, "read", requester_id)
    assert len(datasets) == 1


@pytest.mark.asyncio
async def test_non_member_without_user_management_is_denied(monkeypatch):
    requester_id, tenant_id = uuid4(), uuid4()
    role = FakePrincipal("role", tenant_id=tenant_id, name="engineering")
    _patch(monkeypatch, tenant_id, role, role_names=["design"], may_manage=False)

    with pytest.raises(PermissionDeniedError):
        await authorized_get_principal_datasets(role.id, "read", requester_id)


@pytest.mark.asyncio
async def test_user_manager_may_read_any_role_in_the_tenant(monkeypatch):
    requester_id, tenant_id = uuid4(), uuid4()
    role = FakePrincipal("role", tenant_id=tenant_id, name="engineering")
    _patch(
        monkeypatch,
        tenant_id,
        role,
        datasets=[FakeDataset(tenant_id)],
        role_names=["design"],
        may_manage=True,
    )

    datasets = await authorized_get_principal_datasets(role.id, "read", requester_id)
    assert len(datasets) == 1


@pytest.mark.asyncio
async def test_role_of_another_tenant_is_not_found(monkeypatch):
    """Reported as missing, not forbidden — a foreign role id must not be
    distinguishable from one that does not exist."""
    requester_id, tenant_id = uuid4(), uuid4()
    foreign_role = FakePrincipal("role", tenant_id=uuid4(), name="engineering")
    # Even a user manager who is somehow "a member" cannot cross the boundary.
    _patch(monkeypatch, tenant_id, foreign_role, role_names=["engineering"], may_manage=True)

    with pytest.raises(RoleNotFoundError):
        await authorized_get_principal_datasets(foreign_role.id, "read", requester_id)


@pytest.mark.asyncio
async def test_tenant_datasets_readable_only_from_inside_that_tenant(monkeypatch):
    requester_id, tenant_id = uuid4(), uuid4()
    _patch(
        monkeypatch,
        tenant_id,
        FakePrincipal("tenant", principal_id=tenant_id),
        datasets=[FakeDataset(tenant_id)],
    )

    datasets = await authorized_get_principal_datasets(tenant_id, "read", requester_id)
    assert len(datasets) == 1


@pytest.mark.asyncio
async def test_another_tenants_datasets_are_denied(monkeypatch):
    requester_id, tenant_id = uuid4(), uuid4()
    other_tenant = FakePrincipal("tenant")
    _patch(monkeypatch, tenant_id, other_tenant, datasets=[FakeDataset(tenant_id)], may_manage=True)

    with pytest.raises(PermissionDeniedError):
        await authorized_get_principal_datasets(other_tenant.id, "read", requester_id)


@pytest.mark.asyncio
async def test_datasets_outside_the_callers_tenant_are_filtered_out(monkeypatch):
    """A principal spanning tenants must not disclose another tenant's datasets."""
    requester_id, tenant_id = uuid4(), uuid4()
    mine, theirs = FakeDataset(tenant_id), FakeDataset(uuid4())
    _patch(
        monkeypatch,
        tenant_id,
        FakePrincipal("user", principal_id=requester_id),
        datasets=[mine, theirs],
    )

    datasets = await authorized_get_principal_datasets(requester_id, "read", requester_id)
    assert datasets == [mine]


@pytest.mark.asyncio
async def test_tenant_comes_from_the_requester_not_the_caller(monkeypatch):
    """The requester's own tenant decides scope — there is no argument a caller
    could use to name a different one."""
    requester_id = uuid4()
    requester_tenant, foreign_tenant = uuid4(), uuid4()
    role_in_foreign_tenant = FakePrincipal("role", tenant_id=foreign_tenant, name="engineering")
    _patch(
        monkeypatch,
        requester_tenant,
        role_in_foreign_tenant,
        datasets=[FakeDataset(foreign_tenant)],
        role_names=["engineering"],
        may_manage=True,
    )

    with pytest.raises(RoleNotFoundError):
        await authorized_get_principal_datasets(role_in_foreign_tenant.id, "read", requester_id)


@pytest.mark.asyncio
async def test_unsupported_principal_type_is_denied(monkeypatch):
    requester_id, tenant_id = uuid4(), uuid4()
    _patch(monkeypatch, tenant_id, FakePrincipal("principal"), may_manage=True)

    with pytest.raises(PermissionDeniedError):
        await authorized_get_principal_datasets(uuid4(), "read", requester_id)
