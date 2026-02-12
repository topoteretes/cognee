import importlib
from uuid import uuid4

import pytest

from cognee.modules.users.exceptions import PermissionDeniedError, TenantNotFoundError
from cognee.modules.users.permissions.methods import has_user_management_permission

# Module where get_tenant is looked up when has_user_management_permission runs
_perm_mod = importlib.import_module(has_user_management_permission.__module__)


class FakeTenant:
    def __init__(self, owner_id):
        self.owner_id = owner_id


@pytest.mark.asyncio
async def test_has_user_management_permission_owner_allowed(monkeypatch):
    requester_id = uuid4()
    tenant_id = uuid4()

    async def fake_get_tenant(_):
        return FakeTenant(owner_id=requester_id)

    monkeypatch.setattr(_perm_mod, "get_tenant", fake_get_tenant)

    result = await has_user_management_permission(requester_id, tenant_id)
    assert result is True


@pytest.mark.asyncio
async def test_has_user_management_permission_non_owner_denied(monkeypatch):
    requester_id = uuid4()
    tenant_id = uuid4()

    async def fake_get_tenant(_):
        return FakeTenant(owner_id=uuid4())

    async def fake_get_user_role_names(_user_id, _tenant_id):
        return []  # no allowed roles

    monkeypatch.setattr(_perm_mod, "get_tenant", fake_get_tenant)
    monkeypatch.setattr(_perm_mod, "get_user_role_names_in_tenant", fake_get_user_role_names)

    with pytest.raises(PermissionDeniedError):
        await has_user_management_permission(requester_id, tenant_id)


@pytest.mark.asyncio
async def test_has_user_management_permission_allowed_role(monkeypatch):
    """Requester is not owner but has an allowed role (e.g. tenant_admin)."""
    requester_id = uuid4()
    tenant_id = uuid4()

    async def fake_get_tenant(_):
        return FakeTenant(owner_id=uuid4())  # different from requester_id

    async def fake_get_user_role_names(_user_id, _tenant_id):
        return ["admin"]

    monkeypatch.setattr(_perm_mod, "get_tenant", fake_get_tenant)
    monkeypatch.setattr(_perm_mod, "get_user_role_names_in_tenant", fake_get_user_role_names)

    result = await has_user_management_permission(requester_id, tenant_id)
    assert result is True


@pytest.mark.asyncio
async def test_has_user_management_permission_non_owner_wrong_role_denied(monkeypatch):
    """Requester is not owner and has only roles not in the allowed set."""
    requester_id = uuid4()
    tenant_id = uuid4()

    async def fake_get_tenant(_):
        return FakeTenant(owner_id=uuid4())

    async def fake_get_user_role_names(_user_id, _tenant_id):
        return ["member"]  # not in USER_MANAGEMENT_ALLOWED_ROLE_NAMES

    monkeypatch.setattr(_perm_mod, "get_tenant", fake_get_tenant)
    monkeypatch.setattr(_perm_mod, "get_user_role_names_in_tenant", fake_get_user_role_names)

    with pytest.raises(PermissionDeniedError):
        await has_user_management_permission(requester_id, tenant_id)


@pytest.mark.asyncio
async def test_has_user_management_permission_tenant_not_found(monkeypatch):
    requester_id = uuid4()
    tenant_id = uuid4()

    async def fake_get_tenant(_):
        raise TenantNotFoundError(message="Could not find tenant")

    monkeypatch.setattr(_perm_mod, "get_tenant", fake_get_tenant)

    with pytest.raises(TenantNotFoundError):
        await has_user_management_permission(requester_id, tenant_id)
