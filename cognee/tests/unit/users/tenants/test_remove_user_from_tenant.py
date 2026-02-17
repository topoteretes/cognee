import importlib
from uuid import uuid4

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.tenants.methods.remove_user_from_tenant import (
    remove_user_from_tenant,
)

# Module where remove_user_from_tenant is defined (for patching its imports)
_remove_user_mod = importlib.import_module(
    "cognee.modules.users.tenants.methods.remove_user_from_tenant"
)


class FakeTenant:
    def __init__(self, owner_id):
        self.owner_id = owner_id


@pytest.mark.asyncio
async def test_remove_user_from_tenant_permission_denied(monkeypatch):
    """Requester who is not tenant owner gets PermissionDeniedError (403)."""
    owner_id = uuid4()
    tenant_id = uuid4()
    user_id = uuid4()

    async def fake_has_permission(*, requester_id, tenant_id):  # noqa: ARG001
        raise PermissionDeniedError(
            message="User is not authorized to manage users for this tenant"
        )

    monkeypatch.setattr(
        _remove_user_mod,
        "has_user_management_permission",
        fake_has_permission,
    )

    with pytest.raises(PermissionDeniedError):
        await remove_user_from_tenant(user_id=user_id, tenant_id=tenant_id, owner_id=owner_id)


@pytest.mark.asyncio
async def test_remove_user_from_tenant_cannot_remove_owner(monkeypatch):
    """Removing the tenant owner from their own tenant raises 400."""
    owner_id = uuid4()
    tenant_id = uuid4()
    user_id = owner_id  # same as owner

    async def fake_has_permission(*, requester_id, tenant_id):  # noqa: ARG001
        return True

    async def fake_get_tenant(_):
        return FakeTenant(owner_id=owner_id)

    monkeypatch.setattr(
        _remove_user_mod,
        "has_user_management_permission",
        fake_has_permission,
    )
    monkeypatch.setattr(_remove_user_mod, "get_tenant", fake_get_tenant)

    with pytest.raises(CogneeValidationError) as exc_info:
        await remove_user_from_tenant(user_id=user_id, tenant_id=tenant_id, owner_id=owner_id)

    assert exc_info.value.status_code == 400
    assert "Cannot remove the tenant owner" in exc_info.value.message


@pytest.mark.asyncio
async def test_remove_user_from_tenant_user_not_found(monkeypatch):
    """Removing a non-existent user propagates EntityNotFoundError (404)."""
    owner_id = uuid4()
    tenant_id = uuid4()
    user_id = uuid4()

    async def fake_has_permission(*, requester_id, tenant_id):  # noqa: ARG001
        return True

    async def fake_get_tenant(_):
        return FakeTenant(owner_id=owner_id)

    async def fake_get_user(_):  # noqa: ARG001
        raise EntityNotFoundError(message=f"Could not find user: {user_id}")

    monkeypatch.setattr(
        _remove_user_mod,
        "has_user_management_permission",
        fake_has_permission,
    )
    monkeypatch.setattr(_remove_user_mod, "get_tenant", fake_get_tenant)
    monkeypatch.setattr(_remove_user_mod, "get_user", fake_get_user)

    with pytest.raises(EntityNotFoundError) as exc_info:
        await remove_user_from_tenant(user_id=user_id, tenant_id=tenant_id, owner_id=owner_id)

    assert "Could not find user" in exc_info.value.message
