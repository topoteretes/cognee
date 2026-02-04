import pytest
from uuid import uuid4

from cognee.modules.users.permissions.methods.has_user_management_permission import (
    has_user_management_permission,
)
from cognee.modules.users.exceptions import PermissionDeniedError


class FakeTenant:
    def __init__(self, owner_id):
        self.owner_id = owner_id


@pytest.mark.asyncio
async def test_has_user_management_permission_owner_allowed(monkeypatch):
    requester_id = uuid4()
    tenant_id = uuid4()

    async def fake_get_tenant(_):
        return FakeTenant(owner_id=requester_id)

    monkeypatch.setattr(
        "cognee.modules.users.permissions.methods.has_user_management_permission.get_tenant",
        fake_get_tenant,
    )

    result = await has_user_management_permission(requester_id, tenant_id)
    assert result is True


@pytest.mark.asyncio
async def test_has_user_management_permission_non_owner_denied(monkeypatch):
    requester_id = uuid4()
    tenant_id = uuid4()

    async def fake_get_tenant(_):
        return FakeTenant(owner_id=uuid4())

    monkeypatch.setattr(
        "cognee.modules.users.permissions.methods.has_user_management_permission.get_tenant",
        fake_get_tenant,
    )

    with pytest.raises(PermissionDeniedError):
        await has_user_management_permission(requester_id, tenant_id)
