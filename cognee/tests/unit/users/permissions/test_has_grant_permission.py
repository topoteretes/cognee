import importlib
from uuid import uuid4

import pytest

from cognee.modules.users.exceptions import CapabilityDeniedError, PermissionDeniedError
from cognee.modules.users.permissions.methods import has_grant_permission
from cognee.modules.users.permissions.permission_types import (
    GRANT_CAPABILITIES,
    MANAGE_USERS,
    REVOKE_CAPABILITIES,
)

# Module where get_effective_capabilities and get_user_role_names_in_tenant are
# looked up when has_grant_permission runs
_grant_mod = importlib.import_module(has_grant_permission.__module__)


def _resolve_to(monkeypatch, capabilities, role_names=()):
    async def fake_get_capabilities(_user_id, _tenant_id):
        return set(capabilities)

    async def fake_get_user_role_names(_user_id, _tenant_id):
        return list(role_names)

    monkeypatch.setattr(_grant_mod, "get_effective_capabilities", fake_get_capabilities)
    monkeypatch.setattr(_grant_mod, "get_user_role_names_in_tenant", fake_get_user_role_names)


@pytest.mark.asyncio
@pytest.mark.parametrize("grant_type", [MANAGE_USERS, GRANT_CAPABILITIES, REVOKE_CAPABILITIES])
async def test_holding_the_capability_allows(monkeypatch, grant_type):
    _resolve_to(monkeypatch, {grant_type})

    assert await has_grant_permission(uuid4(), uuid4(), grant_type) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "held, asked",
    [
        (MANAGE_USERS, GRANT_CAPABILITIES),
        (MANAGE_USERS, REVOKE_CAPABILITIES),
        (GRANT_CAPABILITIES, REVOKE_CAPABILITIES),
        (REVOKE_CAPABILITIES, GRANT_CAPABILITIES),
    ],
)
async def test_one_capability_does_not_carry_another(monkeypatch, held, asked):
    """Managing members, granting and revoking are separate jobs.

    If MANAGE_USERS implied GRANT_CAPABILITIES, anyone trusted to add members
    could hand themselves every other capability in the tenant.
    """
    _resolve_to(monkeypatch, {held})

    with pytest.raises(CapabilityDeniedError):
        await has_grant_permission(uuid4(), uuid4(), asked)


@pytest.mark.asyncio
@pytest.mark.parametrize("grant_type", [MANAGE_USERS, GRANT_CAPABILITIES, REVOKE_CAPABILITIES])
async def test_deprecated_admin_role_passes_every_check(monkeypatch, grant_type):
    """Until a tenant is migrated its admin role has no capability rows at all."""
    _resolve_to(monkeypatch, set(), role_names=["admin"])

    assert await has_grant_permission(uuid4(), uuid4(), grant_type) is True


@pytest.mark.asyncio
async def test_other_role_names_do_not_pass(monkeypatch):
    _resolve_to(monkeypatch, set(), role_names=["member"])

    with pytest.raises(CapabilityDeniedError):
        await has_grant_permission(uuid4(), uuid4(), GRANT_CAPABILITIES)


@pytest.mark.asyncio
async def test_denial_names_the_capability_and_is_a_permission_denied_error(monkeypatch):
    """Callers that catch PermissionDeniedError keep working, and the API body
    for user management stays what it was before the check was generalised."""
    _resolve_to(monkeypatch, set())

    with pytest.raises(PermissionDeniedError) as denied:
        await has_grant_permission(uuid4(), uuid4(), MANAGE_USERS)

    assert denied.value.message == "User is not authorized to manage users for this tenant"
    assert denied.value.name == "PermissionDeniedError"
    assert denied.value.status_code == 403
