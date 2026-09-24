from uuid import UUID

from cognee.modules.users.exceptions import CapabilityDeniedError
from cognee.modules.users.permissions.methods.get_effective_capabilities import (
    get_effective_capabilities,
)
from cognee.modules.users.permissions.methods.get_user_role_names_in_tenant import (
    get_user_role_names_in_tenant,
)
from cognee.modules.users.permissions.permission_types import USER_MANAGEMENT_ALLOWED_ROLE_NAMES


async def has_grant_permission(requester_id: UUID, tenant_id: UUID, grant_type: str) -> bool:
    """
    Check if requester holds a capability in a tenant.

    The requester is allowed if the capability was granted to the tenant, to
    one of their roles in it, or to them personally in it. The tenant owner
    holds every capability, so they always pass.

    Reuse this for every operation a capability gates, so each one is
    authorized the same way. has_user_management_permission is this check for
    MANAGE_USERS.

    Args:
        requester_id: Id of the user making the request.
        tenant_id: Id of the tenant.
        grant_type: Capability the operation needs, from CAPABILITY_TYPES.

    Returns:
        True if the requester holds the capability in the tenant.

    Raises:
        CapabilityDeniedError: If the requester does not hold it.
        TenantNotFoundError: If the tenant does not exist.
    """
    capabilities = await get_effective_capabilities(requester_id, tenant_id)
    if grant_type in capabilities:
        return True

    # Deprecated path: tenants upgrading from the role-name check would otherwise
    # lock their "admin" role out until it is granted the capability.
    role_names = await get_user_role_names_in_tenant(requester_id, tenant_id)
    if USER_MANAGEMENT_ALLOWED_ROLE_NAMES & set(role_names):
        return True

    raise CapabilityDeniedError(grant_type)
