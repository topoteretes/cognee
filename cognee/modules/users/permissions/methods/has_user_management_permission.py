from uuid import UUID

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.permissions.methods.get_effective_capabilities import (
    get_effective_capabilities,
)
from cognee.modules.users.permissions.methods.get_user_role_names_in_tenant import (
    get_user_role_names_in_tenant,
)
from cognee.modules.users.permissions.permission_types import (
    MANAGE_USERS,
    USER_MANAGEMENT_ALLOWED_ROLE_NAMES,
)


async def has_user_management_permission(requester_id: UUID, tenant_id: UUID) -> bool:
    """
    Check if requester is allowed to manage users for a tenant.

    The requester is allowed if they hold the MANAGE_USERS capability in this
    tenant, granted either to the tenant or to one of their roles in it. The
    tenant owner holds every capability, so they always pass.

    Reuse this across all user management endpoints (list users, assign/remove
    roles, add/remove users from tenant, etc.) for consistent authorization.

    Args:
        requester_id: Id of the user making the request.
        tenant_id: Id of the tenant.

    Returns:
        True if the requester has permission to manage users for the tenant.

    Raises:
        PermissionDeniedError: If the requester is not authorized.
        TenantNotFoundError: If the tenant does not exist.
    """
    capabilities = await get_effective_capabilities(requester_id, tenant_id)
    if MANAGE_USERS in capabilities:
        return True

    # Deprecated path: tenants upgrading from the role-name check would otherwise
    # lose user management until their "admin" role is granted the capability.
    role_names = await get_user_role_names_in_tenant(requester_id, tenant_id)
    if USER_MANAGEMENT_ALLOWED_ROLE_NAMES & set(role_names):
        return True

    raise PermissionDeniedError(message="User is not authorized to manage users for this tenant")
