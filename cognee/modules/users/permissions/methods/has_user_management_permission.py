from uuid import UUID

from cognee.modules.users.permissions.methods.has_grant_permission import has_grant_permission
from cognee.modules.users.permissions.permission_types import MANAGE_USERS


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
        CapabilityDeniedError: If the requester is not authorized. It is a
            PermissionDeniedError, so callers catching that keep working.
        TenantNotFoundError: If the tenant does not exist.
    """
    return await has_grant_permission(requester_id, tenant_id, MANAGE_USERS)
