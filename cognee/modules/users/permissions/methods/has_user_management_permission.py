from uuid import UUID

from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.permissions.methods.get_tenant import get_tenant
from cognee.modules.users.permissions.methods.get_user_role_names_in_tenant import (
    get_user_role_names_in_tenant,
)
from cognee.modules.users.permissions.permission_types import (
    USER_MANAGEMENT_ALLOWED_ROLE_NAMES,
)


async def has_user_management_permission(requester_id: UUID, tenant_id: UUID) -> bool:
    """
    Check if requester is allowed to manage users for a tenant.

    The requester is allowed if they are the tenant owner or have one of the
    roles in USER_MANAGEMENT_ALLOWED_ROLE_NAMES (e.g. admin). Add role
    names to that set in permission_types.py to extend without changing call sites.

    Reuse this across all user management endpoints (list users, assign/remove
    roles, add/remove users from tenant, etc.) for consistent authorization.

    Args:
        requester_id: Id of the user making the request.
        tenant_id: Id of the tenant.

    Returns:
        True if the requester has permission to manage users for the tenant.

    Raises:
        PermissionDeniedError: If the requester is not authorized (not owner
            and no allowed role in this tenant).
        TenantNotFoundError: If the tenant does not exist.
    """
    tenant = await get_tenant(tenant_id)

    if tenant.owner_id == requester_id:
        return True

    role_names = await get_user_role_names_in_tenant(requester_id, tenant_id)
    if USER_MANAGEMENT_ALLOWED_ROLE_NAMES & set(role_names):
        return True

    raise PermissionDeniedError(message="User is not authorized to manage users for this tenant")
