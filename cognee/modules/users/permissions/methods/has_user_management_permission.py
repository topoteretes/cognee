from uuid import UUID

from cognee.modules.users.permissions.methods.get_tenant import get_tenant
from cognee.modules.users.exceptions import PermissionDeniedError


async def has_user_management_permission(requester_id: UUID, tenant_id: UUID) -> bool:
    """
    Check if requester is allowed to manage users for a tenant.
    Args:
        requester_id: Id of the user making the request
        tenant_id: Id of the tenant

    Returns:
        True if requester has permission

    Raises:
        PermissionDeniedError: If requester is not authorized.
        TenantNotFoundError: If the tenant does not exist.
    """
    tenant = await get_tenant(tenant_id)

    # TODO: extend to support admin roles
    if tenant.owner_id != requester_id:
        raise PermissionDeniedError(
            message="User is not authorized to manage users for this tenant"
        )

    return True
