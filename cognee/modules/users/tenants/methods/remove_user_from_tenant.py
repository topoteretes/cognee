from uuid import UUID

from fastapi import status
from sqlalchemy import delete, select

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.methods import get_user
from cognee.modules.users.models.ACL import ACL
from cognee.modules.users.models.Role import Role
from cognee.modules.users.models.UserRole import UserRole
from cognee.modules.users.models.UserTenant import UserTenant
from cognee.modules.users.permissions.methods import get_tenant, has_user_management_permission


async def remove_user_from_tenant(user_id: UUID, tenant_id: UUID, owner_id: UUID) -> None:
    """
    Remove a user from a tenant.

    Only the tenant owner can remove users. The tenant owner cannot be removed
    from their own tenant. Removes the user from all roles in the tenant and
    revokes their permissions on datasets belonging to the tenant. Data owned
    by the removed user within the tenant (e.g. datasets they created) remains
    in the tenant; only their membership and direct permissions are removed.

    Args:
        user_id: Id of the user to remove.
        tenant_id: Id of the tenant.
        owner_id: Id of the requester (must be tenant owner).

    Raises:
        PermissionDeniedError: If requester is not the tenant owner.
        TenantNotFoundError: If the tenant does not exist.
        EntityNotFoundError: If the user does not exist or is not in the tenant.
        CogneeValidationError: If attempting to remove the tenant owner.
    """
    await has_user_management_permission(requester_id=owner_id, tenant_id=tenant_id)
    tenant = await get_tenant(tenant_id)

    if tenant.owner_id == user_id:
        raise CogneeValidationError(
            message="Cannot remove the tenant owner from their own tenant.",
            name="CogneeValidationError",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    await get_user(user_id)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # Check user is in tenant
        user_tenant_result = await session.execute(
            select(UserTenant).where(
                UserTenant.user_id == user_id,
                UserTenant.tenant_id == tenant_id,
            )
        )
        if user_tenant_result.scalars().first() is None:
            raise EntityNotFoundError(message="User not found in this tenant.")

        # Subquery for role ids in this tenant
        role_ids_in_tenant = select(Role.id).where(Role.tenant_id == tenant_id)
        # Remove user from all roles in this tenant
        await session.execute(
            delete(UserRole).where(
                UserRole.user_id == user_id,
                UserRole.role_id.in_(role_ids_in_tenant),
            )
        )

        # Subquery for dataset ids in this tenant
        dataset_ids_in_tenant = select(Dataset.id).where(Dataset.tenant_id == tenant_id)
        # Revoke user's permissions on datasets in this tenant
        await session.execute(
            delete(ACL).where(
                ACL.principal_id == user_id,
                ACL.dataset_id.in_(dataset_ids_in_tenant),
            )
        )

        # Remove user–tenant association
        await session.execute(
            delete(UserTenant).where(
                UserTenant.user_id == user_id,
                UserTenant.tenant_id == tenant_id,
            )
        )

        await session.commit()
