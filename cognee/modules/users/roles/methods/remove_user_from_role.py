from uuid import UUID
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.users.permissions.methods.has_user_management_permission import (
    has_user_management_permission,
)
from cognee.modules.users.exceptions import (
    UserNotFoundError,
    RoleNotFoundError,
    TenantNotFoundError,
    PermissionDeniedError,
)
from cognee.modules.users.models import User, Role, Tenant, UserRole


async def remove_user_from_role(
    tenant_id: UUID,
    user_id: UUID,
    role_id: UUID,
    requester_id: UUID,
) -> None:
    """
    Remove a role assignment from a user within a specific tenant.

    This function ensures:
        - The tenant exists.
        - The requester has permission to manage users in the tenant.
        - The user exists and belongs to the tenant.
        - The role exists and belongs to the tenant.
        - The user-role association exists before attempting deletion.

    Args:
        tenant_id: UUID of the tenant where the operation is performed.
        user_id: UUID of the user to remove from the role.
        role_id: UUID of the role to be removed.
        requester_id: UUID of the user performing the action (must be tenant owner).

    Raises:
        TenantNotFoundError: If the tenant does not exist.
        PermissionDeniedError: If the requester is not authorized.
        UserNotFoundError: If the user does not exist or is not part of the tenant.
        RoleNotFoundError: If the role does not exist in the tenant.
        EntityNotFoundError: If the user is not assigned to the specified role.

    Returns:
        None. The operation completes silently on success.
    """

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:  # type: AsyncSession

        tenant = (
            await session.execute(
                select(Tenant).where(Tenant.id == tenant_id)
            )
        ).scalars().first()

        if not tenant:
            raise TenantNotFoundError

        await has_user_management_permission(requester_id, tenant_id)

        user = (
            await session.execute(
                select(User).where(User.id == user_id)
            )
        ).scalars().first()

        if not user:
            raise UserNotFoundError

        user_tenants = await user.awaitable_attrs.tenants
        if tenant_id not in [t.id for t in user_tenants]:
            raise UserNotFoundError

        role = (
            await session.execute(
                select(Role).where(
                    Role.id == role_id,
                    Role.tenant_id == tenant_id,
                )
            )
        ).scalars().first()

        if not role:
            raise RoleNotFoundError

        result = await session.execute(
            delete(UserRole).where(
                UserRole.user_id == user_id,
                UserRole.role_id == role_id,
            )
        )

        if result.rowcount == 0:
            raise EntityNotFoundError(
                message="User is not assigned to this role."
            )

        await session.commit()
