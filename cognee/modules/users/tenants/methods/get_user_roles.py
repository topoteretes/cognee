from sqlalchemy import select
from uuid import UUID

from cognee.modules.users.exceptions import UserNotFoundError
from cognee.modules.users.models import Role, User, UserRole, UserTenant
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.permissions.methods.has_user_management_permission import (
    has_user_management_permission,
)


async def get_user_roles(tenant_id: UUID, user_id: UUID, user: User):
    # TODO: Consider if regular users should be able to see what roles does a user have
    await has_user_management_permission(user.id, tenant_id)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        tenant_membership_result = await session.execute(
            select(UserTenant).where(
                UserTenant.user_id == user_id,
                UserTenant.tenant_id == tenant_id,
            )
        )
        tenant_membership = tenant_membership_result.scalars().first()

        if tenant_membership is None:
            raise UserNotFoundError(message="User not found in tenant.")

        role_results = await session.execute(
            select(Role)
            .join(UserRole, Role.id == UserRole.role_id)
            .where(UserRole.user_id == user_id, Role.tenant_id == tenant_id)
        )
        roles = role_results.scalars().all()

        # Format response
        role_list = []
        for role in roles:
            role_list.append(
                {
                    "id": str(role.id),
                    "name": role.name,
                }
            )

    return role_list
