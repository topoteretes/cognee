from sqlalchemy import select
from uuid import UUID

from cognee.modules.users.exceptions import RoleNotFoundError
from cognee.modules.users.models import Role, User, UserRole
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.permissions.methods.has_user_management_permission import (
    has_user_management_permission,
)


async def get_users_in_role(tenant_id: UUID, role_id: UUID, user: User):
    # TODO: Consider if regular users should be able to see the list of users in roles
    await has_user_management_permission(user.id, tenant_id)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # Scope the role to the tenant the caller was authorized for, so a role id
        # belonging to another tenant cannot be used to read that tenant's members.
        role = (
            await session.execute(
                select(Role).where(Role.id == role_id, Role.tenant_id == tenant_id)
            )
        ).scalar_one_or_none()

        if role is None:
            raise RoleNotFoundError(message="Role not found for this tenant.")

        member_results = await session.execute(
            select(User)
            .join(UserRole, User.id == UserRole.user_id)
            .where(UserRole.role_id == role_id)
        )
        members = member_results.scalars().all()

        # Format response
        user_list = []
        for member in members:
            user_list.append(
                {
                    "id": str(member.id),
                    "name": member.email,
                }
            )

    return user_list
