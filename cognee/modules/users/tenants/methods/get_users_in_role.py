from sqlalchemy import select
from sqlalchemy.orm import selectinload
from uuid import UUID

from cognee.modules.users.models import User
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.permissions.methods.has_user_management_permission import (
    has_user_management_permission,
)


async def get_users_in_role(tenant_id: UUID, role_id: UUID, user: User):
    # TODO: Consider if regular users should be able to see the list of users in roles
    await has_user_management_permission(user.id, tenant_id)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        from cognee.modules.users.models import Role

        user_results = await session.execute(
            select(User).join(User.roles).where(Role.id == role_id)
        )
        users = user_results.scalars().all()

        # Format response
        user_list = []
        for user in users:
            user_list.append(
                {
                    "id": str(user.id),
                    "name": user.email,
                }
            )

    return user_list
