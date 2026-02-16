from sqlalchemy import select
from sqlalchemy.orm import selectinload
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.permissions.methods.has_user_management_permission import (
    has_user_management_permission,
)


async def get_tenant_roles(tenant_id: UUID, user):
    # Check permissions - only tenant owner or users with specific roles (e.g. admin) can list roles in this tenant
    # TODO: Consider if regular users should be able to see the list of roles with info
    await has_user_management_permission(user.id, tenant_id)

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        from cognee.modules.users.models import Role

        roles_result = await session.execute(
            select(Role).options(selectinload(Role.users)).where(Role.tenant_id == tenant_id)
        )
        roles = roles_result.scalars().all()

        # Format response
        role_list = []
        for role in roles:
            role_list.append(
                {
                    "id": str(role.id),
                    "name": role.name,
                    "description": getattr(role, "description", None),
                    "user_count": len(role.users) if role.users else 0,
                }
            )

    return role_list
