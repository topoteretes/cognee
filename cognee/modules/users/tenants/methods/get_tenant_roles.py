from sqlalchemy import select
from sqlalchemy.orm import selectinload
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import Role, UserRole
from cognee.modules.users.permissions.methods.has_user_management_permission import (
    has_user_management_permission,
)


async def get_tenant_roles(tenant_id: UUID, user):
    """
    List roles in a tenant.

    Callers with user management permission (owner or admin) see every role in
    the tenant. Everyone else sees only the roles they are a member of.
    """
    try:
        await has_user_management_permission(user.id, tenant_id)
        can_manage_users = True
    except PermissionDeniedError:
        can_manage_users = False

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        query = select(Role).options(selectinload(Role.users)).where(Role.tenant_id == tenant_id)

        if not can_manage_users:
            query = query.join(UserRole, Role.id == UserRole.role_id).where(
                UserRole.user_id == user.id
            )

        roles_result = await session.execute(query)
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
