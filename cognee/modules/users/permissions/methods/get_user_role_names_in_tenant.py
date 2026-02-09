from typing import List
from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import Role, UserRole


async def get_user_role_names_in_tenant(user_id: UUID, tenant_id: UUID) -> List[str]:
    """
    Return the names of all roles the user has in the given tenant.

    Args:
        user_id: Id of the user.
        tenant_id: Id of the tenant.

    Returns:
        List of role names (e.g. ["tenant_admin", "member"]). Empty if the user
        has no roles in the tenant.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(Role.name)
            .join(UserRole, Role.id == UserRole.role_id)
            .where(UserRole.user_id == user_id, Role.tenant_id == tenant_id)
        )
        return [row[0] for row in result.all()]
