import sqlalchemy.exc
from sqlalchemy import select
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import RoleNotFoundError

from ...models.Role import Role


async def get_role(tenant_id: UUID, role_name: str):
    """
        Return the role with the name role_name of the given tenant.
    Args:
        tenant_id: Id of the given tenant
        role_name: Name of the role

    Returns
        The role for the given tenant.

    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        try:
            result = await session.execute(
                select(Role).where(Role.name == role_name).where(Role.tenant_id == tenant_id)
            )
            role = result.unique().scalar_one()
            if not role:
                raise RoleNotFoundError(message=f"Could not find {role_name} for given tenant")
            return role
        except sqlalchemy.exc.NoResultFound:
            raise RoleNotFoundError(message=f"Could not find {role_name} for given tenant")
