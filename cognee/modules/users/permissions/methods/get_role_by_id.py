import sqlalchemy.exc
from sqlalchemy import select
from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import RoleNotFoundError

from ...models.Role import Role


async def get_role_by_id(role_id: UUID):
    """
        Return the role with the given id.

        Companion to get_role, which looks a role up by (tenant_id, name). Use
        this one when the id arrives from a request and the tenant it belongs to
        is what you still have to establish, as authorization does.
    Args:
        role_id: Id of the role.

    Returns
        The role with the given id.

    Raises:
        RoleNotFoundError: If no role with that id exists.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        try:
            result = await session.execute(select(Role).where(Role.id == role_id))
            return result.unique().scalar_one()
        except sqlalchemy.exc.NoResultFound:
            raise RoleNotFoundError(message=f"Could not find role: {role_id}")
