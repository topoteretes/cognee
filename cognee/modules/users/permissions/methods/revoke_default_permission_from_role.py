from uuid import UUID
from sqlalchemy.future import select
from sqlalchemy import delete

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import (
    PermissionNotFoundError,
    RoleNotFoundError,
)
from cognee.modules.users.models import (
    Permission,
    Role,
    RoleDefaultPermissions,
)


async def revoke_default_permission_from_role(role_id: UUID, permission_name: str):
    """
        Take the named default permission away from the role with the given id.

        Revoking is idempotent: a permission the role never had leaves the row set
        unchanged rather than raising, so callers can retry safely.
    Args:
        role_id: Id of the role
        permission_name: Name of the permission

    Returns:
        None

    Raises:
        RoleNotFoundError: If the role does not exist.
        PermissionNotFoundError: If no permission with that name exists.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        role = (await session.execute(select(Role).where(Role.id == role_id))).scalars().first()

        if not role:
            raise RoleNotFoundError

        permission_entity = (
            (await session.execute(select(Permission).where(Permission.name == permission_name)))
            .scalars()
            .first()
        )

        if not permission_entity:
            raise PermissionNotFoundError(message=f"{permission_name} not found")

        await session.execute(
            delete(RoleDefaultPermissions).where(
                RoleDefaultPermissions.role_id == role.id,
                RoleDefaultPermissions.permission_id == permission_entity.id,
            )
        )

        await session.commit()
