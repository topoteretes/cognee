from uuid import UUID
from sqlalchemy.future import select
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import (
    RoleNotFoundError,
)
from cognee.modules.users.models import (
    Permission,
    Role,
    RoleDefaultPermissions,
)


async def give_default_permission_to_role(role_id: UUID, permission_name: str):
    """
        Give the permission with given name to the role with the given id as a default permission.
    Args:
        role_id: Id of the role
        permission_name: Name of the permission

    Returns:
        None
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
            stmt = insert(Permission).values(name=permission_name)
            await session.execute(stmt)
            permission_entity = (
                (
                    await session.execute(
                        select(Permission).where(Permission.name == permission_name)
                    )
                )
                .scalars()
                .first()
            )

        try:
            # add default permission to role
            await session.execute(
                insert(RoleDefaultPermissions).values(
                    role_id=role.id, permission_id=permission_entity.id
                )
            )
        except IntegrityError:
            raise EntityAlreadyExistsError(message="Role permission already exists.")

        await session.commit()
