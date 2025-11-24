from uuid import UUID
from sqlalchemy.future import select
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import (
    UserNotFoundError,
)
from cognee.modules.users.models import (
    Permission,
    User,
    UserDefaultPermissions,
)


async def give_default_permission_to_user(user_id: UUID, permission_name: str):
    """
        Give the permission with given name to the user with the given id as a default permission.
    Args:
        user_id: Id of the tenant
        permission_name: Name of the permission

    Returns:
        None
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        user = (await session.execute(select(User).where(User.id == user_id))).scalars().first()

        if not user:
            raise UserNotFoundError

        permission_entity = (
            (await session.execute(select(Permission).where(Permission.name == permission_name)))
            .scalars()
            .first()
        )

        if not permission_entity:
            create_permission_statement = insert(Permission).values(name=permission_name)
            await session.execute(create_permission_statement)
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
            # add default permission to user
            await session.execute(
                insert(UserDefaultPermissions).values(
                    user_id=user.id, permission_id=permission_entity.id
                )
            )
        except IntegrityError:
            raise EntityAlreadyExistsError(message="User permission already exists.")

        await session.commit()
