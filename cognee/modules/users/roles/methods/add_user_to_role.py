from uuid import UUID

from sqlalchemy.future import select
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from cognee.infrastructure.databases.exceptions import EntityAlreadyExistsError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import (
    UserNotFoundError,
    RoleNotFoundError,
)
from cognee.modules.users.models import (
    User,
    Role,
    UserRole,
)


async def add_user_to_role(user_id: UUID, role_id: UUID):
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        user = (await session.execute(select(User).where(User.id == user_id))).scalars().first()
        role = (await session.execute(select(Role).where(Role.id == role_id))).scalars().first()

        if not user:
            raise UserNotFoundError
        elif not role:
            raise RoleNotFoundError

        try:
            # Add association directly to the association table
            create_user_role_statement = insert(UserRole).values(user_id=user_id, role_id=role_id)
            await session.execute(create_user_role_statement)
        except IntegrityError:
            raise EntityAlreadyExistsError(message="User is already part of group.")

        await session.commit()
