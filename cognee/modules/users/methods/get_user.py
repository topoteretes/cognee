from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import joinedload
import sqlalchemy.exc
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from ..models import User


async def get_user(user_id: UUID):
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        try:
            user = (
                await session.execute(
                    select(User)
                    .options(joinedload(User.roles), joinedload(User.tenant))
                    .where(User.id == user_id)
                )
            ).scalar()

            if not user:
                raise EntityNotFoundError(message=f"Could not find user: {user_id}")

            return user
        except sqlalchemy.exc.NoResultFound:
            raise EntityNotFoundError(message=f"Could not find user: {user_id}")
