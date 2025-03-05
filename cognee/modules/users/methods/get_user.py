from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import User


async def get_user(user_id: UUID):
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        user = (
            await session.execute(
                select(User)
                .options(joinedload(User.roles), joinedload(User.tenant))
                .where(User.id == user_id)
            )
        ).scalar()

        return user
