from sqlalchemy.orm import joinedload
from sqlalchemy.future import select
from cognee.modules.users.models import User
from cognee.infrastructure.databases.relational import get_relational_engine
from .create_default_user import create_default_user


async def get_default_user():
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = (
            select(User)
            .options(joinedload(User.groups))
            .where(User.email == "default_user@example.com")
        )

        result = await session.execute(query)
        user = result.scalars().first()

        if user is None:
            return await create_default_user()

        return user
