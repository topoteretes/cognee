from cognee.modules.users.models import User
from cognee.infrastructure.databases.relational import get_relational_engine

from sqlalchemy.future import select

async def get_default_user():
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        stmt = select(User).where(User.email == "default_user@example.com")
        result = await session.execute(stmt)
        user = result.scalars().first()
    return user