from sqlalchemy.orm import joinedload

from cognee.modules.users.models import User
from cognee.infrastructure.databases.relational import get_relational_engine

from sqlalchemy.future import select

async def get_default_user(session):
    stmt = select(User).options(joinedload(User.groups)).where(User.email == "default_user@example.com")
    result = await session.execute(stmt)
    user = result.scalars().first()
    return user