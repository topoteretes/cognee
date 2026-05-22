from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.User import User
from sqlalchemy import select


async def list_agents(owner_id: UUID) -> list[User]:
    """Return all agent users owned by the given user."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(select(User).where(User.parent_user_id == owner_id))
        return result.scalars().all()
