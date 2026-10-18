from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import User


async def get_user_agents(user_id: UUID) -> list[User]:
    """The agents ``user_id`` created: the users whose parent is ``user_id``."""
    async with get_relational_engine().get_async_session() as session:
        return list(
            (await session.execute(select(User).where(User.parent_user_id == user_id))).scalars()
        )
