from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.modules.users.models import User


async def get_agent_user_ids(session: AsyncSession, user_id: UUID) -> list[UUID]:
    """Ids of the agents ``user_id`` created, and of the agents those created in turn.

    Parents come before their own agents. Runs on the caller's session so that
    a caller holding a transaction open does not need a second connection.
    """
    agent_ids = []
    parent_ids = [user_id]
    while parent_ids:
        parent_ids = list(
            (await session.execute(select(User.id).where(User.parent_user_id.in_(parent_ids))))
            .scalars()
            .all()
        )
        agent_ids.extend(parent_ids)
    return agent_ids
