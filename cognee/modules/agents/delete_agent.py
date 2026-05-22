from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models.User import User


async def delete_agent(agent_id: UUID, owner_id: UUID) -> None:
    relational_engine = get_relational_engine()

    async with relational_engine.get_async_session() as session:
        agent_user = (
            await session.execute(select(User).filter_by(id=agent_id))
        ).scalar_one_or_none()

        if agent_user is None or agent_user.parent_user_id != owner_id:
            raise PermissionError("Not authorized to delete this agent")

        await session.delete(agent_user)
        await session.commit()
